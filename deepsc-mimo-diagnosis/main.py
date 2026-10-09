"""Train DeepSC over a configurable 2x2 MIMO channel or collect traces."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from channels import MIMOChannel, MIMOConfig
from dataset import EurDataset, collate_data, collate_indexed_data, resolve_data_file
from experiment_data import ExperimentDataWriter
from models.transceiver import DeepSC
from utils import SNR_to_noise, initNetParams, train_step, val_step


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_SHARED_VOCAB_SIZE = 90_746
EXPECTED_SPECIAL_TOKEN_IDS = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<UNK>": 3,
}
VALIDATION_REALIZATION_SET = 1_000_000


def build_parser():
    parser = argparse.ArgumentParser(
        description="DeepSC with native-complex 2x2 MIMO and trace collection"
    )
    parser.add_argument("--data-root", default=None,
                        help="Directory containing train_data.pkl/test_data.pkl, or europarl/")
    parser.add_argument("--vocab-file", default=None,
                        help="Explicit shared_vocab.json; auto-detected from --data-root if omitted")
    parser.add_argument(
        "--allow-nonshared-vocab",
        action="store_true",
        help="Permit a vocabulary other than the approved shared 90,746-token vocabulary (tests only)",
    )
    parser.add_argument("--checkpoint-path", default="checkpoints/deepsc-mimo")
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--train-snr-min", default=5.0, type=float)
    parser.add_argument("--train-snr-max", default=10.0, type=float)
    parser.add_argument(
        "--train-noise-sampling",
        choices=("legacy_epoch", "snr_per_batch"),
        default="legacy_epoch",
    )
    parser.add_argument("--validation-noise-std", default=0.1, type=float)
    parser.add_argument("--validation-snr-db", default=None, type=float)
    parser.add_argument("--loss-normalization", choices=("legacy", "tokens"), default="legacy")
    parser.add_argument("--shuffle-training", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--channel-seed", default=1234, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-train-batches", default=None, type=int)
    parser.add_argument("--max-validation-batches", default=None, type=int)

    parser.add_argument("--nt", default=2, type=int)
    parser.add_argument("--nr", default=2, type=int)
    parser.add_argument("--fading-scope", choices=("batch", "sample", "use"), default="batch")
    parser.add_argument(
        "--profile",
        choices=("baseline", "imperfect_csi", "spatial_correlation", "no_fading", "bad_channel", "noisy_csi", "stale_csi"),
        default="baseline",
    )
    parser.add_argument("--equalizer", choices=("lmmse", "zf"), default="lmmse")
    parser.add_argument("--covariance", choices=("system",), default="system")
    parser.add_argument("--stream-variance", default=None, type=float)
    parser.add_argument("--no-total-power-constraint", action="store_true")
    parser.add_argument("--channel-gain", default=1.0, type=float)
    parser.add_argument("--csi-noise-std", default=0.0, type=float)
    parser.add_argument("--csi-rho", default=1.0, type=float)
    parser.add_argument("--spatial-rho", default=0.0, type=float)
    parser.add_argument("--stale-rho", default=0.95, type=float)
    parser.add_argument("--covariance-floor", default=1e-6, type=float)
    parser.add_argument("--allow-legacy-profiles", action="store_true")

    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--collection-split", choices=("train", "test"), default="test")
    parser.add_argument("--collection-snr-db", default=10.0, type=float)
    parser.add_argument("--collect-every-n-batches", default=1, type=int)
    parser.add_argument("--max-collected-batches", default=None, type=int)
    parser.add_argument("--output-root", default="data/mimo_profiles")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--logits-mode", choices=("full", "topk", "none"), default="topk")
    parser.add_argument("--logits-top-k", default=5, type=int)
    parser.add_argument("--allow-untrained-collection", action="store_true")
    parser.add_argument("--allow-nonempty-checkpoint-dir", action="store_true")
    parser.add_argument("--skip-dataset-hash", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_vocab(path=None, data_root=None):
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    if data_root:
        root = Path(data_root).expanduser()
        candidates.extend((
            root / "shared_vocab.json",
            root.parent / "shared_vocab.json",
        ))
    candidates.extend((
        PROJECT_ROOT / "shared_vocab.json",
    ))
    for candidate in dict.fromkeys(item.resolve() for item in candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "A vocabulary file was not found. Pass the approved shared_vocab.json with "
        "--vocab-file or place it beside the domain data directory. Searched:\n  - {}"
        .format("\n  - ".join(str(item.resolve()) for item in candidates))
    )


def validate_vocabulary(vocabulary, allow_nonshared=False):
    token_to_idx = vocabulary.get("token_to_idx")
    if not isinstance(token_to_idx, dict):
        raise ValueError("vocabulary must contain a token_to_idx object")
    problems = []
    if len(token_to_idx) != EXPECTED_SHARED_VOCAB_SIZE:
        problems.append(
            "size is {}, expected {:,}".format(
                len(token_to_idx), EXPECTED_SHARED_VOCAB_SIZE
            )
        )
    for token, expected_id in EXPECTED_SPECIAL_TOKEN_IDS.items():
        actual_id = token_to_idx.get(token)
        if actual_id != expected_id:
            problems.append(
                "{} has ID {!r}, expected {}".format(token, actual_id, expected_id)
            )
    if problems and not allow_nonshared:
        raise ValueError(
            "Vocabulary is not the approved shared vocabulary: {}. "
            "Use --allow-nonshared-vocab only for synthetic tests."
            .format("; ".join(problems))
        )
    return token_to_idx


def channel_config(args):
    return MIMOConfig(
        nt=args.nt,
        nr=args.nr,
        fading_scope=args.fading_scope,
        profile=args.profile,
        equalizer=args.equalizer,
        covariance=args.covariance,
        stream_variance=args.stream_variance,
        total_power_constraint=not args.no_total_power_constraint,
        channel_gain=args.channel_gain,
        csi_noise_std=args.csi_noise_std,
        csi_rho=args.csi_rho,
        spatial_rho=args.spatial_rho,
        stale_rho=args.stale_rho,
        covariance_floor=args.covariance_floor,
        allow_legacy_profiles=args.allow_legacy_profiles,
    )


def build_model(args, vocabulary_size, device):
    model = DeepSC(
        args.num_layers, vocabulary_size, vocabulary_size,
        vocabulary_size, vocabulary_size, args.d_model,
        args.num_heads, args.dff, args.dropout,
    ).to(device)
    initNetParams(model)
    return model


def load_checkpoint(model, path, device):
    if not path:
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    print("Loaded checkpoint: {}".format(Path(path).resolve()))
    return checkpoint


def rng_state_dict():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        raise ValueError("resume checkpoint is missing rng_state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(..., map_location=cuda) also moves saved RNG ByteTensors to
    # CUDA; both RNG restoration APIs require their state tensors on the CPU.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if cuda_state is None:
            raise ValueError("CUDA resume requested but checkpoint has no CUDA RNG state")
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the resume checkpoint")
        torch.cuda.set_rng_state_all([generator_state.cpu() for generator_state in cuda_state])


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path, include_hash=False):
    path = Path(path).resolve()
    stat = path.stat()
    identity = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        identity["sha256"] = sha256_file(path)
    return identity


def source_identity():
    files = sorted(
        path for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".py", ".slurm", ".sh", ".csv", ".md"}
        and "__pycache__" not in path.parts
    )
    records = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest = sha256_file(path)
        records.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    return {"aggregate_sha256": aggregate.hexdigest(), "files": records}


def validation_noise_std(args):
    if args.validation_snr_db is not None:
        return SNR_to_noise(args.validation_snr_db)
    if args.validation_noise_std < 0:
        raise ValueError("--validation-noise-std must be non-negative")
    return args.validation_noise_std


def make_loader(args, split, indexed, device, shuffle=False):
    dataset = EurDataset(split, data_root=args.data_root, indexed=indexed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_indexed_data if indexed else collate_data,
    )


def validate(args, model, criterion, pad_idx, device, channel):
    loader = make_loader(args, "test", False, device)
    model.eval()
    # Keep validation fixed across epochs, but isolated from training streams.
    channel.reset(realization_set=VALIDATION_REALIZATION_SET)
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch_index, sents in enumerate(tqdm(loader, desc="validation")):
            if args.max_validation_batches is not None and batch_index >= args.max_validation_batches:
                break
            sents = sents.to(device)
            total += val_step(
                model, sents, sents, validation_noise_std(args),
                pad_idx, criterion, channel_model=channel,
                loss_normalization=args.loss_normalization,
            )
            batches += 1
    if not batches:
        raise RuntimeError("validation processed zero batches")
    return total / batches


def train_epoch(args, model, optimizer, criterion, pad_idx, device, channel, epoch):
    loader = make_loader(
        args, "train", False, device, shuffle=args.shuffle_training
    )
    model.train()
    # Fresh physical realizations each epoch, reproducible from epoch + base seed.
    channel.reset(realization_set=epoch)
    rng = np.random.default_rng(args.seed + epoch)
    legacy_noise_std = None
    if args.train_noise_sampling == "legacy_epoch":
        endpoints = (SNR_to_noise(args.train_snr_min), SNR_to_noise(args.train_snr_max))
        legacy_noise_std = rng.uniform(min(endpoints), max(endpoints))
    progress = tqdm(loader, desc="train epoch {}".format(epoch + 1))
    total = 0.0
    batches = 0
    for batch_index, sents in enumerate(progress):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        if args.train_noise_sampling == "legacy_epoch":
            noise_std = legacy_noise_std
            snr_db = -10.0 * np.log10(2.0 * noise_std ** 2)
        else:
            snr_db = rng.uniform(args.train_snr_min, args.train_snr_max)
            noise_std = SNR_to_noise(snr_db)
        sents = sents.to(device)
        loss = train_step(
            model, sents, sents, noise_std, pad_idx,
            optimizer, criterion, channel_model=channel,
            loss_normalization=args.loss_normalization,
        )
        total += loss
        batches += 1
        progress.set_postfix(loss="{:.5f}".format(loss), snr_db="{:.2f}".format(snr_db))
    if not batches:
        raise RuntimeError("training processed zero batches")
    return total / batches


def collect(args, model, criterion, pad_idx, device, channel, config, vocab_path):
    if not args.load_checkpoint and not args.allow_untrained_collection:
        raise ValueError(
            "Collection requires --load-checkpoint. Use --allow-untrained-collection only for a smoke test."
        )
    loader = make_loader(args, args.collection_split, True, device)
    data_path = resolve_data_file(args.collection_split, args.data_root)
    writer = ExperimentDataWriter(
        args.output_root,
        args.profile,
        run_id=args.run_id,
        metadata={
            "mode": "collect_only",
            "checkpoint": (
                file_identity(args.load_checkpoint, include_hash=True)
                if args.load_checkpoint else None
            ),
            "collection_split": args.collection_split,
            "snr_db": args.collection_snr_db,
            "mimo": asdict(config),
            "vocabulary": file_identity(vocab_path, include_hash=True),
            "dataset": file_identity(
                data_path, include_hash=not args.skip_dataset_hash
            ),
            "source": source_identity(),
            "data_root": args.data_root,
            "seed": args.seed,
            "channel_seed": args.channel_seed,
        },
        logits_mode=args.logits_mode,
        logits_top_k=args.logits_top_k,
    )
    model.eval()
    channel.reset(realization_set=0)
    saved = 0
    noise_std = SNR_to_noise(args.collection_snr_db)
    with torch.no_grad():
        for batch_index, (sents, sample_ids, lengths) in enumerate(tqdm(loader, desc="collect")):
            if batch_index % args.collect_every_n_batches:
                continue
            sents = sents.to(device)
            loss, trace = val_step(
                model, sents, sents, noise_std, pad_idx, criterion,
                collect_trace=True, channel_model=channel,
                loss_normalization=args.loss_normalization,
            )
            trace.sequence_lengths = lengths.to(device)
            writer.save_batch(
                trace, args.collection_split, 0, batch_index, saved,
                sample_ids=sample_ids, extra_metrics={"snr_db": args.collection_snr_db},
            )
            saved += 1
            if args.max_collected_batches is not None and saved >= args.max_collected_batches:
                break
    print("Saved {} batches to {}".format(saved, writer.run_dir))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.load_checkpoint and args.resume_checkpoint:
        raise ValueError("use only one of --load-checkpoint and --resume-checkpoint")
    if args.collect_every_n_batches <= 0:
        raise ValueError("--collect-every-n-batches must be positive")
    if args.train_snr_min > args.train_snr_max:
        raise ValueError("--train-snr-min cannot exceed --train-snr-max")
    setup_seed(args.seed)
    device = choose_device(args.device)
    config = channel_config(args)
    config.validate()
    vocab_path = resolve_vocab(args.vocab_file, args.data_root)
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    token_to_idx = validate_vocabulary(
        vocabulary, allow_nonshared=args.allow_nonshared_vocab
    )

    print("Project root:", PROJECT_ROOT)
    print("Device:", device)
    print("Vocabulary:", vocab_path)
    print("MIMO configuration:", json.dumps(asdict(config), sort_keys=True))
    if args.preflight_only:
        # Resolve both files without loading their potentially large contents.
        print("Train data:", resolve_data_file("train", args.data_root))
        print("Test data:", resolve_data_file("test", args.data_root))
        print("Preflight passed")
        return 0

    model = build_model(args, len(token_to_idx), device)
    loaded_checkpoint = load_checkpoint(
        model, args.resume_checkpoint or args.load_checkpoint, device
    )
    criterion = nn.CrossEntropyLoss(reduction="none")
    channel = MIMOChannel(config, seed=args.channel_seed)
    pad_idx = token_to_idx["<PAD>"]

    if args.collect_only:
        collect(args, model, criterion, pad_idx, device, channel, config, vocab_path)
        return 0

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98),
        eps=1e-8, weight_decay=args.weight_decay,
    )
    start_epoch = 0
    best_validation = float("inf")
    if args.resume_checkpoint:
        if not isinstance(loaded_checkpoint, dict) or "optimizer_state_dict" not in loaded_checkpoint:
            raise ValueError("--resume-checkpoint requires a full DeepSC-MIMO checkpoint")
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        restore_rng_state(loaded_checkpoint.get("rng_state"))
        start_epoch = int(loaded_checkpoint.get("epoch", 0))
        next_realization = int(
            loaded_checkpoint.get("channel_next_realization_set", -1)
        )
        if next_realization != start_epoch:
            raise ValueError(
                "resume checkpoint channel progression is inconsistent: {} != {}"
                .format(next_realization, start_epoch)
            )
        best_validation = float(
            loaded_checkpoint.get(
                "best_validation", loaded_checkpoint.get("validation_loss", float("inf"))
            )
        )
        if start_epoch >= args.epochs:
            raise ValueError(
                "resume checkpoint is already at epoch {}, but --epochs is {}"
                .format(start_epoch, args.epochs)
            )
    checkpoint_dir = Path(args.checkpoint_path).expanduser().resolve()
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        if not (args.allow_nonempty_checkpoint_dir or args.resume_checkpoint):
            raise FileExistsError(
                "Checkpoint directory is non-empty: {}. Choose a new directory or pass "
                "--allow-nonempty-checkpoint-dir explicitly.".format(checkpoint_dir)
            )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config["mimo"] = asdict(config)
    run_config["vocabulary"] = file_identity(vocab_path, include_hash=True)
    run_config["train_dataset"] = file_identity(
        resolve_data_file("train", args.data_root),
        include_hash=not args.skip_dataset_hash,
    )
    run_config["test_dataset"] = file_identity(
        resolve_data_file("test", args.data_root),
        include_hash=not args.skip_dataset_hash,
    )
    run_config["source"] = source_identity()
    (checkpoint_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    history_path = checkpoint_dir / "history.csv"
    if not (args.resume_checkpoint and history_path.is_file()):
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle,
                fieldnames=("epoch", "train_loss", "validation_loss", "seconds"),
            ).writeheader()

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_loss = train_epoch(
            args, model, optimizer, criterion, pad_idx, device, channel, epoch
        )
        validation_loss = validate(args, model, criterion, pad_idx, device, channel)
        elapsed = time.time() - started
        print(
            "Epoch {} train={:.6f} validation={:.6f} seconds={:.1f}".format(
                epoch + 1, train_loss, validation_loss, elapsed
            )
        )
        improved = validation_loss < best_validation
        if improved:
            best_validation = validation_loss
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_validation": best_validation,
            "rng_state": rng_state_dict(),
            "scheduler_state_dict": None,
            "channel_next_realization_set": epoch + 1,
            "config": run_config,
        }
        torch.save(state, checkpoint_dir / "last.pth")
        if improved:
            torch.save(state, checkpoint_dir / "best.pth")
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle,
                fieldnames=("epoch", "train_loss", "validation_loss", "seconds"),
            ).writerow({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "seconds": elapsed,
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
