"""Collect approved DeepSC-MIMO diagnosis profiles without target leakage."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile

import numpy as np
import torch

from channels import MIMOChannel, MIMOConfig
from main import resolve_vocab, source_identity, validate_vocabulary
from models.transceiver import DeepSC
from profiles import DEFAULT_OP_WEIGHTS, corrupt_sequence
from utils import PowerNormalize, SNR_to_noise, subsequent_mask


APPROVED_PROFILES = (
    "normal", "low_snr", "high_snr_control", "imperfect_csi",
    "spatial_correlation", "literal", "mismatch", "ood_hansard", "ood_arxiv",
)
PROFILE_LABELS = {
    "normal": "normal", "low_snr": "low_snr", "high_snr_control": "control",
    "imperfect_csi": "imperfect_csi", "spatial_correlation": "spatial_correlation",
    "literal": "literal", "mismatch": "mismatch", "ood_hansard": "ood",
    "ood_arxiv": "HOLDOUT",
}
PROFILE_DOMAINS = {
    "normal": "europarl", "low_snr": "europarl",
    "high_snr_control": "europarl", "imperfect_csi": "europarl",
    "spatial_correlation": "europarl", "literal": "europarl",
    "mismatch": "europarl", "ood_hansard": "ukhansard", "ood_arxiv": "arxiv",
}
PROFILE_SNRS = {"normal": 10.0, "low_snr": 1.0, "high_snr_control": 19.0}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", required=True, choices=APPROVED_PROFILES)
    result.add_argument("--severity", default=None, type=float)
    result.add_argument("--physical-seed", required=True, type=int)
    result.add_argument("--seed-index", required=True, type=int, choices=range(5))
    result.add_argument("--selection-seed", default=20260823, type=int)
    result.add_argument("--corruption-seed", default=7301, type=int)
    result.add_argument("--batch-size", default=128, type=int)
    result.add_argument("--batches", default=10, type=int)
    result.add_argument("--max-generation-length", default=31, type=int)
    result.add_argument("--shared-data-root", required=True)
    result.add_argument("--vocab-file", default=None)
    result.add_argument("--checkpoint-c", required=True)
    result.add_argument("--checkpoint-d", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--allow-nonshared-vocab", action="store_true")
    result.add_argument("--allow-incomplete-checkpoint", action="store_true")
    return result


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload, destination):
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_model(path, vocabulary_size, device, allow_incomplete=False):
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("collection requires a full DeepSC-MIMO checkpoint: {}".format(path))
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != 80 and not allow_incomplete:
        raise ValueError("canonical collection requires epoch-80 last.pth; got epoch {}".format(epoch))
    config = checkpoint.get("config", {})
    validation_loss = float(checkpoint.get("validation_loss", float("nan")))
    if not math.isfinite(validation_loss):
        raise ValueError("checkpoint validation loss is not finite: {}".format(path))
    model = DeepSC(
        int(config.get("num_layers", 4)), vocabulary_size, vocabulary_size,
        vocabulary_size, vocabulary_size, int(config.get("d_model", 128)),
        int(config.get("num_heads", 8)), int(config.get("dff", 512)),
        float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, {
        "path": str(path), "sha256": sha256_file(path), "epoch": epoch,
        "validation_loss": validation_loss,
        "architecture": {
            key: config.get(key) for key in (
                "num_layers", "d_model", "num_heads", "dff", "dropout"
            )
        },
    }


def strip_padding(sequence, padding_idx):
    result = [int(value) for value in sequence]
    while result and result[-1] == padding_idx:
        result.pop()
    return result


def pad_sequences(sequences, padding_idx):
    maximum = max(map(len, sequences))
    array = np.full((len(sequences), maximum), padding_idx, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        array[row, :len(sequence)] = sequence
    return torch.from_numpy(array)


def greedy_hidden_states(model, memory, source_mask, start_idx, end_idx, pad_idx, max_len):
    """Free-running R3: final decoder layer's last position before projection."""
    batch = memory.size(0)
    generated = torch.full(
        (batch, 1), start_idx, dtype=torch.long, device=memory.device
    )
    active = torch.ones(batch, dtype=torch.bool, device=memory.device)
    lengths = torch.zeros(batch, dtype=torch.long, device=memory.device)
    hidden_steps = []
    predicted_steps = []
    for step in range(max_len - 1):
        target_padding = (generated == pad_idx).unsqueeze(-2)
        target_mask = torch.logical_or(
            target_padding, subsequent_mask(generated.size(1), device=generated.device)
        )
        decoded = model.decoder(generated, memory, target_mask, source_mask)
        hidden = decoded[:, -1, :]
        next_token = model.dense(hidden).argmax(dim=-1)
        hidden_steps.append(hidden * active.unsqueeze(-1))
        predicted_steps.append(torch.where(active, next_token, torch.full_like(next_token, pad_idx)))
        just_finished = active & (next_token == end_idx)
        lengths[just_finished] = step + 1
        active = active & ~just_finished
        append_token = torch.where(active | just_finished, next_token, torch.full_like(next_token, pad_idx))
        generated = torch.cat((generated, append_token[:, None]), dim=1)
        if not bool(active.any()):
            break
    lengths[active] = len(hidden_steps)
    hidden_states = torch.stack(hidden_steps, dim=1)
    predicted_tokens = torch.stack(predicted_steps, dim=1)
    mask = torch.arange(hidden_states.size(1), device=memory.device)[None, :] < lengths[:, None]
    return hidden_states, predicted_tokens, lengths, mask


def profile_parameters(profile, severity):
    snr_db = PROFILE_SNRS.get(profile, 10.0)
    kwargs = {}
    severity_name = None
    if profile == "imperfect_csi":
        if severity not in {0.99, 0.95, 0.90}:
            raise ValueError("imperfect_csi severity must be one of 0.99, 0.95, 0.90")
        kwargs = {"profile": "imperfect_csi", "csi_rho": severity}
        severity_name = "csi_rho"
    elif profile == "spatial_correlation":
        if severity not in {0.3, 0.6, 0.9}:
            raise ValueError("spatial_correlation severity must be one of 0.3, 0.6, 0.9")
        kwargs = {"profile": "spatial_correlation", "spatial_rho": severity}
        severity_name = "spatial_rho"
    else:
        kwargs = {"profile": "baseline"}
        if profile == "literal":
            severity = 0.20 if severity is None else severity
            if severity != 0.20:
                raise ValueError("approved literal severity is exactly 0.20")
            severity_name = "edit_rate"
        elif severity is not None:
            raise ValueError("--severity is not used by profile {}".format(profile))
    config = MIMOConfig(fading_scope="batch", equalizer="lmmse", **kwargs)
    config.validate()
    return config, snr_db, severity_name


def main(argv=None):
    args = parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA collection requested but CUDA is unavailable")
    if args.batch_size <= 0 or args.batches <= 0:
        raise ValueError("batch size and batches must be positive")
    device = torch.device(args.device)
    shared_root = Path(args.shared_data_root).expanduser().resolve()
    vocab_path = resolve_vocab(args.vocab_file, shared_root)
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    token_to_idx = validate_vocabulary(
        vocabulary, allow_nonshared=args.allow_nonshared_vocab
    )
    idx_to_token = {int(index): token for token, index in token_to_idx.items()}
    pad, start, end = (
        int(token_to_idx["<PAD>"]), int(token_to_idx["<START>"]),
        int(token_to_idx["<END>"]),
    )
    domain = PROFILE_DOMAINS[args.profile]
    dataset_path = shared_root / domain / "test_data.pkl"
    with dataset_path.open("rb") as handle:
        data = pickle.load(handle)
    needed = args.batch_size * args.batches
    if needed * 5 > len(data):
        raise ValueError("{} needs at least {} samples for five disjoint blocks".format(domain, needed * 5))
    indices = list(range(len(data)))
    domain_offset = {"europarl": 0, "ukhansard": 100, "arxiv": 200}[domain]
    random.Random(args.selection_seed + domain_offset).shuffle(indices)
    lower = args.seed_index * needed
    chosen = indices[lower:lower + needed]

    tx_model, checkpoint_c = checkpoint_model(
        args.checkpoint_c, len(token_to_idx), device, args.allow_incomplete_checkpoint
    )
    if args.profile == "mismatch":
        rx_model, checkpoint_d = checkpoint_model(
            args.checkpoint_d, len(token_to_idx), device, args.allow_incomplete_checkpoint
        )
    else:
        rx_model, checkpoint_d = tx_model, None
    config, snr_db, severity_name = profile_parameters(args.profile, args.severity)
    channel = MIMOChannel(config, seed=args.physical_seed)
    channel.reset(realization_set=0)

    severity_text = "none" if args.severity is None else ("{:g}".format(args.severity).replace(".", "p"))
    run_id = "seed{}_{}_{}".format(args.physical_seed, args.profile, severity_text)
    run_dir = Path(args.output_root).expanduser().resolve() / args.profile / run_id
    if run_dir.exists():
        raise FileExistsError("refusing to overwrite immutable run {}".format(run_dir))
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    configuration = {
        "schema_version": 3,
        "profile": args.profile,
        "diagnosis_label": PROFILE_LABELS[args.profile],
        "holdout": args.profile == "ood_arxiv",
        "domain": domain,
        "severity_name": severity_name,
        "severity": args.severity,
        "snr_db": snr_db,
        "physical_seed": args.physical_seed,
        "seed_index": args.seed_index,
        "selection_seed": args.selection_seed,
        "channel": vars(config),
        "checkpoint_c": checkpoint_c,
        "checkpoint_d": checkpoint_d,
        "vocab_path": str(vocab_path),
        "vocab_sha256": sha256_file(vocab_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "batch_size": args.batch_size,
        "batches": args.batches,
        "source_snapshot": source_identity(),
        "r3_definition": "free-running final decoder-layer last-position hidden state before vocabulary projection; includes END-predicting state",
        "literal_operation_weights": DEFAULT_OP_WEIGHTS if args.profile == "literal" else None,
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    configuration["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (run_dir / "metadata.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = run_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            "batch_index", "artifact", "first_sample_id", "last_sample_id",
            "physical_seed", "source_max_length", "r3_max_length", "literal_events",
        ])

    with torch.no_grad():
        for batch_index in range(args.batches):
            batch_ids = chosen[
                batch_index * args.batch_size:(batch_index + 1) * args.batch_size
            ]
            clean = [strip_padding(data[index], pad) for index in batch_ids]
            source = clean
            corruption_metadata = []
            if args.profile == "literal":
                source = []
                for sample_id, sequence in zip(batch_ids, clean):
                    rng = random.Random(args.corruption_seed * 1_000_003 + sample_id)
                    corrupted, metadata = corrupt_sequence(
                        sequence, idx_to_token, token_to_idx, rng, severity=0.20
                    )
                    source.append(corrupted)
                    corruption_metadata.append(metadata)
            source_tensor = pad_sequences(source, pad).to(device)
            clean_tensor = pad_sequences(clean, pad)
            source_mask = (source_tensor == pad).unsqueeze(-2)
            semantic = tx_model.encoder(source_tensor, source_mask)
            channel_raw = tx_model.channel_encoder(semantic)
            transmitted = PowerNormalize(channel_raw)
            received, channel_trace = channel(
                transmitted, SNR_to_noise(snr_db), return_trace=True
            )
            memory = rx_model.channel_decoder(received)
            r3, generated, generated_lengths, generated_mask = greedy_hidden_states(
                rx_model, memory, source_mask, start, end, pad,
                args.max_generation_length,
            )
            artifact = {
                "schema_version": 3,
                "profile": args.profile,
                "diagnosis_label": PROFILE_LABELS[args.profile],
                "holdout": args.profile == "ood_arxiv",
                "domain": domain,
                "severity_name": severity_name,
                "severity": args.severity,
                "snr_db": snr_db,
                "physical_seed": args.physical_seed,
                "batch_index": batch_index,
                "sample_ids": torch.tensor(batch_ids, dtype=torch.long),
                "source_tokens": source_tensor.cpu(),
                "clean_reference_tokens": clean_tensor,
                "source_lengths": (source_tensor != pad).sum(dim=1).cpu(),
                "clean_reference_lengths": (clean_tensor != pad).sum(dim=1),
                "corruption_metadata": corruption_metadata or None,
                "R0_noisy_received": channel_trace.noisy_signal.cpu(),
                "R1_equalized_decoder_input": received.cpu(),
                "R2_channel_decoder_output": memory.cpu(),
                "R3_free_running_decoder_hidden": r3.cpu(),
                "R3_generated_tokens": generated.cpu(),
                "R3_generated_lengths": generated_lengths.cpu(),
                "R3_generated_mask": generated_mask.cpu(),
                "true_h": channel_trace.true_h.cpu(),
                "underlying_iid_h": channel_trace.underlying_iid_h.cpu(),
                "estimated_h": channel_trace.estimated_h.cpu(),
                "faded_signal": channel_trace.faded_signal.cpu(),
                "standardized_noise": (
                    channel_trace.noise / max(channel_trace.noise_std, 1e-30)
                ).cpu(),
                "noise": channel_trace.noise.cpu(),
                "config_sha256": configuration["config_sha256"],
                "tx_checkpoint_sha256": checkpoint_c["sha256"],
                "rx_checkpoint_sha256": (
                    checkpoint_d["sha256"] if checkpoint_d else checkpoint_c["sha256"]
                ),
            }
            destination = artifact_dir / "batch_{:03d}.pt".format(batch_index)
            atomic_torch_save(artifact, destination)
            literal_events = sum(
                int(item.get("events_applied", 0)) for item in corruption_metadata
            )
            with manifest_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([
                    batch_index, str(destination.relative_to(run_dir)), batch_ids[0],
                    batch_ids[-1], args.physical_seed, source_tensor.size(1),
                    r3.size(1), literal_events,
                ])
            print("saved {}".format(destination), flush=True)
    print("COLLECTION COMPLETE: {}".format(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
