"""Evaluate a trained DeepSC-MIMO checkpoint with BLEU-1 across SNR values."""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from channels import MIMOChannel, MIMOConfig
from dataset import EurDataset, collate_data
from main import (
    build_model, choose_device, load_checkpoint, resolve_vocab, setup_seed,
    validate_vocabulary,
)
from utils import BleuScore, SNR_to_noise, SeqtoText, greedy_decode


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", default=None)
    result.add_argument("--vocab-file", default=None)
    result.add_argument("--allow-nonshared-vocab", action="store_true")
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--batch-size", default=64, type=int)
    result.add_argument("--num-workers", default=0, type=int)
    result.add_argument("--max-length", default=30, type=int)
    result.add_argument("--repeats", default=1, type=int)
    result.add_argument("--snr-db", nargs="+", type=float, default=[0, 3, 6, 9, 12, 15, 18])
    result.add_argument("--max-batches", default=None, type=int)
    result.add_argument("--seed", default=42, type=int)
    result.add_argument("--channel-seed", default=1234, type=int)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--d-model", default=128, type=int)
    result.add_argument("--dff", default=512, type=int)
    result.add_argument("--num-layers", default=4, type=int)
    result.add_argument("--num-heads", default=8, type=int)
    result.add_argument("--dropout", default=0.1, type=float)
    result.add_argument("--nt", default=2, type=int)
    result.add_argument("--nr", default=2, type=int)
    result.add_argument("--fading-scope", choices=("batch", "sample", "use"), default="batch")
    result.add_argument("--profile", choices=("baseline", "imperfect_csi", "spatial_correlation", "no_fading", "bad_channel", "noisy_csi", "stale_csi"), default="baseline")
    result.add_argument("--equalizer", choices=("lmmse", "zf"), default="lmmse")
    result.add_argument("--covariance", choices=("system",), default="system")
    result.add_argument("--stream-variance", default=None, type=float)
    result.add_argument("--no-total-power-constraint", action="store_true")
    result.add_argument("--channel-gain", default=1.0, type=float)
    result.add_argument("--csi-noise-std", default=0.0, type=float)
    result.add_argument("--csi-rho", default=1.0, type=float)
    result.add_argument("--spatial-rho", default=0.0, type=float)
    result.add_argument("--stale-rho", default=0.95, type=float)
    result.add_argument("--covariance-floor", default=1e-6, type=float)
    result.add_argument("--allow-legacy-profiles", action="store_true")
    return result


def make_config(args):
    return MIMOConfig(
        nt=args.nt, nr=args.nr, fading_scope=args.fading_scope,
        profile=args.profile, equalizer=args.equalizer,
        covariance=args.covariance,
        stream_variance=args.stream_variance,
        total_power_constraint=not args.no_total_power_constraint,
        channel_gain=args.channel_gain, csi_noise_std=args.csi_noise_std,
        csi_rho=args.csi_rho, spatial_rho=args.spatial_rho,
        stale_rho=args.stale_rho, covariance_floor=args.covariance_floor,
        allow_legacy_profiles=args.allow_legacy_profiles,
    )


def evaluate(args, model, config, loader, converter, indices, device):
    metric = BleuScore(1, 0, 0, 0)
    repeat_scores = []
    model.eval()
    with torch.no_grad():
        for repeat in range(args.repeats):
            snr_scores = []
            for snr_db in args.snr_db:
                # SNR points within a repeat share standardized H/noise draws;
                # repeats use independent deterministic streams.
                channel = MIMOChannel(config, seed=args.channel_seed + repeat)
                predictions, references = [], []
                for batch_index, sentences in enumerate(tqdm(loader, desc="SNR {} dB".format(snr_db))):
                    if args.max_batches is not None and batch_index >= args.max_batches:
                        break
                    sentences = sentences.to(device)
                    output = greedy_decode(
                        model, sentences, SNR_to_noise(snr_db), args.max_length,
                        indices["pad"], indices["start"], channel_model=channel,
                    )
                    predictions.extend(map(converter.sequence_to_text, output.cpu().tolist()))
                    references.extend(map(converter.sequence_to_text, sentences.cpu().tolist()))
                scores = metric.compute_blue_score(references, predictions)
                snr_scores.append(float(np.mean(scores)) if scores else float("nan"))
            repeat_scores.append(snr_scores)
    return np.nanmean(np.asarray(repeat_scores), axis=0)


def main(argv=None):
    args = parser().parse_args(argv)
    setup_seed(args.seed)
    device = choose_device(args.device)
    vocab_path = resolve_vocab(args.vocab_file, args.data_root)
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    token_to_idx = validate_vocabulary(
        vocabulary, allow_nonshared=args.allow_nonshared_vocab
    )
    indices = {
        "pad": token_to_idx["<PAD>"],
        "start": token_to_idx["<START>"],
        "end": token_to_idx["<END>"],
    }
    model = build_model(args, len(token_to_idx), device)
    load_checkpoint(model, args.checkpoint, device)
    dataset = EurDataset("test", args.data_root)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", collate_fn=collate_data,
    )
    config = make_config(args)
    config.validate()
    converter = SeqtoText(token_to_idx, indices["end"])
    scores = evaluate(args, model, config, loader, converter, indices, device)
    for snr_db, score in zip(args.snr_db, scores):
        print("SNR {:g} dB: BLEU-1 {:.8f}".format(snr_db, score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
