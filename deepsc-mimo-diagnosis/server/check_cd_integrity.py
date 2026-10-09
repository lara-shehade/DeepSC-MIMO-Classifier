"""Validate canonical C/D checkpoints and C->C versus C->D first divergence."""

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from channels import MIMOChannel, MIMOConfig
from collect_profiles import checkpoint_model, pad_sequences, strip_padding
from main import resolve_vocab, validate_vocabulary
from utils import PowerNormalize, SNR_to_noise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-data-root", required=True)
    parser.add_argument("--vocab-file", required=True)
    parser.add_argument("--checkpoint-c", required=True)
    parser.add_argument("--checkpoint-d", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA integrity check requested but unavailable")
    device = torch.device(args.device)
    vocab_path = resolve_vocab(args.vocab_file, args.shared_data_root)
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    token_to_idx = validate_vocabulary(vocabulary)
    pad = token_to_idx["<PAD>"]
    c, info_c = checkpoint_model(args.checkpoint_c, len(token_to_idx), device)
    d, info_d = checkpoint_model(args.checkpoint_d, len(token_to_idx), device)
    if info_c["sha256"] == info_d["sha256"]:
        raise AssertionError("C and D checkpoints are byte-identical")
    if info_c["architecture"] != info_d["architecture"]:
        raise AssertionError("C and D architectures differ")

    data_path = Path(args.shared_data_root) / "europarl" / "test_data.pkl"
    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    source = pad_sequences(
        [strip_padding(row, pad) for row in data[:args.batch_size]], pad
    ).to(device)
    source_mask = (source == pad).unsqueeze(-2)
    with torch.no_grad():
        encoded = c.encoder(source, source_mask)
        transmitted = PowerNormalize(c.channel_encoder(encoded))
        left_channel = MIMOChannel(MIMOConfig(fading_scope="batch"), seed=3101)
        right_channel = MIMOChannel(MIMOConfig(fading_scope="batch"), seed=3101)
        left_received, left_trace = left_channel(
            transmitted, SNR_to_noise(10), return_trace=True
        )
        right_received, right_trace = right_channel(
            transmitted, SNR_to_noise(10), return_trace=True
        )
        memory_c = c.channel_decoder(left_received)
        memory_d = d.channel_decoder(right_received)

    paired = {
        "true_h": torch.equal(left_trace.true_h, right_trace.true_h),
        "estimated_h": torch.equal(left_trace.estimated_h, right_trace.estimated_h),
        "noise": torch.equal(left_trace.noise, right_trace.noise),
        "noisy_signal": torch.equal(left_trace.noisy_signal, right_trace.noisy_signal),
        "equalized_input": torch.equal(left_received, right_received),
    }
    if not all(paired.values()):
        raise AssertionError("C->C and C->D diverged before channel decoding: {}".format(paired))
    if torch.equal(memory_c, memory_d):
        raise AssertionError("C and D channel-decoder outputs are unexpectedly identical")
    if not torch.isfinite(memory_c).all() or not torch.isfinite(memory_d).all():
        raise AssertionError("checkpoint outputs contain non-finite values")
    report = {
        "status": "ok",
        "checkpoint_c": info_c,
        "checkpoint_d": info_d,
        "paired_equal_through_R1": paired,
        "first_divergence": "R2_channel_decoder_output",
        "r2_mean_absolute_difference": float((memory_c - memory_d).abs().mean().cpu()),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
