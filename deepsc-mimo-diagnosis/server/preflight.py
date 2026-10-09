"""Fail-fast deployment check for an unpacked DeepSC-MIMO directory."""

import argparse
import json
from pathlib import Path
import platform
import pickle
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_TEST_CAPACITY = {
    "europarl": 5 * 10 * 128,
    "ukhansard": 5 * 10 * 128,
    "arxiv": 5 * 6 * 128,
}

from channels import MIMOChannel, MIMOConfig
from dataset import resolve_data_file
from main import resolve_vocab, validate_vocabulary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--shared-data-root", default=None)
    parser.add_argument("--vocab-file", default=None)
    parser.add_argument("--allow-nonshared-vocab", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report-file", default=None)
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
    train_path = resolve_data_file("train", args.data_root)
    test_path = resolve_data_file("test", args.data_root)
    vocab_path = resolve_vocab(args.vocab_file, args.data_root)
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    token_to_idx = validate_vocabulary(
        vocabulary, allow_nonshared=args.allow_nonshared_vocab
    )
    shared_root = Path(args.shared_data_root).resolve() if args.shared_data_root else None
    domain_files = {}
    if shared_root:
        for domain in ("europarl", "ukhansard", "arxiv"):
            for split in ("train", "test"):
                path = shared_root / domain / "{}_data.pkl".format(split)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with path.open("rb") as handle:
                    count = len(pickle.load(handle))
                if count <= 0:
                    raise ValueError("dataset is empty: {}".format(path))
                if split == "test" and count < REQUIRED_TEST_CAPACITY[domain]:
                    raise ValueError(
                        "{} test set has {} samples, but collection requires at least {}"
                        .format(domain, count, REQUIRED_TEST_CAPACITY[domain])
                    )
                domain_files["{}_{}".format(domain, split)] = {
                    "path": str(path), "samples": count, "bytes": path.stat().st_size,
                }

    device = torch.device("cuda" if args.require_cuda else "cpu")
    probe = torch.randn(2, 3, 16, device=device, requires_grad=True)
    output, trace = MIMOChannel(MIMOConfig(), seed=123)(probe, 0.1, return_trace=True)
    if output.shape != probe.shape or not torch.isfinite(output).all():
        raise RuntimeError("MIMO smoke test failed")
    output.square().mean().backward()
    if probe.grad is None or not torch.isfinite(probe.grad).all():
        raise RuntimeError("MIMO backward smoke test failed")
    if args.require_cuda and (output.device.type != "cuda" or trace.true_h.device.type != "cuda"):
        raise RuntimeError("CUDA preflight unexpectedly executed on CPU")
    for profile, kwargs in (
        ("imperfect_csi", {"csi_rho": 0.90}),
        ("spatial_correlation", {"spatial_rho": 0.90}),
    ):
        approved = MIMOChannel(MIMOConfig(profile=profile, **kwargs), seed=123)
        candidate = approved(probe.detach(), 0.1)
        if candidate.shape != probe.shape or not torch.isfinite(candidate).all():
            raise RuntimeError("approved {} smoke test failed".format(profile))

    report = {
        "status": "ok",
        "project_root": str(ROOT),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "probe_device": str(output.device),
        "train_data": str(train_path),
        "test_data": str(test_path),
        "vocab": str(vocab_path),
        "vocabulary_size": len(token_to_idx),
        "probe_h_shape": list(trace.true_h.shape),
        "shared_domain_files": domain_files,
        "required_test_capacity": REQUIRED_TEST_CAPACITY,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report_file:
        destination = Path(args.report_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
