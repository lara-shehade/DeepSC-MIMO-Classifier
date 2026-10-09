"""Extract compact per-batch MIMO diagnostic features from schema-v2 traces."""

import argparse
import csv
from pathlib import Path

import torch


STAGES = (
    "semantic_encoder_output",
    "channel_encoder_raw",
    "tx_normalized",
    "channel_decoder_input",
    "channel_decoder_output",
    "semantic_decoder_output",
)
CHANNEL_STAGES = (
    "tx_scaled", "faded_signal", "noise", "noisy_signal", "equalized_signal"
)


def tensor_stats(prefix, tensor):
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return {}
    value = tensor.detach()
    magnitude = value.abs().float() if value.is_complex() else value.float().abs()
    signed = value.real.float() if value.is_complex() else value.float()
    return {
        prefix + "_mean": float(signed.mean()),
        prefix + "_std": float(signed.std(unbiased=False)),
        prefix + "_abs_mean": float(magnitude.mean()),
        prefix + "_power": float(magnitude.square().mean()),
        prefix + "_max_abs": float(magnitude.max()),
    }


def artifact_row(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 2:
        raise ValueError("{} is not a schema-v2 MIMO trace".format(path))
    channel = payload["channel"]
    row = {
        "artifact": path.name,
        "global_step": payload["global_step"],
        "batch_index": payload["batch_index"],
        "batch_loss": float(payload["batch_loss"]),
        "noise_std": channel["noise_std"],
        "power_scale": channel["power_scale"],
        "fading_scope": channel["fading_scope"],
        "profile": channel["profile"],
        "equalizer": channel["equalizer"],
    }
    target = payload["target_tokens"]
    predicted = payload["predicted_tokens"]
    valid = target != payload["padding_idx"]
    row["token_accuracy"] = float(((predicted == target) & valid).sum().float() / valid.sum().clamp_min(1))
    for stage in STAGES:
        row.update(tensor_stats(stage, payload.get(stage)))
    for stage in CHANNEL_STAGES:
        row.update(tensor_stats(stage, channel.get(stage)))
    h = channel["true_h"]
    singular = torch.linalg.svdvals(h)
    row["channel_condition_mean"] = float((singular[..., 0] / singular[..., -1].clamp_min(1e-12)).mean())
    row["csi_error_l2"] = float((h - channel["estimated_h"]).abs().square().sum((-2, -1)).sqrt().mean())
    row.update(tensor_stats(
        "assumed_stream_covariance", channel["assumed_stream_covariance"]
    ))
    row.update(tensor_stats(
        "realized_stream_covariance", channel["realized_stream_covariance"]
    ))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    artifacts = sorted((run_dir / "artifacts").glob("*.pt"))
    if not artifacts:
        raise FileNotFoundError("no .pt artifacts found under {}".format(run_dir / "artifacts"))
    rows = [artifact_row(path) for path in artifacts]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote {} rows to {}".format(len(rows), output.resolve()))


if __name__ == "__main__":
    main()
