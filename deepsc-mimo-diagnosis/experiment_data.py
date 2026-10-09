"""Versioned, MIMO-aware experiment trace writer."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict

import torch

from utils import PipelineTrace


SCHEMA_VERSION = 2


def _cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def _power(value):
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return None
    if value.is_complex():
        return float(value.detach().abs().square().mean().cpu())
    return float(value.detach().float().square().mean().cpu())


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


class ExperimentDataWriter:
    CSV_FIELDS = (
        "schema_version", "split", "epoch", "batch_index", "global_step",
        "batch_size", "token_count", "batch_loss", "token_accuracy",
        "snr_db", "noise_std", "tx_normalized_power", "tx_scaled_power",
        "faded_power", "noise_power", "noisy_power", "equalized_power",
        "channel_decoder_power", "semantic_decoder_power", "csi_error_l2",
        "channel_condition_mean", "assumed_stream_power_mean",
        "realized_stream_power_mean", "realized_stream_correlation_mean",
        "artifact",
    )

    def __init__(
        self,
        output_root,
        profile,
        run_id=None,
        metadata=None,
        logits_mode="topk",
        logits_top_k=5,
    ):
        if logits_mode not in {"full", "topk", "none"}:
            raise ValueError("logits_mode must be full, topk, or none")
        if logits_top_k <= 0:
            raise ValueError("logits_top_k must be positive")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or timestamp
        self.run_dir = Path(output_root).expanduser().resolve() / profile / self.run_id
        self.artifact_dir = self.run_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.run_dir / "batches.csv"
        self.manifest_path = self.run_dir / "manifest.jsonl"
        self.logits_mode = logits_mode
        self.logits_top_k = int(logits_top_k)

        payload = dict(metadata or {})
        payload.update({
            "schema_version": SCHEMA_VERSION,
            "created_utc": timestamp,
            "profile": profile,
            "run_id": self.run_id,
            "logits_mode": logits_mode,
            "logits_top_k": self.logits_top_k,
            "torch_version": torch.__version__,
        })
        (self.run_dir / "metadata.json").write_text(
            json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.CSV_FIELDS).writeheader()
        self.manifest_path.touch()

    def _logits_payload(self, logits):
        if self.logits_mode == "none":
            return {}
        if self.logits_mode == "full":
            return {"logits": _cpu(logits)}
        k = min(self.logits_top_k, logits.shape[-1])
        values, indices = torch.topk(logits, k=k, dim=-1)
        return {
            "logits_topk_values": _cpu(values),
            "logits_topk_indices": _cpu(indices),
        }

    @staticmethod
    def _token_accuracy(trace):
        valid = trace.target_tokens != trace.padding_idx
        if not bool(valid.any()):
            return None
        correct = (trace.predicted_tokens == trace.target_tokens) & valid
        return float(correct.sum().float().div(valid.sum()).detach().cpu())

    @staticmethod
    def _channel_metrics(channel):
        true_h = channel.true_h.detach()
        estimated_h = channel.estimated_h.detach()
        csi_error = float((true_h - estimated_h).abs().square().sum((-2, -1)).sqrt().mean().cpu())
        singular = torch.linalg.svdvals(true_h)
        condition = singular[..., 0] / singular[..., -1].clamp_min(1e-12)
        assumed = channel.assumed_stream_covariance.detach()
        realized = channel.realized_stream_covariance.detach()
        diagonal = realized.diagonal(dim1=-2, dim2=-1).real
        off_diagonal = realized - torch.diag_embed(realized.diagonal(dim1=-2, dim2=-1))
        return {
            "csi_error_l2": csi_error,
            "channel_condition_mean": float(condition.mean().cpu()),
            "assumed_stream_power_mean": float(
                assumed.diagonal(dim1=-2, dim2=-1).real.mean().cpu()
            ),
            "realized_stream_power_mean": float(diagonal.mean().cpu()),
            "realized_stream_correlation_mean": float(off_diagonal.abs().mean().cpu()),
        }

    @staticmethod
    def _atomic_torch_save(payload, destination):
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

    def save_batch(
        self,
        trace: PipelineTrace,
        split,
        epoch,
        batch_index,
        global_step,
        sample_ids=None,
        extra_metrics=None,
    ):
        if not isinstance(trace, PipelineTrace):
            raise TypeError("trace must be a PipelineTrace")
        channel = trace.channel
        filename = "batch_{:06d}.pt".format(int(global_step))
        destination = self.artifact_dir / filename

        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "global_step": int(global_step),
            "sample_ids": _cpu(sample_ids),
            "source_tokens": _cpu(trace.source_tokens),
            "source_mask": _cpu(trace.source_mask),
            "sequence_lengths": _cpu(trace.sequence_lengths),
            "decoder_input_tokens": _cpu(trace.decoder_input_tokens),
            "target_tokens": _cpu(trace.target_tokens),
            "target_mask": _cpu(trace.target_mask),
            "padding_idx": trace.padding_idx,
            "semantic_encoder_output": _cpu(trace.semantic_encoder_output),
            "channel_encoder_raw": _cpu(trace.channel_encoder_raw),
            "tx_normalized": _cpu(trace.tx_normalized),
            "channel": {
                "underlying_iid_h": _cpu(channel.underlying_iid_h),
                "true_h": _cpu(channel.true_h),
                "estimated_h": _cpu(channel.estimated_h),
                "tx_grouped": _cpu(channel.tx_grouped),
                "tx_scaled": _cpu(channel.tx_scaled),
                "faded_signal": _cpu(channel.faded_signal),
                "noise": _cpu(channel.noise),
                "noisy_signal": _cpu(channel.noisy_signal),
                "equalized_signal": _cpu(channel.equalized_signal),
                "assumed_stream_covariance": _cpu(channel.assumed_stream_covariance),
                "realized_stream_covariance": _cpu(channel.realized_stream_covariance),
                "noise_covariance": _cpu(channel.noise_covariance),
                "equalized_real": _cpu(channel.equalized_real),
                "noise_std": channel.noise_std,
                "power_scale": channel.power_scale,
                "fading_scope": channel.fading_scope,
                "profile": channel.profile,
                "equalizer": channel.equalizer,
            },
            "channel_decoder_input": _cpu(trace.channel_decoder_input),
            "channel_decoder_output": _cpu(trace.channel_decoder_output),
            "semantic_decoder_output": _cpu(trace.semantic_decoder_output),
            "predicted_tokens": _cpu(trace.predicted_tokens),
            "per_token_loss": _cpu(trace.per_token_loss),
            "batch_loss": _cpu(trace.batch_loss),
        }
        payload.update(self._logits_payload(trace.logits))
        self._atomic_torch_save(payload, destination)

        valid_tokens = int((trace.target_tokens != trace.padding_idx).sum().detach().cpu())
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "global_step": int(global_step),
            "batch_size": int(trace.source_tokens.shape[0]),
            "token_count": valid_tokens,
            "batch_loss": float(trace.batch_loss.detach().cpu()),
            "token_accuracy": self._token_accuracy(trace),
            "snr_db": None,
            "noise_std": channel.noise_std,
            "tx_normalized_power": _power(trace.tx_normalized),
            "tx_scaled_power": _power(channel.tx_scaled),
            "faded_power": _power(channel.faded_signal),
            "noise_power": _power(channel.noise),
            "noisy_power": _power(channel.noisy_signal),
            "equalized_power": _power(channel.equalized_signal),
            "channel_decoder_power": _power(trace.channel_decoder_output),
            "semantic_decoder_power": _power(trace.semantic_decoder_output),
            "artifact": str(destination.relative_to(self.run_dir)),
        }
        metrics.update(self._channel_metrics(channel))
        for key, value in (extra_metrics or {}).items():
            if key in metrics:
                metrics[key] = value
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.CSV_FIELDS).writerow(metrics)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "global_step": int(global_step),
            "batch_index": int(batch_index),
            "artifact": str(destination.relative_to(self.run_dir)),
            "sample_count": int(trace.source_tokens.shape[0]),
            "tensor_shapes": {
                "tx_normalized": list(trace.tx_normalized.shape),
                "true_h": list(channel.true_h.shape),
                "tx_grouped": list(channel.tx_grouped.shape),
                "logits": list(trace.logits.shape),
            },
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, sort_keys=True) + "\n")
        return destination


__all__ = ["ExperimentDataWriter", "SCHEMA_VERSION"]
