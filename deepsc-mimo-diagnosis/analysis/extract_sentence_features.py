"""Extract padding-aware sentence-level blind features from schema-v3 artifacts."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


STAGES = {
    "R0": "R0_noisy_received",
    "R1": "R1_equalized_decoder_input",
    "R2": "R2_channel_decoder_output",
    "R3": "R3_free_running_decoder_hidden",
}


def real_matrix(value):
    value = value.detach().cpu()
    if value.is_complex():
        value = torch.view_as_real(value).flatten(-2)
    return value.float().reshape(value.shape[0], -1)


def gini(values):
    values = values.flatten().clamp_min(0)
    if not values.numel() or float(values.sum()) == 0:
        return 0.0
    ordered = values.sort().values
    indices = torch.arange(1, ordered.numel() + 1, dtype=ordered.dtype)
    return float(
        (2 * (indices * ordered).sum() / (ordered.numel() * ordered.sum()))
        - (ordered.numel() + 1) / ordered.numel()
    )


def sequence_features(prefix, value):
    matrix = real_matrix(value)
    vector = matrix.flatten()
    finite = vector[torch.isfinite(vector)]
    if not finite.numel():
        return {}
    quantiles = torch.quantile(finite, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    mean = finite.mean()
    centered = finite - mean
    std = finite.std(unbiased=False)
    result = {
        prefix + "__mean": float(mean),
        prefix + "__std": float(std),
        prefix + "__min": float(finite.min()),
        prefix + "__max": float(finite.max()),
        prefix + "__rms": float(torch.sqrt((finite * finite).mean())),
        prefix + "__energy": float((finite * finite).sum()),
        prefix + "__p10": float(quantiles[0]),
        prefix + "__p25": float(quantiles[1]),
        prefix + "__p50": float(quantiles[2]),
        prefix + "__p75": float(quantiles[3]),
        prefix + "__p90": float(quantiles[4]),
        prefix + "__abs_mean": float(finite.abs().mean()),
        prefix + "__zero_fraction": float((finite == 0).float().mean()),
        prefix + "__gini_energy": gini(finite.square()),
    }
    if float(std) > 0:
        normalized = centered / std
        result[prefix + "__skew"] = float((normalized ** 3).mean())
        result[prefix + "__kurtosis"] = float((normalized ** 4).mean())
    for dimension in range(matrix.size(1)):
        column = matrix[:, dimension]
        result["{}__dim{:03d}_mean".format(prefix, dimension)] = float(column.mean())
        result["{}__dim{:03d}_std".format(prefix, dimension)] = float(column.std(unbiased=False))
        result["{}__dim{:03d}_rms".format(prefix, dimension)] = float(torch.sqrt((column * column).mean()))
    if matrix.size(0) > 1:
        centered_matrix = matrix - matrix.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered_matrix)
        energy = singular.square()
        probability = energy / energy.sum().clamp_min(1e-12)
        nonzero = probability[probability > 0]
        entropy = -(nonzero * nonzero.log()).sum()
        result[prefix + "__effective_rank"] = float(entropy.exp())
        result[prefix + "__spectral_entropy"] = float(entropy)
        result[prefix + "__top1_concentration"] = float(probability[0])
        result[prefix + "__top3_concentration"] = float(probability[:3].sum())
        result[prefix + "__singular_max"] = float(singular[0])
        result[prefix + "__singular_min"] = float(singular[-1])
        result[prefix + "__condition"] = float(singular[0] / singular[-1].clamp_min(1e-12))
        covariance = centered_matrix.mT @ centered_matrix / max(1, matrix.size(0) - 1)
        result[prefix + "__covariance_abs_mean"] = float(covariance.abs().mean())
        if covariance.size(0) > 1:
            off_diagonal = covariance[~torch.eye(covariance.size(0), dtype=torch.bool)]
            result[prefix + "__covariance_offdiag_abs_mean"] = float(off_diagonal.abs().mean())
        row_norms = torch.linalg.vector_norm(matrix, dim=1)
        result[prefix + "__row_norm_mean"] = float(row_norms.mean())
        result[prefix + "__row_norm_std"] = float(row_norms.std(unbiased=False))
        for lag in (1, 2, 5):
            if matrix.size(0) > lag:
                similarity = functional.cosine_similarity(matrix[:-lag], matrix[lag:], dim=1)
                result["{}__lag{}_cosine_mean".format(prefix, lag)] = float(similarity.mean())
                result["{}__lag{}_delta_rms".format(prefix, lag)] = float(
                    torch.sqrt(((matrix[lag:] - matrix[:-lag]) ** 2).mean())
                )
    return result


def r0_mimo_features(value):
    samples = value.detach().cpu().reshape(-1, value.shape[-1])
    covariance = samples.mT @ samples.conj() / max(1, samples.size(0))
    diagonal = covariance.diagonal().real
    denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :]).clamp_min(1e-12)
    correlation = covariance / denominator
    singular = torch.linalg.svdvals(samples)
    return {
        "R0__rx0_power": float(diagonal[0]),
        "R0__rx1_power": float(diagonal[1]),
        "R0__antenna_corr_abs": float(correlation[0, 1].abs()),
        "R0__antenna_corr_real": float(correlation[0, 1].real),
        "R0__antenna_corr_imag": float(correlation[0, 1].imag),
        "R0__stream_singular_ratio": float(singular[0] / singular[-1].clamp_min(1e-12)),
    }


def artifact_rows(path):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    source_lengths = artifact["source_lengths"]
    generated_lengths = artifact["R3_generated_lengths"]
    rows = []
    for index, sample_id in enumerate(artifact["sample_ids"].tolist()):
        source_length = int(source_lengths[index])
        generated_length = int(generated_lengths[index])
        row = {
            "artifact": str(path),
            "sample_id": int(sample_id),
            "profile": artifact["profile"],
            "label": artifact["diagnosis_label"],
            "holdout": int(bool(artifact["holdout"])),
            "domain": artifact["domain"],
            "severity_name": artifact.get("severity_name") or "",
            "severity": "" if artifact.get("severity") is None else artifact["severity"],
            "snr_db": artifact["snr_db"],
            "physical_seed": artifact["physical_seed"],
            "batch_index": artifact["batch_index"],
            "source_length": source_length,
            "generated_length": generated_length,
        }
        lengths = {"R0": source_length * 4, "R1": source_length, "R2": source_length, "R3": generated_length}
        stage_rms = {}
        for stage, key in STAGES.items():
            tensor = artifact[key][index, :lengths[stage]]
            row.update(sequence_features(stage, tensor))
            real = real_matrix(tensor)
            stage_rms[stage] = float(torch.sqrt(real.square().mean()))
            if stage == "R0":
                row.update(r0_mimo_features(tensor))
        for left, right in (("R0", "R1"), ("R1", "R2"), ("R2", "R3")):
            row["T__{}_to_{}__rms_ratio".format(left, right)] = (
                stage_rms[right] / max(stage_rms[left], 1e-12)
            )
            row["T__{}_to_{}__log_rms_ratio".format(left, right)] = math.log(
                max(stage_rms[right], 1e-12) / max(stage_rms[left], 1e-12)
            )
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-holdout", action="store_true")
    args = parser.parse_args(argv)
    paths = sorted(Path(args.trace_root).rglob("artifacts/*.pt"))
    if not paths:
        raise FileNotFoundError("no schema-v3 artifacts found under {}".format(args.trace_root))
    rows = []
    for path in paths:
        loaded = artifact_rows(path)
        if not args.include_holdout:
            loaded = [row for row in loaded if not row["holdout"]]
        rows.extend(loaded)
    if not rows:
        raise RuntimeError("no eligible rows after holdout filtering")
    fields = sorted({key for row in rows for key in row})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} sentence rows to {}".format(len(rows), destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
