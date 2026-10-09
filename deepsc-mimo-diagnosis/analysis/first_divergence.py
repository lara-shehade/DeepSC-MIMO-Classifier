"""Audit paired causal first divergence for the approved profiles."""

import argparse
import json
from pathlib import Path

import torch


STAGES = [
    ("source", "source_tokens"),
    ("R0", "R0_noisy_received"),
    ("R1", "R1_equalized_decoder_input"),
    ("R2", "R2_channel_decoder_output"),
    ("R3", "R3_free_running_decoder_hidden"),
]
EXPECTED = {
    "low_snr": "R0",
    "imperfect_csi": "R1",
    "spatial_correlation": "R0",
    "mismatch": "R2",
}


def same(left, right):
    if tuple(left.shape) != tuple(right.shape):
        return False
    if left.is_floating_point() or left.is_complex():
        return bool(torch.allclose(left, right, rtol=0, atol=0))
    return bool(torch.equal(left, right))


def key(artifact):
    return (
        int(artifact["physical_seed"]), int(artifact["batch_index"]),
        tuple(artifact["sample_ids"].tolist()),
    )


def load_profile(root, profile):
    result = {}
    for path in sorted((Path(root) / profile).rglob("artifacts/*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        result.setdefault(key(artifact), []).append((path, artifact))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    baseline = load_profile(args.trace_root, "normal")
    if not baseline:
        raise FileNotFoundError("normal paired artifacts are missing")
    reports = []
    for profile, expected in EXPECTED.items():
        candidates = load_profile(args.trace_root, profile)
        if not candidates:
            raise FileNotFoundError("{} paired artifacts are missing".format(profile))
        for pair_key, variants in candidates.items():
            if pair_key not in baseline:
                raise AssertionError("{} has no matching normal artifact".format(pair_key))
            normal_path, normal = baseline[pair_key][0]
            for impaired_path, impaired in variants:
                equality = {name: same(normal[field], impaired[field]) for name, field in STAGES}
                first = next((name for name, _ in STAGES if not equality[name]), "none")
                if first != expected:
                    raise AssertionError(
                        "{} expected first divergence {}, observed {}: {}".format(
                            profile, expected, first, equality
                        )
                    )
                if profile == "low_snr":
                    if not same(normal["true_h"], impaired["true_h"]):
                        raise AssertionError("low-SNR H is not paired")
                    if not torch.allclose(
                        normal["standardized_noise"], impaired["standardized_noise"],
                        rtol=2e-6, atol=2e-6,
                    ):
                        raise AssertionError("low-SNR standardized noise is not paired")
                if profile == "imperfect_csi":
                    for field in ("true_h", "noise", "R0_noisy_received"):
                        if not same(normal[field], impaired[field]):
                            raise AssertionError("CSI profile changed {}".format(field))
                if profile == "spatial_correlation":
                    if not same(normal["underlying_iid_h"], impaired["underlying_iid_h"]):
                        raise AssertionError("spatial profile changed underlying iid Hw")
                reports.append({
                    "profile": profile,
                    "severity": impaired.get("severity"),
                    "physical_seed": pair_key[0],
                    "batch_index": pair_key[1],
                    "normal_artifact": str(normal_path),
                    "impaired_artifact": str(impaired_path),
                    "stage_equality": equality,
                    "first_divergence": first,
                })
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print("validated {} paired artifacts".format(len(reports)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
