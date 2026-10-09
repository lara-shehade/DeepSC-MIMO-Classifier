"""Small executable smoke test for grouped classical diagnostics."""

import csv
import json
from pathlib import Path
import random
import tempfile

import numpy as np

from analysis.classical_diagnostics import CLASSES, main
from server.classical_recovery import fold, finalize


def smoke():
    rng = random.Random(17)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        feature_file = root / "features.csv"
        fields = [
            "artifact", "sample_id", "profile", "label", "holdout", "domain",
            "severity_name", "severity", "snr_db", "physical_seed", "batch_index",
            "source_length", "generated_length",
        ] + ["{}__feature{}".format(stage, index) for stage in ("R0", "R1", "R2", "R3") for index in range(4)]
        with feature_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            sample_id = 0
            for seed in (3101, 3102, 3103, 3104, 3105):
                for label_index, label in enumerate(CLASSES):
                    severities = (0.9, 0.6, 0.3) if label == "spatial_correlation" else (
                        (0.99, 0.95, 0.90) if label == "imperfect_csi" else ("", "", "")
                    )
                    for severity in severities:
                        row = {
                            "artifact": "synthetic", "sample_id": sample_id,
                            "profile": label, "label": label, "holdout": 0,
                            "domain": "synthetic", "severity_name": "",
                            "severity": severity, "snr_db": 10,
                            "physical_seed": seed, "batch_index": 0,
                            "source_length": 8, "generated_length": 8,
                        }
                        for stage in ("R0", "R1", "R2", "R3"):
                            for index in range(4):
                                row["{}__feature{}".format(stage, index)] = (
                                    label_index + index * 0.1 + rng.random() * 0.01
                                )
                        writer.writerow(row)
                        sample_id += 1
        output = root / "classical"
        np.random.seed(12345)
        assert main([str(feature_file), "--output-root", str(output)]) == 0
        assert (output / "grouped_results.json").is_file()
        assert (output / "frozen_classical.joblib").is_file()
        assert (output / "FROZEN").is_file()
        split_output = root / "classical_split"
        np.random.seed(12345)
        for fold_index in range(25):
            fold(feature_file, split_output, fold_index, 4401)
        finalize(feature_file, split_output, 4401)
        original = json.loads((output / "grouped_results.json").read_text(encoding="utf-8"))
        recovered = json.loads((split_output / "grouped_results.json").read_text(encoding="utf-8"))
        assert original == recovered, "split execution changed classical results"
        assert (split_output / "frozen_classical.joblib").is_file()
        assert (split_output / "FROZEN").is_file()
    print("Grouped classical diagnostic smoke test passed")


if __name__ == "__main__":
    smoke()
