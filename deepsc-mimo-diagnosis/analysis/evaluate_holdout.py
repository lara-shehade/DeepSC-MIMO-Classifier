"""Evaluate already-frozen classical and neural diagnostics on arXiv once."""

import argparse
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.classical_diagnostics import matrix
from analysis.neural_diagnostics import (
    CLASSES, make_model, metrics, predictions, scan_records,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-features", required=True)
    parser.add_argument("--holdout-traces", required=True)
    parser.add_argument("--classical-model", required=True)
    parser.add_argument("--neural-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    with Path(args.holdout_features).open(newline="", encoding="utf-8") as handle:
        feature_rows = list(csv.DictReader(handle))
    if not feature_rows or not all(row["holdout"] in {"1", "True", "true"} for row in feature_rows):
        raise ValueError("holdout feature file must contain arXiv HOLDOUT rows only")
    classical = joblib.load(args.classical_model)
    classical_prediction = classical["pipeline"].predict(
        matrix(feature_rows, classical["columns"])
    )
    classical_ood_recall = float(np.mean(classical_prediction == "ood"))

    frozen = torch.load(args.neural_model, map_location="cpu", weights_only=False)
    records = scan_records(args.holdout_traces, allow_holdout=True)
    model = make_model(frozen["architecture"], frozen["stage"])
    model.load_state_dict(frozen["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    actual, predicted = predictions(model, records, frozen["stage"], 64, device)
    neural_report = metrics(actual, predicted)
    report = {
        "status": "evaluated_after_freeze",
        "holdout": "arxiv",
        "samples": len(records),
        "classical_ood_recall": classical_ood_recall,
        "classical_prediction_counts": {
            label: int(np.sum(classical_prediction == label)) for label in CLASSES
        },
        "neural": neural_report,
        "neural_selection": frozen["selection"],
    }
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
