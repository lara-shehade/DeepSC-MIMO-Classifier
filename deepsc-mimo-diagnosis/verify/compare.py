"""Compare re-run outputs with the stored results in results/.

Usage (from the repository root):
  python verify/compare.py classical-quick FEATURES.csv [--stage fusion] [--model elasticnet_logistic] [--held-out 3101]
  python verify/compare.py classical NEW_grouped_results.json
  python verify/compare.py features NEW.csv REFERENCE.csv
  python verify/compare.py localization NEW_first_divergence.json
  python verify/compare.py neural NEW_fold.json

Nothing here trains DeepSC. Small differences in the classifiers (< 0.01 macro-F1)
can come from solver randomness (saga, liblinear) or CPU vs GPU arithmetic.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
TOL = 0.01


def verdict(diff, tol):
    return "PASS" if diff <= tol else "CHECK"


def stored_classical():
    data = json.loads((RESULTS / "classical" / "grouped_results.json").read_text())
    return {(f["stage"], f["model"], f["held_out_physical_seed"]): f for f in data["folds"]}


def classical_quick(args):
    from analysis.classical_diagnostics import (
        balanced_indices, estimators, feature_columns, load_csv, matrix,
    )
    rows = load_csv(args.features)
    columns = feature_columns(rows, args.stage)
    x = matrix(rows, columns)
    y = np.asarray([row["label"] for row in rows])
    groups = np.asarray([int(row["physical_seed"]) for row in rows])
    # identical fold construction to analysis/classical_diagnostics.py
    train_rows = [r for r, g in zip(rows, groups) if g != args.held_out]
    train_idx = np.flatnonzero(groups != args.held_out)[balanced_indices(train_rows, args.seed + args.held_out)]
    test_rows = [r for r, g in zip(rows, groups) if g == args.held_out]
    test_idx = np.flatnonzero(groups == args.held_out)[
        balanced_indices(test_rows, args.seed + 100_000 + args.held_out)]
    model = estimators(len(columns))[args.model]
    model.fit(x[train_idx], y[train_idx])
    from sklearn.metrics import balanced_accuracy_score, f1_score
    pred = model.predict(x[test_idx])
    new_f1 = f1_score(y[test_idx], pred, average="macro")
    new_ba = balanced_accuracy_score(y[test_idx], pred)
    ref = stored_classical()[(args.stage, args.model, args.held_out)]
    print("{} / {} / held-out {}: {} features, {} train, {} test".format(
        args.stage, args.model, args.held_out, len(columns), len(train_idx), len(test_idx)))
    print("  macro-F1          new {:.4f}  stored {:.4f}".format(new_f1, ref["macro_f1"]))
    print("  balanced accuracy new {:.4f}  stored {:.4f}".format(new_ba, ref["balanced_accuracy"]))
    diff = max(abs(new_f1 - ref["macro_f1"]), abs(new_ba - ref["balanced_accuracy"]))
    print("  max difference {:.4f} -> {}".format(diff, verdict(diff, TOL)))


def classical(args):
    new = json.loads(Path(args.new).read_text())["folds"]
    ref = stored_classical()
    worst = 0.0
    print("{:<7}{:<22}{:>6}{:>9}{:>9}".format("stage", "model", "seed", "new F1", "stored"))
    for f in new:
        r = ref[(f["stage"], f["model"], f["held_out_physical_seed"])]
        worst = max(worst, abs(f["macro_f1"] - r["macro_f1"]))
        print("{:<7}{:<22}{:>6}{:>9.4f}{:>9.4f}".format(
            f["stage"], f["model"], f["held_out_physical_seed"], f["macro_f1"], r["macro_f1"]))
    print("{} folds compared, max macro-F1 difference {:.4f} -> {}".format(len(new), worst, verdict(worst, TOL)))


def read_features(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = sorted(c for c in reader.fieldnames if "__" in c)
        out = {}
        for row in reader:
            key = (row["profile"], row["severity"], row["physical_seed"], row["batch_index"], row["sample_id"])
            vals = []
            for c in cols:
                try:
                    vals.append(float(row[c]))
                except (TypeError, ValueError):
                    vals.append(float("nan"))
            out[key] = np.asarray(vals)
    return cols, out


def features(args):
    cols_new, new = read_features(args.new)
    cols_ref, ref = read_features(args.reference)
    if cols_new != cols_ref:
        print("feature columns differ: {} new vs {} reference".format(len(cols_new), len(cols_ref)))
        return
    missing = set(ref) ^ set(new)
    bad, worst = 0, 0.0
    for key in set(ref) & set(new):
        a, b = new[key], ref[key]
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-5, atol=1e-6) | both_nan
        if not close.all():
            bad += 1
            d = np.abs(a - b)[~close]
            worst = max(worst, float(np.nanmax(d)) if np.isfinite(d).any() else math.inf)
    print("{} feature columns, {} sentences in both, {} only in one file".format(
        len(cols_ref), len(set(ref) & set(new)), len(missing)))
    print("sentences with any differing feature: {} (max abs difference {:.3g}) -> {}".format(
        bad, worst, "PASS" if bad == 0 and not missing else "CHECK"))


def localization(args):
    def keyed(path):
        return {(r["profile"], str(r["severity"]), r["physical_seed"], r["batch_index"]): r
                for r in json.loads(Path(path).read_text())}
    new, ref = keyed(args.new), keyed(RESULTS / "localization" / "first_divergence.json")
    same = sum(1 for k in ref if k in new and new[k]["first_divergence"] == ref[k]["first_divergence"]
               and new[k]["stage_equality"] == ref[k]["stage_equality"])
    print("{} of {} paired batches give the same entry stage and stage equalities -> {}".format(
        same, len(ref), "PASS" if same == len(ref) == len(new) else "CHECK"))


def neural(args):
    new = json.loads(Path(args.new).read_text())
    name = "{}_{}_seed{}.json".format(new["architecture"], new["stage"], new["held_out_physical_seed"])
    ref = json.loads((RESULTS / "neural" / "folds" / name).read_text())
    print("{}: macro-F1 new {:.4f} stored {:.4f}; balanced accuracy new {:.4f} stored {:.4f}".format(
        name, new["macro_f1"], ref["macro_f1"], new["balanced_accuracy"], ref["balanced_accuracy"]))
    diff = max(abs(new["macro_f1"] - ref["macro_f1"]), abs(new["balanced_accuracy"] - ref["balanced_accuracy"]))
    print("max difference {:.4f} -> {} (GPU vs CPU training is not bit-identical; tolerance 0.02)".format(
        diff, verdict(diff, 0.02)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("classical-quick")
    q.add_argument("features")
    q.add_argument("--stage", default="fusion", choices=("R0", "R1", "R2", "R3", "fusion"))
    q.add_argument("--model", default="elasticnet_logistic",
                   choices=("anova_logistic", "mrmr_linear_svm", "elasticnet_logistic"))
    q.add_argument("--held-out", default=3101, type=int)
    q.add_argument("--seed", default=4401, type=int)
    sub.add_parser("classical").add_argument("new")
    f = sub.add_parser("features")
    f.add_argument("new")
    f.add_argument("reference")
    sub.add_parser("localization").add_argument("new")
    sub.add_parser("neural").add_argument("new")
    args = p.parse_args()
    {"classical-quick": classical_quick, "classical": classical, "features": features,
     "localization": localization, "neural": neural}[args.cmd](args)


if __name__ == "__main__":
    main()
