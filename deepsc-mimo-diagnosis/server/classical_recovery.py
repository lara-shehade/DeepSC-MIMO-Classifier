"""Resume classical diagnosis as independent stage/physical-seed fold jobs.

The estimators, balancing, scoring, model selection, and final fit are imported
from the original analysis. This changes execution granularity, not the study.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np

from analysis.classical_diagnostics import (
    CLASSES,
    balanced_indices,
    estimators,
    feature_columns,
    load_csv,
    matrix,
    score,
)


STAGES = ("R0", "R1", "R2", "R3", "fusion")


def write_json_exclusive(path, value):
    """Publish a complete JSON file; never overwrite another job's result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_marker_exclusive(path):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        temporary.write_text(
            "Model selection completed without arXiv holdout data.\n", encoding="utf-8"
        )
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fold(features, output_root, fold_index, seed):
    rows = load_csv(features)
    groups = sorted({int(row["physical_seed"]) for row in rows})
    if len(groups) < 2:
        raise ValueError("grouped evaluation requires at least two physical seeds")
    tasks = [(stage, held_out) for stage in STAGES for held_out in groups]
    if not 0 <= fold_index < len(tasks):
        raise ValueError("fold index {} outside 0..{}".format(fold_index, len(tasks) - 1))
    stage, held_out = tasks[fold_index]
    destination = output_root / "folds" / "{}-{}.json".format(stage, held_out)
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if (previous.get("stage"), previous.get("held_out_physical_seed")) != (stage, held_out):
            raise ValueError("existing fold result does not match task: {}".format(destination))
        if {item["model"] for item in previous["results"]} != set(estimators(1)):
            raise ValueError("existing fold result is incomplete: {}".format(destination))
        print("Already complete: {}".format(destination), flush=True)
        return
    columns = feature_columns(rows, stage)
    if not columns:
        raise ValueError("no features for stage {}".format(stage))
    x = matrix(rows, columns)
    y = np.asarray([row["label"] for row in rows])
    group_array = np.asarray([int(row["physical_seed"]) for row in rows])
    train_rows = [row for row, group in zip(rows, group_array) if group != held_out]
    train_global = np.flatnonzero(group_array != held_out)
    train_indices = train_global[balanced_indices(train_rows, seed + held_out)]
    test_rows = [row for row, group in zip(rows, group_array) if group == held_out]
    test_global = np.flatnonzero(group_array == held_out)
    test_indices = test_global[balanced_indices(test_rows, seed + 100_000 + held_out)]
    results = []
    for name, estimator in estimators(len(columns)).items():
        print("Fitting {} {} seed {}".format(stage, name, held_out), flush=True)
        estimator.fit(x[train_indices], y[train_indices])
        prediction = estimator.predict(x[test_indices])
        results.append({
            "stage": stage,
            "model": name,
            "held_out_physical_seed": held_out,
            "train_samples": int(len(train_indices)),
            "test_samples": int(len(test_indices)),
            **score(y[test_indices], prediction),
        })
    write_json_exclusive(destination, {
        "stage": stage,
        "held_out_physical_seed": held_out,
        "results": results,
    })
    print("Completed {}".format(destination), flush=True)


def finalize(features, output_root, seed):
    rows = load_csv(features)
    groups = sorted({int(row["physical_seed"]) for row in rows})
    if len(groups) < 2:
        raise ValueError("grouped evaluation requires at least two physical seeds")
    model_path = output_root / "frozen_classical.joblib"
    result_path = output_root / "grouped_results.json"
    marker_path = output_root / "FROZEN"
    if marker_path.exists():
        raise FileExistsError("classical model already frozen")
    if model_path.exists() or result_path.exists():
        raise FileExistsError("partial classical finalization exists; inspect before retry")
    results = []
    expected_models = tuple(estimators(1))
    for stage in STAGES:
        for held_out in groups:
            path = output_root / "folds" / "{}-{}.json".format(stage, held_out)
            report = json.loads(path.read_text(encoding="utf-8"))
            if (report.get("stage"), report.get("held_out_physical_seed")) != (stage, held_out):
                raise ValueError("incorrect fold result: {}".format(path))
            fold_results = report["results"]
            if tuple(item["model"] for item in fold_results) != expected_models:
                raise ValueError("missing or reordered models in {}".format(path))
            if any((item["stage"], item["held_out_physical_seed"]) != (stage, held_out)
                   for item in fold_results):
                raise ValueError("mismatched result in {}".format(path))
            results.extend(fold_results)
    aggregate = []
    for stage in STAGES:
        for name in expected_models:
            subset = [row for row in results if row["stage"] == stage and row["model"] == name]
            aggregate.append({
                "stage": stage,
                "model": name,
                "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in subset])),
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in subset])),
            })
    best = max(aggregate, key=lambda row: row["macro_f1_mean"])
    columns = feature_columns(rows, best["stage"])
    x = matrix(rows, columns)
    all_balanced = balanced_indices(rows, seed)
    frozen = estimators(len(columns))[best["model"]]
    print("Fitting frozen model {} {}".format(best["stage"], best["model"]), flush=True)
    frozen.fit(x[all_balanced], np.asarray([row["label"] for row in rows])[all_balanced])
    temporary_model = model_path.with_name(model_path.name + ".tmp.{}".format(os.getpid()))
    try:
        joblib.dump(
            {"pipeline": frozen, "columns": columns, "classes": CLASSES, "selection": best},
            temporary_model,
        )
        temporary_model.replace(model_path)
    finally:
        temporary_model.unlink(missing_ok=True)
    write_json_exclusive(result_path, {"folds": results, "aggregate": aggregate, "best": best})
    write_marker_exclusive(marker_path)
    print(json.dumps(best, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("fold", "finalize"))
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--seed", default=4401, type=int)
    args = parser.parse_args(argv)
    if args.mode == "fold":
        if args.fold_index is None:
            parser.error("fold mode requires --fold-index")
        fold(args.features, args.output_root, args.fold_index, args.seed)
    else:
        if args.fold_index is not None:
            parser.error("finalize mode does not accept --fold-index")
        finalize(args.features, args.output_root, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
