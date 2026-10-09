"""Recover only fusion folds 3103--3105 as estimator-sized tasks."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analysis.classical_diagnostics import (
    balanced_indices, estimators, feature_columns, load_csv, matrix, score,
)
from server.classical_recovery import finalize


MISSING_SEEDS = (3103, 3104, 3105)
MODEL_NAMES = tuple(estimators(1))
TASKS = tuple((seed, model) for seed in MISSING_SEEDS for model in MODEL_NAMES)
ALL_SEEDS = (3101, 3102, 3103, 3104, 3105)


def write_json_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_part(path, seed, model):
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("stage"), value.get("model"),
            value.get("held_out_physical_seed")) != ("fusion", model, seed):
        raise ValueError("incorrect recovery part: {}".format(path))
    for key in ("balanced_accuracy", "macro_f1", "confusion_matrix"):
        if key not in value:
            raise ValueError("incomplete recovery part: {}".format(path))
    return value


def run_part(features, output_root, task_index, seed=4401):
    if not 0 <= task_index < len(TASKS):
        raise ValueError("task index must be 0..{}".format(len(TASKS) - 1))
    held_out, model_name = TASKS[task_index]
    destination = output_root / "parts" / "fusion-{}-{}.json".format(
        held_out, model_name
    )
    if destination.exists():
        validate_part(destination, held_out, model_name)
        print("Already complete: {}".format(destination), flush=True)
        return
    rows = load_csv(features)
    groups = tuple(sorted({int(row["physical_seed"]) for row in rows}))
    if groups != ALL_SEEDS:
        raise ValueError("expected physical seeds {}, got {}".format(ALL_SEEDS, groups))
    columns = feature_columns(rows, "fusion")
    if not columns:
        raise ValueError("fusion has no features")
    x = matrix(rows, columns)
    y = np.asarray([row["label"] for row in rows])
    group_array = np.asarray([int(row["physical_seed"]) for row in rows])
    train_rows = [row for row, group in zip(rows, group_array) if group != held_out]
    train_global = np.flatnonzero(group_array != held_out)
    train_indices = train_global[balanced_indices(train_rows, seed + held_out)]
    test_rows = [row for row, group in zip(rows, group_array) if group == held_out]
    test_global = np.flatnonzero(group_array == held_out)
    test_indices = test_global[balanced_indices(test_rows, seed + 100_000 + held_out)]
    estimator = estimators(len(columns))[model_name]
    print("Fitting fusion {} seed {}".format(model_name, held_out), flush=True)
    estimator.fit(x[train_indices], y[train_indices])
    prediction = estimator.predict(x[test_indices])
    write_json_once(destination, {
        "stage": "fusion", "model": model_name,
        "held_out_physical_seed": held_out,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        **score(y[test_indices], prediction),
    })
    print("Completed {}".format(destination), flush=True)


def assemble_and_finalize(features, output_root, recovery_root):
    folds = output_root / "folds"
    for held_out in MISSING_SEEDS:
        destination = folds / "fusion-{}.json".format(held_out)
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if (existing.get("stage"), existing.get("held_out_physical_seed")) != (
                    "fusion", held_out):
                raise ValueError("conflicting existing fold {}".format(destination))
            if tuple(item.get("model") for item in existing.get("results", [])) != MODEL_NAMES:
                raise ValueError("incomplete existing fold {}".format(destination))
            continue
        results = [validate_part(
            recovery_root / "parts" / "fusion-{}-{}.json".format(held_out, model),
            held_out, model,
        ) for model in MODEL_NAMES]
        write_json_once(destination, {
            "stage": "fusion", "held_out_physical_seed": held_out,
            "results": results,
        })
        print("Assembled {}".format(destination), flush=True)
    finalize(features, output_root, 4401)
    print("CLASSICAL RECOVERY COMPLETE", flush=True)


def check(output_root):
    folds = output_root / "folds"
    for stage in ("R0", "R1", "R2", "R3"):
        for seed in ALL_SEEDS:
            path = folds / "{}-{}.json".format(stage, seed)
            if not path.is_file():
                raise FileNotFoundError("missing completed fold {} seed {}".format(stage, seed))
            value = json.loads(path.read_text(encoding="utf-8"))
            if ((value.get("stage"), value.get("held_out_physical_seed")) != (stage, seed)
                    or tuple(item.get("model") for item in value.get("results", [])) != MODEL_NAMES):
                raise ValueError("invalid completed fold {}".format(path))
    for seed in (3101, 3102):
        path = folds / "fusion-{}.json".format(seed)
        if not path.is_file():
            raise FileNotFoundError("missing completed fusion fold seed {}".format(seed))
        value = json.loads(path.read_text(encoding="utf-8"))
        if ((value.get("stage"), value.get("held_out_physical_seed")) != ("fusion", seed)
                or tuple(item.get("model") for item in value.get("results", [])) != MODEL_NAMES):
            raise ValueError("invalid completed fold {}".format(path))
    for seed in MISSING_SEEDS:
        if (folds / "fusion-{}.json".format(seed)).exists():
            raise ValueError("fusion seed {} is already complete; recovery not needed".format(seed))
    if (output_root / "FROZEN").exists():
        raise ValueError("classical analysis is already frozen")
    for name in ("frozen_classical.joblib", "grouped_results.json"):
        if (output_root / name).exists():
            raise ValueError("partial classical finalization exists: {}".format(name))
    print("PASS: 22 completed folds preserved; only fusion 3103--3105 are missing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "part", "finalize"))
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args(argv)
    if args.mode == "check":
        check(args.output_root)
    elif args.mode == "part":
        if args.features is None or args.recovery_root is None or args.task_index is None:
            parser.error("part requires features, recovery-root and task-index")
        run_part(args.features, args.recovery_root, args.task_index)
    else:
        if args.features is None or args.recovery_root is None:
            parser.error("finalize requires features and recovery-root")
        assemble_and_finalize(args.features, args.output_root, args.recovery_root)


if __name__ == "__main__":
    main()
