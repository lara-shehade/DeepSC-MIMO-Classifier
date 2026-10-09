"""Finish the selected full-data classical fit in two warm-start chunks."""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np

from analysis.classical_diagnostics import (
    CLASSES, balanced_indices, estimators, feature_columns, load_csv, matrix,
)


STAGES = ("R0", "R1", "R2", "R3", "fusion")
SEEDS = (3101, 3102, 3103, 3104, 3105)
TOTAL_CHUNKS = 2
ITERATIONS_PER_CHUNK = 1500


def atomic_joblib(path, value):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fold_summary(output_root):
    expected_models = tuple(estimators(1))
    results = []
    for stage in STAGES:
        for held_out in SEEDS:
            path = output_root / "folds" / "{}-{}.json".format(stage, held_out)
            report = json.loads(path.read_text(encoding="utf-8"))
            if (report.get("stage"), report.get("held_out_physical_seed")) != (stage, held_out):
                raise ValueError("incorrect fold result: {}".format(path))
            fold_results = report.get("results", [])
            if tuple(item.get("model") for item in fold_results) != expected_models:
                raise ValueError("missing or reordered models: {}".format(path))
            if any((item.get("stage"), item.get("held_out_physical_seed")) !=
                   (stage, held_out) for item in fold_results):
                raise ValueError("mismatched fold contents: {}".format(path))
            results.extend(fold_results)
    aggregate = []
    for stage in STAGES:
        for name in expected_models:
            subset = [row for row in results
                      if row["stage"] == stage and row["model"] == name]
            aggregate.append({
                "stage": stage,
                "model": name,
                "balanced_accuracy_mean": float(np.mean([
                    row["balanced_accuracy"] for row in subset
                ])),
                "macro_f1_mean": float(np.mean([
                    row["macro_f1"] for row in subset
                ])),
            })
    best = max(aggregate, key=lambda row: row["macro_f1_mean"])
    return results, aggregate, best


def check(output_root):
    if (output_root / "FROZEN").exists():
        print("Classical analysis is already frozen; recovery is unnecessary")
        return 3
    for name in ("frozen_classical.joblib", "grouped_results.json"):
        if (output_root / name).exists():
            raise FileExistsError("unexpected partial final output: {}".format(name))
    _, _, best = fold_summary(output_root)
    if (best.get("stage"), best.get("model")) != ("fusion", "elasticnet_logistic"):
        raise ValueError("selected model changed: {}".format(best))
    print("PASS: all 25 folds valid; selected model is fusion elasticnet_logistic")
    return 0


def run_chunk(features, output_root, recovery_root, chunk_index, seed=4401):
    if not 0 <= chunk_index < TOTAL_CHUNKS:
        raise ValueError("chunk index must be 0 or 1")
    if (output_root / "FROZEN").exists():
        raise FileExistsError("classical model is already frozen")
    results, aggregate, best = fold_summary(output_root)
    if (best.get("stage"), best.get("model")) != ("fusion", "elasticnet_logistic"):
        raise ValueError("selected model changed: {}".format(best))
    state_path = recovery_root / "freeze_state.joblib"
    rows = load_csv(features)
    columns = feature_columns(rows, best["stage"])
    selected = balanced_indices(rows, seed)
    x = matrix(rows, columns)[selected]
    y = np.asarray([row["label"] for row in rows])[selected]
    if chunk_index == 0:
        if state_path.exists():
            state = joblib.load(state_path)
            if state.get("chunks_completed") != 1:
                raise ValueError("invalid existing chunk checkpoint")
            print("Chunk 1 checkpoint already complete", flush=True)
            return
        pipeline = estimators(len(columns))[best["model"]]
        classifier = pipeline.named_steps["classifier"]
        classifier.set_params(warm_start=True, max_iter=ITERATIONS_PER_CHUNK)
    else:
        if not state_path.is_file():
            raise FileNotFoundError("chunk 1 checkpoint is missing")
        state = joblib.load(state_path)
        if (state.get("chunks_completed"), state.get("columns"), state.get("selection")) != (
                1, columns, best):
            raise ValueError("chunk checkpoint does not match selected model")
        pipeline = state["pipeline"]
        classifier = pipeline.named_steps["classifier"]
        if (not classifier.warm_start or
                classifier.max_iter != ITERATIONS_PER_CHUNK):
            raise ValueError("checkpoint is not a 1500-iteration warm-start fit")
    print("Fitting frozen model chunk {}/{}".format(
        chunk_index + 1, TOTAL_CHUNKS), flush=True)
    pipeline.fit(x, y)
    iterations = int(np.max(pipeline.named_steps["classifier"].n_iter_))
    if chunk_index == 0:
        atomic_joblib(state_path, {
            "pipeline": pipeline,
            "columns": columns,
            "classes": CLASSES,
            "selection": best,
            "chunks_completed": 1,
            "iterations_in_chunk": iterations,
        })
        print("Saved resumable checkpoint after chunk 1", flush=True)
        return
    model_path = output_root / "frozen_classical.joblib"
    result_path = output_root / "grouped_results.json"
    marker_path = output_root / "FROZEN"
    for path in (model_path, result_path, marker_path):
        if path.exists():
            raise FileExistsError(path)
    atomic_joblib(model_path, {
        "pipeline": pipeline,
        "columns": columns,
        "classes": CLASSES,
        "selection": best,
        "fit_execution": {
            "method": "two_job_warm_start",
            "chunks": TOTAL_CHUNKS,
            "max_iterations_per_chunk": ITERATIONS_PER_CHUNK,
        },
    })
    atomic_text(result_path, json.dumps({
        "folds": results, "aggregate": aggregate, "best": best,
    }, indent=2) + "\n")
    atomic_text(marker_path,
                "Model selection completed without arXiv holdout data.\n")
    print(json.dumps(best, sort_keys=True), flush=True)
    print("CLASSICAL FREEZE RECOVERY COMPLETE", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "chunk"))
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--chunk-index", type=int)
    args = parser.parse_args(argv)
    if args.mode == "check":
        return check(args.output_root)
    if args.features is None or args.recovery_root is None or args.chunk_index is None:
        parser.error("chunk requires features, recovery-root and chunk-index")
    run_chunk(args.features, args.output_root, args.recovery_root, args.chunk_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
