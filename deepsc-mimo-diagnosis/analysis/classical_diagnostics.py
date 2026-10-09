"""Grouped classical diagnosis with strict holdout exclusion and class balance."""

import argparse
import csv
import json
from pathlib import Path
import random

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


CLASSES = [
    "normal", "low_snr", "imperfect_csi", "spatial_correlation",
    "literal", "mismatch", "ood",
]
META = {
    "artifact", "sample_id", "profile", "label", "holdout", "domain",
    "severity_name", "severity", "snr_db", "physical_seed", "batch_index",
    "source_length", "generated_length",
}


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Greedy relevance-minus-redundancy selector using ANOVA relevance."""

    def __init__(self, k=64):
        self.k = k

    def fit(self, x, y):
        scores, _ = f_classif(x, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = scores / max(float(scores.max()), 1e-12)
        correlation = np.nan_to_num(np.abs(np.corrcoef(x, rowvar=False)), nan=0.0)
        selected = [int(np.argmax(normalized))]
        candidates = set(range(x.shape[1])) - set(selected)
        while candidates and len(selected) < min(self.k, x.shape[1]):
            best = max(
                candidates,
                key=lambda index: normalized[index] - correlation[index, selected].mean(),
            )
            selected.append(best)
            candidates.remove(best)
        self.indices_ = np.asarray(selected, dtype=np.int64)
        return self

    def transform(self, x):
        return x[:, self.indices_]


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty feature CSV")
    if any(row["holdout"] not in {"0", "False", "false", ""} for row in rows):
        raise ValueError("development feature CSV contains holdout rows")
    rows = [row for row in rows if row["label"] in CLASSES]
    if set(row["label"] for row in rows) != set(CLASSES):
        raise ValueError("development data must contain all seven classes")
    return rows


def feature_columns(rows, stage):
    candidates = sorted(set(rows[0]) - META)
    if stage == "fusion":
        return [name for name in candidates if name.startswith(("R0__", "R1__", "R2__", "R3__", "T__"))]
    return [name for name in candidates if name.startswith(stage + "__")]


def matrix(rows, columns):
    values = []
    for row in rows:
        current = []
        for column in columns:
            try:
                current.append(float(row.get(column, "nan")))
            except (TypeError, ValueError):
                current.append(float("nan"))
        values.append(current)
    return np.asarray(values, dtype=np.float64)


def balanced_indices(rows, seed):
    rng = random.Random(seed)
    grouped = {label: [] for label in CLASSES}
    for index, row in enumerate(rows):
        grouped[row["label"]].append(index)
    target = min(map(len, grouped.values()))
    selected = []
    for label in CLASSES:
        candidates = grouped[label]
        severity_groups = {}
        if label in {"imperfect_csi", "spatial_correlation"}:
            for index in candidates:
                severity_groups.setdefault(rows[index]["severity"], []).append(index)
        if len(severity_groups) > 1:
            per_severity = target // len(severity_groups)
            chosen = []
            for severity in sorted(severity_groups):
                pool = severity_groups[severity][:]
                rng.shuffle(pool)
                chosen.extend(pool[:per_severity])
            remaining = [index for index in candidates if index not in set(chosen)]
            rng.shuffle(remaining)
            chosen.extend(remaining[:target - len(chosen)])
        else:
            chosen = candidates[:]
            rng.shuffle(chosen)
            chosen = chosen[:target]
        selected.extend(chosen)
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def estimators(feature_count):
    k = max(1, min(64, feature_count))
    return {
        "anova_logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=k)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "mrmr_linear_svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", MRMRSelector(k=k)),
            ("classifier", LinearSVC(class_weight="balanced", max_iter=5000)),
        ]),
        "elasticnet_logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                max_iter=3000, class_weight="balanced",
            )),
        ]),
    }


def score(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_recall": {
            label: float(value) for label, value in zip(
                CLASSES, recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", default=4401, type=int)
    args = parser.parse_args(argv)
    rows = load_csv(args.features)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    groups = sorted(set(int(row["physical_seed"]) for row in rows))
    if len(groups) < 2:
        raise ValueError("grouped evaluation requires at least two physical seeds")
    results = []
    for stage in ("R0", "R1", "R2", "R3", "fusion"):
        columns = feature_columns(rows, stage)
        if not columns:
            raise ValueError("no features for stage {}".format(stage))
        x = matrix(rows, columns)
        y = np.asarray([row["label"] for row in rows])
        group_array = np.asarray([int(row["physical_seed"]) for row in rows])
        for held_out in groups:
            train_rows = [row for row, group in zip(rows, group_array) if group != held_out]
            train_global = np.flatnonzero(group_array != held_out)
            local_balanced = balanced_indices(train_rows, args.seed + held_out)
            train_indices = train_global[local_balanced]
            test_rows = [row for row, group in zip(rows, group_array) if group == held_out]
            test_global = np.flatnonzero(group_array == held_out)
            test_local_balanced = balanced_indices(
                test_rows, args.seed + 100_000 + held_out
            )
            test_indices = test_global[test_local_balanced]
            for name, estimator in estimators(len(columns)).items():
                estimator.fit(x[train_indices], y[train_indices])
                prediction = estimator.predict(x[test_indices])
                metrics = score(y[test_indices], prediction)
                results.append({
                    "stage": stage, "model": name, "held_out_physical_seed": held_out,
                    "train_samples": int(len(train_indices)), "test_samples": int(len(test_indices)),
                    **metrics,
                })
    aggregate = []
    for stage in ("R0", "R1", "R2", "R3", "fusion"):
        for name in estimators(1):
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
    all_balanced = balanced_indices(rows, args.seed)
    frozen = estimators(len(columns))[best["model"]]
    frozen.fit(x[all_balanced], np.asarray([row["label"] for row in rows])[all_balanced])
    joblib.dump(
        {"pipeline": frozen, "columns": columns, "classes": CLASSES, "selection": best},
        output / "frozen_classical.joblib",
    )
    (output / "grouped_results.json").write_text(
        json.dumps({"folds": results, "aggregate": aggregate, "best": best}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "FROZEN").write_text(
        "Model selection completed without arXiv holdout data.\n", encoding="utf-8"
    )
    print(json.dumps(best, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
