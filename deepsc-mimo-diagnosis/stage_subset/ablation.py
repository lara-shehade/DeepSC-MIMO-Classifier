"""Fixed-TCN ablation over saved DeepSC-MIMO receiver-stage subsets.

This is supplementary development-set analysis. It never reads arXiv traces,
does not replace the frozen neural diagnostic, and does not select a new
holdout-tested model.
"""

import argparse
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from analysis.neural_diagnostics import (
    CLASSES,
    CLASS_TO_INDEX,
    INPUT_DIMS,
    StageClassifier,
    TCNEncoder,
    balanced_records,
    collate,
    load_artifact,
    metrics,
    move,
    scan_records,
    setup_seed,
    stage_tensor,
)


STAGES = ("R0", "R1", "R2", "R3")
SEEDS = (3101, 3102, 3103, 3104, 3105)
SUBSETS = tuple(
    subset for size in range(1, 5) for subset in combinations(STAGES, size)
)
MULTI_SUBSETS = tuple(subset for subset in SUBSETS if len(subset) > 1)
TASKS = tuple((subset, seed) for subset in MULTI_SUBSETS for seed in SEEDS)
PILOT_INDEX = TASKS.index((STAGES, SEEDS[0]))
TRAINING = {"epochs": 12, "batch_size": 64, "learning_rate": 3e-4,
            "weight_decay": 1e-4, "seed": 5501}


def subset_name(stages):
    return "_".join(stages)


def record_digest(records):
    digest = hashlib.sha256()
    for record in records:
        item = (
            record["path"], int(record["index"]), int(record["sample_id"]),
            record["label"], int(record["physical_seed"]),
        )
        digest.update((json.dumps(item, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


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


class SubsetDataset(Dataset):
    def __init__(self, records, stages):
        self.records = records
        self.stages = stages

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        artifact = load_artifact(record["path"])
        values = {
            stage: stage_tensor(artifact, record["index"], stage)
            for stage in self.stages
        }
        return values, CLASS_TO_INDEX[record["label"]]


class SubsetTCN(nn.Module):
    """One unchanged TCN encoder per stage, concatenated before a linear head."""

    def __init__(self, stages):
        super().__init__()
        if tuple(stages) not in SUBSETS:
            raise ValueError("invalid receiver-stage subset: {}".format(stages))
        self.stages = tuple(stages)
        self.encoders = nn.ModuleDict({
            stage: TCNEncoder(INPUT_DIMS[stage]) for stage in self.stages
        })
        self.classifier = nn.Linear(
            sum(self.encoders[stage].output_dim for stage in self.stages),
            len(CLASSES),
        )

    def forward(self, values, masks):
        embeddings = [
            self.encoders[stage](values[stage], masks[stage])
            for stage in self.stages
        ]
        return self.classifier(torch.cat(embeddings, dim=-1))


def train(model, records, stages, epochs, device):
    # Identical optimizer, shuffle generator, loss, batch size, and epoch count
    # to analysis.neural_diagnostics.train_model.
    loader = DataLoader(
        SubsetDataset(records, stages), batch_size=TRAINING["batch_size"],
        shuffle=True, collate_fn=collate, num_workers=0,
        generator=torch.Generator().manual_seed(TRAINING["seed"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for values, masks, labels in loader:
            values, masks, labels = move(values, masks, labels, device)
            optimizer.zero_grad()
            loss = functional.cross_entropy(model(values, masks), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print("epoch {}/{} loss_sum {:.6f}".format(epoch + 1, epochs, total), flush=True)


def predict(model, records, stages, device):
    loader = DataLoader(
        SubsetDataset(records, stages), batch_size=TRAINING["batch_size"],
        shuffle=False, collate_fn=collate, num_workers=0,
    )
    actual, predicted = [], []
    model.eval()
    with torch.no_grad():
        for values, masks, labels in loader:
            values, masks, labels = move(values, masks, labels, device)
            actual.extend(labels.cpu().tolist())
            predicted.extend(model(values, masks).argmax(dim=-1).cpu().tolist())
    return actual, predicted


def evaluate_fold(records, stages, held_out_seed, device, epochs=12):
    if held_out_seed not in SEEDS:
        raise ValueError("unapproved physical seed")
    if tuple(stages) not in MULTI_SUBSETS:
        raise ValueError("fold job requires a multi-stage subset")
    train_records = balanced_records(
        [record for record in records if record["physical_seed"] != held_out_seed],
        TRAINING["seed"] + held_out_seed,
    )
    test_records = balanced_records(
        [record for record in records if record["physical_seed"] == held_out_seed],
        TRAINING["seed"] + 100_000 + held_out_seed,
    )
    if set(record["physical_seed"] for record in train_records) & {held_out_seed}:
        raise AssertionError("physical-seed leakage into training")
    setup_seed(TRAINING["seed"] + held_out_seed)
    model = SubsetTCN(stages).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train(model, train_records, stages, epochs, device)
    actual, predicted = predict(model, test_records, stages, device)
    return {
        "architecture": "tcn_branchwise_subset",
        "subset": subset_name(stages),
        "stages": list(stages),
        "held_out_physical_seed": held_out_seed,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "train_records_sha256": record_digest(train_records),
        "test_records_sha256": record_digest(test_records),
        "parameter_count": parameter_count,
        "training": {**TRAINING, "epochs": epochs},
        **metrics(actual, predicted),
    }


def run_fold(trace_root, output_root, task_index, device):
    if not 0 <= task_index < len(TASKS):
        raise ValueError("task index outside 0..{}".format(len(TASKS) - 1))
    stages, held_out_seed = TASKS[task_index]
    destination = output_root / "folds" / "{}_seed{}.json".format(
        subset_name(stages), held_out_seed
    )
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if (previous.get("subset"), previous.get("held_out_physical_seed")) != (
            subset_name(stages), held_out_seed
        ) or "macro_f1" not in previous:
            raise ValueError("existing fold result is invalid: {}".format(destination))
        print("Already complete: {}".format(destination), flush=True)
        return
    records = scan_records(trace_root)
    seeds = tuple(sorted({int(record["physical_seed"]) for record in records}))
    if seeds != SEEDS:
        raise ValueError("expected physical seeds {}, got {}".format(SEEDS, seeds))
    report = evaluate_fold(records, stages, held_out_seed, torch.device(device))
    report["task_index"] = task_index
    write_json_once(destination, report)
    print("Completed {}".format(destination), flush=True)


def summarize(output_root, singleton_root):
    destination = output_root / "stage_subset_report.json"
    if destination.exists():
        raise FileExistsError(destination)
    fold_rows = []
    paired_digests = {}
    paired_counts = {}
    for stages in SUBSETS:
        name = subset_name(stages)
        for seed in SEEDS:
            if len(stages) == 1:
                path = singleton_root / "tcn_{}_seed{}.json".format(stages[0], seed)
                row = json.loads(path.read_text(encoding="utf-8"))
                if (row.get("architecture"), row.get("stage"),
                        row.get("held_out_physical_seed")) != ("tcn", stages[0], seed):
                    raise ValueError("invalid existing singleton report: {}".format(path))
                row = {**row, "subset": name, "stages": list(stages),
                       "parameter_count": sum(
                           parameter.numel() for parameter in
                           StageClassifier("tcn", stages[0]).parameters()
                       ), "source": "existing_neural_fold"}
            else:
                path = output_root / "folds" / "{}_seed{}.json".format(name, seed)
                row = json.loads(path.read_text(encoding="utf-8"))
                if (row.get("subset"), row.get("stages"),
                        row.get("held_out_physical_seed")) != (
                            name, list(stages), seed
                        ):
                    raise ValueError("invalid multi-stage report: {}".format(path))
                if row.get("training") != TRAINING:
                    raise ValueError("changed training protocol in {}".format(path))
                digests = (row.get("train_records_sha256"),
                           row.get("test_records_sha256"))
                if not all(isinstance(value, str) and len(value) == 64 for value in digests):
                    raise ValueError("missing paired sample IDs in {}".format(path))
                if seed in paired_digests and paired_digests[seed] != digests:
                    raise ValueError("different samples across subsets for seed {}".format(seed))
                paired_digests[seed] = digests
                row = {**row, "source": "new_subset_fold"}
            counts = (row["train_samples"], row["test_samples"])
            if seed in paired_counts and paired_counts[seed] != counts:
                raise ValueError("different sample counts across subsets for seed {}".format(seed))
            paired_counts[seed] = counts
            if not 0 <= row["macro_f1"] <= 1 or not 0 <= row["balanced_accuracy"] <= 1:
                raise ValueError("invalid metric in {}".format(path))
            fold_rows.append(row)
    aggregate = []
    for stages in SUBSETS:
        name = subset_name(stages)
        values = [row for row in fold_rows if row["subset"] == name]
        if len(values) != len(SEEDS):
            raise ValueError("missing fold for {}".format(name))
        if len({row["parameter_count"] for row in values}) != 1:
            raise ValueError("inconsistent parameter count for {}".format(name))
        aggregate.append({
            "subset": name,
            "stages": list(stages),
            "folds": len(values),
            "parameter_count": values[0]["parameter_count"],
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in values])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in values], ddof=1)),
            "balanced_accuracy_mean": float(np.mean(
                [row["balanced_accuracy"] for row in values]
            )),
            "balanced_accuracy_std": float(np.std(
                [row["balanced_accuracy"] for row in values], ddof=1
            )),
            "per_class_recall_mean": {
                label: float(np.mean([row["per_class_recall"][label] for row in values]))
                for label in CLASSES
            },
        })
    report = {
        "status": "complete",
        "analysis_role": "supplementary exploratory development-set ablation",
        "holdout_used": False,
        "claim_limit": "The best subset is not an independently validated seven-class estimate.",
        "receiver_stages": list(STAGES),
        "physical_seeds": list(SEEDS),
        "training": TRAINING,
        "folds": fold_rows,
        "aggregate": aggregate,
        "best_exploratory": max(aggregate, key=lambda row: row["macro_f1_mean"]),
    }
    write_json_once(destination, report)
    print("Wrote {}".format(destination), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("fold", "summary"))
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--singleton-root", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.mode == "fold":
        if args.trace_root is None or args.task_index is None:
            parser.error("fold mode requires --trace-root and --task-index")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        run_fold(args.trace_root, args.output_root, args.task_index, args.device)
    else:
        if args.singleton_root is None:
            parser.error("summary mode requires --singleton-root")
        summarize(args.output_root, args.singleton_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
