"""Train grouped receiver-stage neural diagnostics and freeze the best setup."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset


CLASSES = [
    "normal", "low_snr", "imperfect_csi", "spatial_correlation",
    "literal", "mismatch", "ood",
]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
COARSE = {
    "normal": "normal", "low_snr": "physical", "imperfect_csi": "physical",
    "spatial_correlation": "physical", "literal": "semantic",
    "mismatch": "semantic", "ood": "semantic",
}
STAGE_KEYS = {
    "R0": "R0_noisy_received", "R1": "R1_equalized_decoder_input",
    "R2": "R2_channel_decoder_output", "R3": "R3_free_running_decoder_hidden",
}
INPUT_DIMS = {"R0": 4, "R1": 16, "R2": 128, "R3": 128}


@lru_cache(maxsize=None)
def load_artifact(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def scan_records(root, allow_holdout=False):
    records = []
    for path in sorted(Path(root).rglob("artifacts/*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("holdout") and not allow_holdout:
            raise ValueError("development root contains holdout artifact: {}".format(path))
        label = artifact.get("diagnosis_label")
        if allow_holdout and artifact.get("holdout"):
            label = "ood"
        if label not in CLASSES:
            continue
        for index, sample_id in enumerate(artifact["sample_ids"].tolist()):
            records.append({
                "path": str(path), "index": index, "sample_id": sample_id,
                "label": label, "physical_seed": int(artifact["physical_seed"]),
                "severity": artifact.get("severity"),
            })
    if not records:
        raise FileNotFoundError("no eligible development artifacts under {}".format(root))
    return records


def stage_tensor(artifact, index, stage):
    source_length = int(artifact["source_lengths"][index])
    generated_length = int(artifact["R3_generated_lengths"][index])
    length = source_length * 4 if stage == "R0" else (
        generated_length if stage == "R3" else source_length
    )
    value = artifact[STAGE_KEYS[stage]][index, :length]
    if value.is_complex():
        value = torch.view_as_real(value).flatten(-2)
    return value.float()


class TraceDataset(Dataset):
    def __init__(self, records, stage):
        self.records = records
        self.stage = stage

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        artifact = load_artifact(record["path"])
        stages = ("R0", "R1", "R2", "R3") if self.stage == "fusion" else (self.stage,)
        values = {stage: stage_tensor(artifact, record["index"], stage) for stage in stages}
        return values, CLASS_TO_INDEX[record["label"]]


def collate(batch):
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    stages = batch[0][0].keys()
    values, masks = {}, {}
    for stage in stages:
        sequences = [item[0][stage] for item in batch]
        maximum = max(sequence.size(0) for sequence in sequences)
        tensor = torch.zeros(len(batch), maximum, sequences[0].size(1))
        mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            tensor[row, :sequence.size(0)] = sequence
            mask[row, :sequence.size(0)] = True
        values[stage], masks[stage] = tensor, mask
    return values, masks, labels


def masked_pool(sequence, mask):
    mask_float = mask.unsqueeze(-1).type_as(sequence)
    mean = (sequence * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1)
    maximum = sequence.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, width=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.output_dim = 2 * width

    def forward(self, value, mask):
        encoded = self.network(value.transpose(1, 2)).transpose(1, 2)
        return masked_pool(encoded, mask)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.convolution = nn.Conv1d(
            width, width, 3, padding=dilation, dilation=dilation
        )
        self.normalization = nn.BatchNorm1d(width)

    def forward(self, value):
        return functional.relu(value + self.normalization(self.convolution(value)))


class TCNEncoder(nn.Module):
    def __init__(self, input_dim, width=64):
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 1)
        self.blocks = nn.Sequential(*[
            ResidualDilatedBlock(width, dilation) for dilation in (1, 2, 4, 8)
        ])
        self.output_dim = 2 * width

    def forward(self, value, mask):
        encoded = self.blocks(self.input(value.transpose(1, 2))).transpose(1, 2)
        return masked_pool(encoded, mask)


class ConvTransformerEncoder(nn.Module):
    def __init__(self, input_dim, width=64):
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 3, padding=1)
        layer = nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.output_dim = 2 * width

    def forward(self, value, mask):
        encoded = functional.relu(self.input(value.transpose(1, 2)).transpose(1, 2))
        positions = torch.arange(encoded.size(1), device=encoded.device, dtype=encoded.dtype)
        frequencies = torch.exp(
            torch.arange(0, encoded.size(2), 2, device=encoded.device, dtype=encoded.dtype)
            * (-np.log(10000.0) / encoded.size(2))
        )
        positional = torch.zeros_like(encoded)
        positional[:, :, 0::2] = torch.sin(positions[:, None] * frequencies)
        positional[:, :, 1::2] = torch.cos(positions[:, None] * frequencies)
        encoded = self.transformer(encoded + positional, src_key_padding_mask=~mask)
        return masked_pool(encoded, mask)


def encoder(architecture, input_dim):
    if architecture == "cnn":
        return CNNEncoder(input_dim)
    if architecture == "tcn":
        return TCNEncoder(input_dim)
    if architecture == "conv_transformer":
        return ConvTransformerEncoder(input_dim)
    raise ValueError("unknown architecture {}".format(architecture))


class StageClassifier(nn.Module):
    def __init__(self, architecture, stage):
        super().__init__()
        self.stage = stage
        self.encoder = encoder(architecture, INPUT_DIMS[stage])
        self.classifier = nn.Linear(self.encoder.output_dim, len(CLASSES))

    def forward(self, values, masks):
        return self.classifier(self.encoder(values[self.stage], masks[self.stage]))


class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({
            stage: CNNEncoder(INPUT_DIMS[stage], width=32)
            for stage in ("R0", "R1", "R2", "R3")
        })
        size = sum(item.output_dim for item in self.encoders.values())
        self.classifier = nn.Sequential(
            nn.Linear(size, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, values, masks):
        embeddings = [
            self.encoders[stage](values[stage], masks[stage])
            for stage in ("R0", "R1", "R2", "R3")
        ]
        return self.classifier(torch.cat(embeddings, dim=-1))


def make_model(architecture, stage):
    if architecture == "fusion":
        if stage != "fusion":
            raise ValueError("fusion architecture requires fusion stage")
        return FusionClassifier()
    if stage == "fusion":
        raise ValueError("only fusion architecture accepts fusion stage")
    return StageClassifier(architecture, stage)


def balanced_records(records, seed):
    rng = random.Random(seed)
    grouped = {label: [] for label in CLASSES}
    for record in records:
        grouped[record["label"]].append(record)
    target = min(map(len, grouped.values()))
    output = []
    for label, candidates in grouped.items():
        severity_groups = {}
        if label in {"imperfect_csi", "spatial_correlation"}:
            for record in candidates:
                severity_groups.setdefault(record["severity"], []).append(record)
        if len(severity_groups) > 1:
            chosen = []
            per = target // len(severity_groups)
            for severity in sorted(severity_groups):
                pool = severity_groups[severity][:]
                rng.shuffle(pool)
                chosen.extend(pool[:per])
            chosen_ids = {id(record) for record in chosen}
            remainder = [record for record in candidates if id(record) not in chosen_ids]
            rng.shuffle(remainder)
            chosen.extend(remainder[:target - len(chosen)])
        else:
            chosen = candidates[:]
            rng.shuffle(chosen)
            chosen = chosen[:target]
        output.extend(chosen)
    rng.shuffle(output)
    return output


def move(values, masks, labels, device):
    return (
        {key: value.to(device) for key, value in values.items()},
        {key: value.to(device) for key, value in masks.items()},
        labels.to(device),
    )


def train_model(model, train_records, stage, args, device):
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TraceDataset(train_records, stage), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate, num_workers=0, generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        model.train()
        for values, masks, labels in loader:
            values, masks, labels = move(values, masks, labels, device)
            optimizer.zero_grad()
            loss = functional.cross_entropy(model(values, masks), labels)
            loss.backward()
            optimizer.step()
    return model


def predictions(model, records, stage, batch_size, device):
    loader = DataLoader(
        TraceDataset(records, stage), batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    actual, predicted = [], []
    model.eval()
    with torch.no_grad():
        for values, masks, labels in loader:
            values, masks, labels = move(values, masks, labels, device)
            actual.extend(labels.cpu().tolist())
            predicted.extend(model(values, masks).argmax(dim=-1).cpu().tolist())
    return actual, predicted


def metrics(actual, predicted):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    actual_labels = [CLASSES[index] for index in actual]
    predicted_labels = [CLASSES[index] for index in predicted]
    coarse_actual = [COARSE[label] for label in actual_labels]
    coarse_predicted = [COARSE[label] for label in predicted_labels]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(actual_labels, predicted_labels)),
        "macro_f1": float(f1_score(actual_labels, predicted_labels, average="macro")),
        "per_class_recall": dict(zip(
            CLASSES,
            map(float, recall_score(actual_labels, predicted_labels, labels=CLASSES, average=None, zero_division=0)),
        )),
        "confusion_matrix": confusion_matrix(actual_labels, predicted_labels, labels=CLASSES).tolist(),
        "coarse_balanced_accuracy": float(balanced_accuracy_score(coarse_actual, coarse_predicted)),
        "coarse_macro_f1": float(f1_score(coarse_actual, coarse_predicted, average="macro")),
    }


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fold_mode(args):
    records = scan_records(args.trace_root)
    seeds = sorted(set(record["physical_seed"] for record in records))
    if args.held_out_seed not in seeds:
        raise ValueError("held-out seed is absent")
    train = balanced_records(
        [record for record in records if record["physical_seed"] != args.held_out_seed],
        args.seed + args.held_out_seed,
    )
    test = balanced_records(
        [record for record in records if record["physical_seed"] == args.held_out_seed],
        args.seed + 100_000 + args.held_out_seed,
    )
    device = torch.device(args.device)
    setup_seed(args.seed + args.held_out_seed)
    model = make_model(args.architecture, args.stage).to(device)
    train_model(model, train, args.stage, args, device)
    actual, predicted = predictions(model, test, args.stage, args.batch_size, device)
    report = {
        "architecture": args.architecture, "stage": args.stage,
        "held_out_physical_seed": args.held_out_seed,
        "train_samples": len(train), "test_samples": len(test),
        **metrics(actual, predicted),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def freeze_mode(args):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in Path(args.fold_root).rglob("*.json")]
    grouped = {}
    for report in reports:
        key = (report["architecture"], report["stage"])
        grouped.setdefault(key, []).append(report)
    if not grouped:
        raise FileNotFoundError("no neural fold reports")
    aggregate = [
        {
            "architecture": key[0], "stage": key[1],
            "macro_f1_mean": float(np.mean([item["macro_f1"] for item in values])),
            "balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in values])),
            "folds": len(values),
        }
        for key, values in grouped.items()
    ]
    best = max(aggregate, key=lambda item: item["macro_f1_mean"])
    records = balanced_records(scan_records(args.trace_root), args.seed)
    device = torch.device(args.device)
    setup_seed(args.seed)
    model = make_model(best["architecture"], best["stage"]).to(device)
    train_model(model, records, best["stage"], args, device)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "architecture": best["architecture"],
        "stage": best["stage"], "classes": CLASSES, "selection": best,
    }, destination)
    destination.with_suffix(".selection.json").write_text(
        json.dumps({"aggregate": aggregate, "best": best}, indent=2) + "\n",
        encoding="utf-8",
    )
    destination.with_suffix(".FROZEN").write_text(
        "Model selection completed without arXiv holdout data.\n", encoding="utf-8"
    )
    print(json.dumps(best, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("fold", "freeze"))
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--architecture", choices=("cnn", "tcn", "conv_transformer", "fusion"))
    parser.add_argument("--stage", choices=("R0", "R1", "R2", "R3", "fusion"))
    parser.add_argument("--held-out-seed", type=int)
    parser.add_argument("--fold-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=3e-4, type=float)
    parser.add_argument("--seed", default=5501, type=int)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.mode == "fold":
        if args.architecture is None or args.stage is None or args.held_out_seed is None:
            parser.error("fold mode requires architecture, stage, and held-out-seed")
        fold_mode(args)
    else:
        if not args.fold_root:
            parser.error("freeze mode requires --fold-root")
        freeze_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
