"""Per-sentence probabilities for the frozen neural-fusion protocol, plus the analyses the paper needs.

fold mode (GPU, one held-out seed per call):
    Same training procedure, sampling seeds and hyperparameters as
    analysis/neural_diagnostics.py --mode fold --architecture fusion --stage fusion,
    but the held-out seed's sentences are all scored and their softmax probabilities saved.
    Rows used by the original balanced test set are flagged (in_balanced_test=1), so the
    original fold metrics can be checked for reproduction.

analyze mode (CPU, after all folds):
    1. reproduction check against analysis/neural_folds/fusion_fusion_seed*.json
    2. per-severity recall (imperfect CSI, spatial correlation)
    3. calibration (expected calibration error, 15 bins)
    4. accuracy versus observation window: probabilities of k consecutive sentences from
       the same batch (same channel realization) are averaged; k fixed in advance
    5. if batch_channel.csv from paper_metrics.py is given: recall by channel condition-number tercile
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

CLASSES = ["normal", "low_snr", "imperfect_csi", "spatial_correlation", "literal", "mismatch", "ood"]
COARSE = {"normal": "normal", "low_snr": "physical", "imperfect_csi": "physical",
          "spatial_correlation": "physical", "literal": "semantic", "mismatch": "semantic", "ood": "semantic"}
WINDOWS = [1, 2, 4, 8, 16, 32, 64, 128]   # fixed before looking at results


# ---------------------------------------------------------------- fold (GPU)
def fold(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from analysis import neural_diagnostics as nd

    records = nd.scan_records(args.trace_root)
    held = args.held_out_seed
    train = nd.balanced_records([r for r in records if r["physical_seed"] != held], args.seed + held)
    balanced_test = nd.balanced_records([r for r in records if r["physical_seed"] == held],
                                        args.seed + 100_000 + held)
    balanced_keys = {(r["path"], r["index"]) for r in balanced_test}
    full_test = [r for r in records if r["physical_seed"] == held]

    device = torch.device(args.device)
    nd.setup_seed(args.seed + held)
    model = nd.make_model("fusion", "fusion").to(device)
    nd.train_model(model, train, "fusion", args, device)

    loader = DataLoader(nd.TraceDataset(full_test, "fusion"), batch_size=args.batch_size,
                        shuffle=False, collate_fn=nd.collate, num_workers=0)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for values, masks, labels in loader:
            values, masks, labels = nd.move(values, masks, labels, device)
            probabilities.extend(F.softmax(model(values, masks), dim=-1).cpu().tolist())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["held_out_seed", "path", "index", "sample_id", "batch_index", "label",
                         "severity", "in_balanced_test"] + ["p_" + c for c in CLASSES])
        for record, p in zip(full_test, probabilities):
            batch_index = int(Path(record["path"]).stem.split("_")[-1])
            writer.writerow([held, record["path"], record["index"], record["sample_id"], batch_index,
                             record["label"], "" if record["severity"] is None else record["severity"],
                             int((record["path"], record["index"]) in balanced_keys)]
                            + ["{:.6g}".format(x) for x in p])
    print("wrote {} rows ({} in original balanced test)".format(len(full_test), len(balanced_keys)))


# ---------------------------------------------------------------- analysis (CPU, numpy only)
def read_probs(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            for r in csv.DictReader(handle):
                r["probs"] = [float(r["p_" + c]) for c in CLASSES]
                r["held_out_seed"] = int(r["held_out_seed"])
                r["batch_index"] = int(r["batch_index"])
                r["index"] = int(r["index"])
                r["in_balanced_test"] = int(r["in_balanced_test"])
                rows.append(r)
    return rows


def scores(actual, predicted, labels):
    """Balanced accuracy, macro-F1, per-class recall, confusion matrix."""
    index = {c: i for i, c in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for a, p in zip(actual, predicted):
        cm[index[a]][index[p]] += 1
    recall, f1 = {}, []
    for i, c in enumerate(labels):
        support = sum(cm[i])
        tp = cm[i][i]
        pred = sum(cm[j][i] for j in range(len(labels)))
        if support == 0:
            continue
        r = tp / support
        p = tp / pred if pred else 0.0
        recall[c] = r
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"balanced_accuracy": sum(recall.values()) / len(recall),
            "macro_f1": sum(f1) / len(f1), "per_class_recall": recall, "confusion_matrix": cm}


def argmax_label(probs):
    return CLASSES[max(range(len(probs)), key=lambda i: probs[i])]


def mean_std(values):
    m = sum(values) / len(values)
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return m, s


def per_fold(rows, fn):
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["held_out_seed"]].append(r)
    return {seed: fn(group) for seed, group in sorted(by_seed.items())}


def ece(rows, bins=15):
    edges = [i / bins for i in range(bins + 1)]
    total, error = len(rows), 0.0
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [r for r in rows if lo < max(r["probs"]) <= hi or (lo == 0 and max(r["probs"]) == 0)]
        if not bucket:
            continue
        conf = sum(max(r["probs"]) for r in bucket) / len(bucket)
        acc = sum(argmax_label(r["probs"]) == r["label"] for r in bucket) / len(bucket)
        error += len(bucket) / total * abs(acc - conf)
        table.append({"bin": "({:.2f},{:.2f}]".format(lo, hi), "count": len(bucket),
                      "confidence": conf, "accuracy": acc})
    return error, table


def windowed(rows, k):
    """Average probabilities over k consecutive sentences within one batch (one channel realization)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["held_out_seed"], r["path"])].append(r)
    out = []
    for (seed, _), group in groups.items():
        group.sort(key=lambda r: r["index"])
        for start in range(0, len(group) - k + 1, k):
            chunk = group[start:start + k]
            avg = [sum(r["probs"][j] for r in chunk) / k for j in range(len(CLASSES))]
            out.append({"held_out_seed": seed, "label": chunk[0]["label"], "probs": avg,
                        "severity": chunk[0]["severity"]})
    return out


def analyze(args):
    paths = sorted(Path(args.probs_dir).glob("fusion_probs_seed*.csv"))
    if len(paths) != 5:
        print("WARNING: expected 5 fold files, found {}".format(len(paths)))
    rows = read_probs(paths)
    report = {"fold_files": [str(p) for p in paths], "windows": WINDOWS}

    # 1. reproduction check
    repro = {}
    for seed, group in per_fold([r for r in rows if r["in_balanced_test"]],
                                lambda g: scores([r["label"] for r in g],
                                                 [argmax_label(r["probs"]) for r in g], CLASSES)).items():
        original = Path(args.original_folds) / "fusion_fusion_seed{}.json".format(seed)
        entry = {"reproduced_balanced_accuracy": group["balanced_accuracy"],
                 "reproduced_macro_f1": group["macro_f1"]}
        if original.exists():
            o = json.loads(original.read_text())
            entry.update({"original_balanced_accuracy": o["balanced_accuracy"],
                          "original_macro_f1": o["macro_f1"],
                          "identical_confusion_matrix": o["confusion_matrix"] == group["confusion_matrix"]})
        repro[seed] = entry
    report["reproduction"] = repro

    # 2. full held-out seed, sentence level, plus coarse and per-severity
    def full_scores(g):
        fine = scores([r["label"] for r in g], [argmax_label(r["probs"]) for r in g], CLASSES)
        coarse = scores([COARSE[r["label"]] for r in g], [COARSE[argmax_label(r["probs"])] for r in g],
                        ["normal", "physical", "semantic"])
        return fine, coarse
    folds = per_fold(rows, full_scores)
    report["sentence_level_full"] = {
        "balanced_accuracy": mean_std([f[0]["balanced_accuracy"] for f in folds.values()]),
        "macro_f1": mean_std([f[0]["macro_f1"] for f in folds.values()]),
        "coarse_balanced_accuracy": mean_std([f[1]["balanced_accuracy"] for f in folds.values()]),
        "per_class_recall": {c: mean_std([f[0]["per_class_recall"][c] for f in folds.values()])
                             for c in CLASSES},
    }
    severity = defaultdict(list)
    for seed, group in per_fold(rows, lambda g: g).items():
        cell = defaultdict(lambda: [0, 0])
        for r in group:
            if r["label"] in ("imperfect_csi", "spatial_correlation"):
                pred = argmax_label(r["probs"])
                cell[(r["label"], r["severity"])][0] += int(pred == r["label"])
                cell[(r["label"], r["severity"])][1] += 1
                cell[(r["label"], r["severity"], "predicted_normal")][0] += int(pred == "normal")
                cell[(r["label"], r["severity"], "predicted_normal")][1] += 1
        for key, (hit, n) in cell.items():
            severity[key].append(hit / n)
    report["recall_by_severity"] = {
        "{}|{}".format(k[0], k[1]): {"recall": mean_std(v),
                                     "share_predicted_normal": mean_std(severity[k + ("predicted_normal",)])}
        for k, v in severity.items() if len(k) == 2}

    # 3. calibration
    ece_values = [ece(g)[0] for g in per_fold(rows, lambda g: g).values()]
    report["expected_calibration_error"] = mean_std(ece_values)
    report["reliability_table_all_folds"] = ece(rows)[1]

    # 4. observation window
    window_table = []
    for k in WINDOWS:
        w = windowed(rows, k)
        if not w:
            continue
        fw = per_fold(w, lambda g: (
            scores([r["label"] for r in g], [argmax_label(r["probs"]) for r in g], CLASSES),
            scores([COARSE[r["label"]] for r in g], [COARSE[argmax_label(r["probs"])] for r in g],
                   ["normal", "physical", "semantic"])))
        row = {"window_sentences": k, "decisions": len(w),
               "balanced_accuracy": mean_std([f[0]["balanced_accuracy"] for f in fw.values()]),
               "macro_f1": mean_std([f[0]["macro_f1"] for f in fw.values()]),
               "coarse_balanced_accuracy": mean_std([f[1]["balanced_accuracy"] for f in fw.values()])}
        for c in CLASSES:
            row["recall_" + c] = mean_std([f[0]["per_class_recall"].get(c, float("nan")) for f in fw.values()])
        window_table.append(row)
    report["observation_window"] = window_table

    # 5. channel condition number
    if args.batch_channel and Path(args.batch_channel).exists():
        cond = {}
        with open(args.batch_channel, newline="") as handle:
            for r in csv.DictReader(handle):
                cond[r["artifact"].split("traces/")[-1]] = float(r["condition_number"])
        by_label = defaultdict(list)
        for r in rows:
            key = r["path"].split("traces/")[-1]
            if key in cond:
                by_label[r["label"]].append((cond[key], argmax_label(r["probs"]) == r["label"]))
        terciles = {}
        for label, items in by_label.items():
            items.sort()
            n = len(items)
            parts = [items[:n // 3], items[n // 3:2 * n // 3], items[2 * n // 3:]]
            terciles[label] = [{"tercile": i + 1, "cond_range": [p[0][0], p[-1][0]],
                                "recall": sum(h for _, h in p) / len(p)} for i, p in enumerate(parts) if p]
        report["recall_by_condition_number_tercile"] = terciles

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion_analysis.json").write_text(json.dumps(report, indent=2))
    with (out / "observation_window.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_sentences", "decisions", "balanced_accuracy_mean", "balanced_accuracy_std",
                         "macro_f1_mean", "macro_f1_std", "coarse_ba_mean", "coarse_ba_std"]
                        + ["recall_" + c for c in CLASSES])
        for r in window_table:
            writer.writerow([r["window_sentences"], r["decisions"], *r["balanced_accuracy"], *r["macro_f1"],
                             *r["coarse_balanced_accuracy"]] + [r["recall_" + c][0] for c in CLASSES])
    print(json.dumps({k: report[k] for k in ("reproduction", "sentence_level_full",
                                             "expected_calibration_error")}, indent=1))
    for r in window_table:
        print("k={:>3}  BA={:.3f}±{:.3f}  macroF1={:.3f}  coarseBA={:.3f}".format(
            r["window_sentences"], *r["balanced_accuracy"], r["macro_f1"][0], r["coarse_balanced_accuracy"][0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("fold", "analyze"))
    p.add_argument("--trace-root")
    p.add_argument("--held-out-seed", type=int)
    p.add_argument("--output")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", default=12, type=int)        # identical to neural_diagnostics defaults
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--learning-rate", default=3e-4, type=float)
    p.add_argument("--seed", default=5501, type=int)
    p.add_argument("--probs-dir")
    p.add_argument("--original-folds")
    p.add_argument("--batch-channel")
    p.add_argument("--output-dir")
    args = p.parse_args()
    fold(args) if args.mode == "fold" else analyze(args)


if __name__ == "__main__":
    main()
