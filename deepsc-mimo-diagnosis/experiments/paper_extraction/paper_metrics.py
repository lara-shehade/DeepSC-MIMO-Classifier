"""Paper numbers from existing trace artifacts. Read-only on the run; CPU only.

Outputs (in --output-dir):
  sentence_metrics.csv   one row per sentence: BLEU-1, BLEU-4, positional token accuracy,
                         lengths, <UNK> rate, rate of tokens never seen in Europarl training
  batch_channel.csv      one row per batch: channel condition number, CSI error, literal edits
  degradation_table.csv  per profile x severity: mean +/- std across the 5 physical seeds
  degradation_table.md   same, ready to paste
"""
import argparse
import csv
import json
import math
import pickle
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PAD, START, END, UNK = 0, 1, 2, 3
SPECIAL = {PAD, START, END}


# ---------------------------------------------------------------- BLEU
def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu_fallback(reference, hypothesis, max_n, epsilon=0.1):
    """Sentence BLEU with uniform weights; epsilon smoothing (NLTK method1) for zero counts."""
    if not hypothesis:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        hyp = _ngrams(hypothesis, n)
        total = sum(hyp.values())
        if total == 0:
            return 0.0
        ref = _ngrams(reference, n)
        overlap = sum(min(count, ref[gram]) for gram, count in hyp.items())
        precision = (overlap if overlap > 0 else epsilon) / total
        if overlap == 0 and max_n == 1:
            return 0.0
        log_sum += math.log(precision) / max_n
    bp = 1.0 if len(hypothesis) > len(reference) else math.exp(1 - len(reference) / len(hypothesis))
    return bp * math.exp(log_sum)


def make_bleu():
    """Use NLTK exactly as the DeepSC codebase does (utils.BleuScore); fall back if absent."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        smooth = SmoothingFunction().method1

        def bleu(reference, hypothesis, max_n):
            if not hypothesis:
                return 0.0
            weights = tuple([1.0 / max_n] * max_n)
            return float(sentence_bleu([reference], hypothesis, weights=weights,
                                       smoothing_function=smooth if max_n > 1 else None))
        return bleu, "nltk sentence_bleu (BLEU-4 with SmoothingFunction.method1)"
    except Exception:
        return sentence_bleu_fallback, "built-in sentence BLEU (epsilon smoothing)"


def strip(tokens):
    out = []
    for token in tokens:
        if token == END:
            break
        if token not in SPECIAL:
            out.append(int(token))
    return out


def positional_accuracy(reference, hypothesis):
    if not reference:
        return float("nan")
    hits = sum(1 for i, token in enumerate(reference) if i < len(hypothesis) and hypothesis[i] == token)
    return hits / len(reference)


# ---------------------------------------------------------------- helpers
def load_train_vocabulary(shared_root):
    """Token ids that appear in Europarl training data (what checkpoint C actually saw)."""
    if not shared_root:
        return None
    path = Path(shared_root) / "europarl" / "train_data.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        data = pickle.load(handle)
    seen = set()
    for sentence in data:
        try:
            seen.update(int(token) for token in sentence)
        except TypeError:
            return None
    return seen


def channel_stats(artifact):
    import torch
    out = {"condition_number": float("nan"), "csi_relative_error": float("nan")}
    h = artifact.get("true_h")
    if h is None:
        return out
    if not torch.is_complex(h) and h.shape[-1] == 2:
        h = torch.view_as_complex(h.contiguous())
    first = h.reshape(-1, h.shape[-2], h.shape[-1])[0]
    singular = torch.linalg.svdvals(first.to(torch.complex128))
    out["condition_number"] = float(singular.max() / singular.min().clamp_min(1e-30))
    est = artifact.get("estimated_h")
    if est is not None:
        if not torch.is_complex(est) and est.shape[-1] == 2:
            est = torch.view_as_complex(est.contiguous())
        est0 = est.reshape(-1, est.shape[-2], est.shape[-1])[0]
        out["csi_relative_error"] = float(
            torch.linalg.norm(est0 - first) / torch.linalg.norm(first).clamp_min(1e-30))
    return out


def summarize(rows, key, group_keys):
    """Mean of per-seed means, std across seeds."""
    per_seed = defaultdict(list)
    for row in rows:
        value = row[key]
        if value == value:  # skip NaN
            per_seed[tuple(row[k] for k in group_keys) + (row["physical_seed"],)].append(value)
    grouped = defaultdict(list)
    for group_seed, values in per_seed.items():
        grouped[group_seed[:-1]].append(statistics.mean(values))
    result = {}
    for group, seed_means in grouped.items():
        mean = statistics.mean(seed_means)
        std = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
        result[group] = (mean, std, len(seed_means))
    return result


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--shared-root", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import torch

    run_root = Path(args.run_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bleu, bleu_name = make_bleu()
    train_vocab = load_train_vocabulary(args.shared_root)

    artifacts = sorted((run_root / "traces").rglob("artifacts/*.pt"))
    if not artifacts:
        raise SystemExit("no artifacts under {}/traces".format(run_root))

    sentence_rows, batch_rows = [], []
    for count, path in enumerate(artifacts, 1):
        a = torch.load(path, map_location="cpu", weights_only=False)
        profile = a["profile"]
        severity = "" if a.get("severity") is None else a.get("severity")
        seed = int(a["physical_seed"])
        batch = int(a["batch_index"])
        split = "holdout" if a.get("holdout") else "development"

        ref_tok = a["clean_reference_tokens"]
        hyp_tok = a["R3_generated_tokens"]
        hyp_len = a["R3_generated_lengths"]
        src_tok = a["source_tokens"]
        ids = a["sample_ids"].tolist()
        for i, sample_id in enumerate(ids):
            reference = strip(ref_tok[i].tolist())
            hypothesis = strip(hyp_tok[i, :int(hyp_len[i])].tolist())
            source = [int(t) for t in src_tok[i].tolist() if int(t) not in SPECIAL]
            unk = sum(1 for t in source if t == UNK) / max(len(source), 1)
            unseen = (sum(1 for t in source if t not in train_vocab) / max(len(source), 1)
                      if train_vocab is not None else float("nan"))
            sentence_rows.append({
                "split": split, "profile": profile, "severity": severity,
                "physical_seed": seed, "batch_index": batch, "sample_id": int(sample_id),
                "bleu1": bleu(reference, hypothesis, 1), "bleu4": bleu(reference, hypothesis, 4),
                "token_accuracy": positional_accuracy(reference, hypothesis),
                "reference_length": len(reference), "generated_length": len(hypothesis),
                "unk_rate_source": unk, "unseen_in_train_rate_source": unseen,
            })
        stats = channel_stats(a)
        edits = a.get("corruption_metadata") or []
        batch_rows.append({
            "split": split, "profile": profile, "severity": severity,
            "physical_seed": seed, "batch_index": batch,
            "condition_number": stats["condition_number"],
            "csi_relative_error": stats["csi_relative_error"],
            "literal_events": sum(int(e.get("events_applied", 0)) for e in edits),
            "artifact": str(path.relative_to(run_root)),
        })
        if count % 50 == 0:
            print("processed {}/{} artifacts".format(count, len(artifacts)), flush=True)

    for name, rows in (("sentence_metrics.csv", sentence_rows), ("batch_channel.csv", batch_rows)):
        with (out_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    group = ("split", "profile", "severity")
    metrics = ["bleu1", "bleu4", "token_accuracy", "unk_rate_source", "unseen_in_train_rate_source"]
    sentence_summary = {m: summarize(sentence_rows, m, group) for m in metrics}
    channel_summary = {m: summarize(batch_rows, m, group) for m in ("condition_number", "csi_relative_error")}

    keys = sorted(sentence_summary["bleu1"].keys(), key=lambda k: (k[0], k[1], str(k[2])))
    table = []
    for key in keys:
        row = {"split": key[0], "profile": key[1], "severity": key[2]}
        for m in metrics:
            mean, std, seeds = sentence_summary[m].get(key, (float("nan"), float("nan"), 0))
            row[m + "_mean"], row[m + "_std"], row["seeds"] = mean, std, seeds
        for m in ("condition_number", "csi_relative_error"):
            mean, std, _ = channel_summary[m].get(key, (float("nan"), float("nan"), 0))
            row[m + "_mean"], row[m + "_std"] = mean, std
        table.append(row)
    with (out_dir / "degradation_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    def fmt(r, m, d=3):
        return "{:.{d}f} ± {:.{d}f}".format(r[m + "_mean"], r[m + "_std"], d=d)

    lines = [
        "BLEU: " + bleu_name + ". Free-running decoder output vs clean reference. "
        "Mean of per-seed means ± std across physical seeds.", "",
        "| split | profile | severity | BLEU-1 | BLEU-4 | token acc. | source UNK | unseen in train | cond(H) | seeds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["split"], r["profile"], r["severity"], fmt(r, "bleu1"), fmt(r, "bleu4"),
            fmt(r, "token_accuracy"), fmt(r, "unk_rate_source"),
            fmt(r, "unseen_in_train_rate_source"), fmt(r, "condition_number", 2), r["seeds"]))
    (out_dir / "degradation_table.md").write_text("\n".join(lines) + "\n")
    (out_dir / "paper_metrics.meta.json").write_text(json.dumps({
        "artifacts": len(artifacts), "sentences": len(sentence_rows), "bleu": bleu_name,
        "train_vocabulary_loaded": train_vocab is not None,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
