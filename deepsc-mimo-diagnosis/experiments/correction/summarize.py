"""Aggregate correction_v1 folds into paper tables (mean +- std over held-out seeds)."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def ms(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def fmt(pair, d=3):
    return "{:.{d}f} ± {:.{d}f}".format(pair[0], pair[1], d=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    pattern = "fold_seed*_smoke.json" if args.smoke else "fold_seed[0-9][0-9][0-9][0-9].json"
    folds = [json.loads(p.read_text()) for p in sorted((Path(args.output) / "folds").glob(pattern))]
    if not folds:
        raise SystemExit("no fold files")
    lines = ["# correction_v1 results ({} folds: {})".format(len(folds), [f["held_out_seed"] for f in folds]), ""]
    lines.append("Sanity: " + json.dumps([f["sanity"] for f in folds]))
    lines.append("Temperature per fold: " + str([round(f["temperature"], 2) for f in folds]))
    lines.append("")

    # ---- oracle restoration table
    per = defaultdict(lambda: defaultdict(list))         # (cause, sev, method) -> fold -> rows
    for f in folds:
        for r in f["oracle"]:
            per[(r["cause"], r["severity"], r["method"])][f["held_out_seed"]].append(r)
    lines += ["## Stage 1: oracle-cause restoration", "",
              "| Cause | Severity | Map | Stage NMSE | Cosine | BLEU-1 | BLEU-4 | Token acc. | Gap recovered |",
              "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for key in sorted(per, key=lambda k: (k[0], str(k[1]), k[2] != "none", k[2])):
        cause, sev, method = key
        fold_vals = defaultdict(list)
        for seed, rows in per[key].items():
            none_rows = per[(cause, sev, "none")][seed]
            b_none = np.mean([r["bleu1"] for r in none_rows])
            b_nom = np.mean([r["nominal_bleu1"] for r in none_rows])
            b = np.mean([r["bleu1"] for r in rows])
            for m in ("nmse", "cos", "bleu1", "bleu4", "token_acc"):
                fold_vals[m].append(np.mean([r[m] for r in rows]))
            fold_vals["gap"].append((b - b_none) / (b_nom - b_none) if abs(b_nom - b_none) > 1e-9 else float("nan"))
            fold_vals["nominal"].append(b_nom)
        s = {m: ms(v) for m, v in fold_vals.items()}
        summary["|".join(map(str, key))] = s
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            cause, sev, method, fmt(s["nmse"]), fmt(s["cos"]), fmt(s["bleu1"]), fmt(s["bleu4"]),
            fmt(s["token_acc"]), "–" if method == "none" else fmt(s["gap"])))
    lines.append("")
    lines.append("Nominal reference BLEU-1 (C->C for physical causes, D->D for mismatch) is in summary.json.")
    lines.append("")

    # ---- variance calibration
    vc = defaultdict(list)
    for f in folds:
        for r in f.get("variance_calibration", []):
            vc[(r["cause"], r["severity"])].append(r["spearman_var_vs_sqerr"])
    if vc:
        lines += ["## Predicted variance vs actual error (Spearman, per batch)", "", "| Cause | Severity | rho |", "|---|---|---|"]
        for k, v in sorted(vc.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            lines.append("| {} | {} | {} |".format(k[0], k[1], fmt(ms(v))))
        lines.append("")

    # ---- nominal harm
    nh = defaultdict(lambda: defaultdict(list))
    for f in folds:
        for r in f["nominal_harm"]:
            nh[(r["map"], r["method"])][f["held_out_seed"]].append(r["delta_bleu1"])
    lines += ["## Nominal harm (ΔBLEU-1 when the map is applied to nominal blocks)", "",
              "| Map | Method | ΔBLEU-1 |", "|---|---|---|"]
    for k in sorted(nh):
        lines.append("| {} | {} | {} |".format(k[0], k[1], fmt(ms([np.mean(v) for v in nh[k].values()]))))
    lines.append("")

    # ---- wrong-map harm
    wm = defaultdict(lambda: defaultdict(list))
    for f in folds:
        for r in f["wrong_map"]:
            wm[(r["true_cause"], r["applied_map"])][f["held_out_seed"]].append(r["delta_bleu1"])
    causes = ["low_snr", "imperfect_csi", "spatial_correlation", "mismatch"]
    lines += ["## Wrong-map harm (ΔBLEU-1 vs uncorrected; rows = true cause, columns = applied map)", "",
              "| True \\ applied | " + " | ".join(causes) + " |", "|---" * (len(causes) + 1) + "|"]
    for t in causes:
        cells = []
        for a in causes:
            cells.append("–" if a == t else fmt(ms([np.mean(v) for v in wm[(t, a)].values()])))
        lines.append("| {} | {} |".format(t, " | ".join(cells)))
    lines.append("")

    # ---- routing
    variants = ["none", "oracle", "hard_sentence", "hard_block", "soft_sentence", "soft_block"]
    rt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in folds:
        for r in f["routing"]:
            for v in variants:
                rt[r["profile"]][v][f["held_out_seed"]].append(r[v])
    lines += ["## Stage 2: routing with the diagnostic (BLEU-1 per profile)", "",
              "| Profile | " + " | ".join(variants) + " |", "|---" * (len(variants) + 1) + "|"]
    route_summary = {}
    for p in rt:
        cells = []
        for v in variants:
            s = ms([np.mean(x) for x in rt[p][v].values()])
            route_summary["{}|{}".format(p, v)] = s
            cells.append(fmt(s))
        lines.append("| {} | {} |".format(p, " | ".join(cells)))
    lines.append("")
    # Receiver C (all profiles except mismatch) and receiver D (mismatch) are separate regimes:
    # the benchmark has no D-receiver nominal/physical/source profiles, so they are not pooled.
    correctable = ["low_snr", "imperfect_csi", "spatial_correlation"]
    others = ["normal", "literal", "ood_hansard"]
    lines += ["Receiver C regime only. Mismatch (receiver D) is reported in its own row of the table above.", "",
              "| Routing | Mean ΔBLEU-1, physical profiles (receiver C) | Mean ΔBLEU-1, normal/literal/OOD (harm, receiver C) |", "|---|---|---|"]
    for v in variants[1:]:
        gain = ms([np.mean([np.mean(rt[p][v][s]) - np.mean(rt[p]["none"][s]) for p in correctable if p in rt])
                   for s in rt["normal"]["none"]])
        harm = ms([np.mean([np.mean(rt[p][v][s]) - np.mean(rt[p]["none"][s]) for p in others if p in rt])
                   for s in rt["normal"]["none"]])
        lines.append("| {} | {} | {} |".format(v, fmt(gain), fmt(harm)))
    out = Path(args.output)
    (out / ("summary_smoke.md" if args.smoke else "summary.md")).write_text("\n".join(lines) + "\n")
    (out / ("summary_smoke.json" if args.smoke else "summary.json")).write_text(
        json.dumps({"oracle": summary, "routing": route_summary}, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
