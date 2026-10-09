"""Cheap, read-only gate for the 20 already-completed TCN singleton folds."""

import json
from pathlib import Path
import sys

from analysis.neural_diagnostics import CLASSES
from stage_subset.ablation import SEEDS, STAGES


def validate(root):
    root = Path(root)
    counts = {}
    for stage in STAGES:
        for seed in SEEDS:
            path = root / "tcn_{}_seed{}.json".format(stage, seed)
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row.get("architecture"), row.get("stage"),
                    row.get("held_out_physical_seed")) != ("tcn", stage, seed):
                raise ValueError("wrong singleton fold identity: {}".format(path))
            pair = (row.get("train_samples"), row.get("test_samples"))
            if not all(isinstance(number, int) and number > 0 for number in pair):
                raise ValueError("invalid singleton sample counts: {}".format(path))
            if seed in counts and counts[seed] != pair:
                raise ValueError("singleton folds use different counts at seed {}".format(seed))
            counts[seed] = pair
            for metric in ("macro_f1", "balanced_accuracy"):
                value = row.get(metric)
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError("invalid {} in {}".format(metric, path))
            recalls = row.get("per_class_recall")
            if not isinstance(recalls, dict) or set(recalls) != set(CLASSES):
                raise ValueError("incomplete class recall in {}".format(path))
    print("PASS: 20 existing TCN singleton folds", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m stage_subset.validate_existing FOLD_DIR")
    validate(sys.argv[1])
