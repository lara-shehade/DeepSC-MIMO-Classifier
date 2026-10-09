"""Assemble immutable run evidence into a compact final summary."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    args = parser.parse_args(argv)
    root = Path(args.run_root).resolve()
    required = {
        "preflight": root / "preflight" / "report.json",
        "server_preflight": root / "preflight" / "server_preflight.json",
        "integrity": root / "integrity" / "cd_integrity.json",
        "first_divergence": root / "analysis" / "first_divergence.json",
        "classical": root / "analysis" / "classical" / "grouped_results.json",
        "neural": root / "analysis" / "frozen_neural.selection.json",
        "arxiv_holdout": root / "analysis" / "arxiv_holdout.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing final inputs: {}".format(missing))
    important_files = [
        root / "source_snapshot.tar.gz",
        root / "jobs.tsv",
        root / "checkpoints" / "C" / "last.pth",
        root / "checkpoints" / "C" / "best.pth",
        root / "checkpoints" / "D" / "last.pth",
        root / "checkpoints" / "D" / "best.pth",
        *required.values(),
    ]
    summary = {
        "status": "complete",
        "run_root": str(root),
        "reports": {name: load(path) for name, path in required.items()},
        "files": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in important_files
        },
        "holdout_policy": "arXiv evaluated only after classical and neural FROZEN markers",
    }
    destination = root / "final_summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
