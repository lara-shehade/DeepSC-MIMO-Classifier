"""Check the repository files against the SHA-256 lists saved by the cluster jobs.

Usage (from the repository root):  python provenance/check_hashes.py
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHES = Path(__file__).resolve().parent / "hashes"

# where each recorded path lives in this repository
PREFIXES = [
    ("<RUN_ROOT>/paper_extraction_v1/code/", "experiments/paper_extraction/"),
    ("<RUN_ROOT>/code/", ""),
    ("code/", ""),
]
SPECIAL = {"correction_code.sha256": "experiments/correction/"}
# documentation edited for publication (cluster paths replaced)
EDITED_DOCS = {"stage_subset/README.md", "classical_tail/README.md"}
NOT_IN_REPO = {"<RUN_ROOT>/source_snapshot.tar.gz"}


def locate(manifest, recorded):
    if manifest.name in SPECIAL:
        return SPECIAL[manifest.name] + recorded
    for prefix, target in PREFIXES:
        if recorded.startswith(prefix):
            return target + recorded[len(prefix):]
    return recorded


def main():
    counts = {"match": 0, "edited doc": 0, "not in repo": 0, "MISMATCH": 0}
    for manifest in sorted(HASHES.glob("*.sha256")):
        for line in manifest.read_text().splitlines():
            digest, recorded = line.split(None, 1)
            recorded = recorded.strip()
            if recorded in NOT_IN_REPO:
                counts["not in repo"] += 1
                continue
            path = locate(manifest, recorded)
            actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            if actual == digest:
                counts["match"] += 1
            elif path in EDITED_DOCS:
                counts["edited doc"] += 1
            else:
                counts["MISMATCH"] += 1
                print("MISMATCH {} ({})".format(path, manifest.name))
    print(", ".join("{}: {}".format(k, v) for k, v in counts.items()))
    return 1 if counts["MISMATCH"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
