#!/bin/bash
#SBATCH --job-name=ds_preflight
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
mkdir -p "${RUN_ROOT:?}/preflight"
python - <<'PY'
import sklearn, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("sklearn", sklearn.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY
python -m compileall -q .
python -m unittest discover -s tests -v
python -m tests.runtime_checks
python -m tests.integration_smoke
python -m tests.classical_smoke
python server/preflight.py \
  --data-root "${DEEPSC_SHARED_ROOT:?}/europarl" \
  --shared-data-root "$DEEPSC_SHARED_ROOT" \
  --vocab-file "$DEEPSC_SHARED_ROOT/shared_vocab.json" \
  --require-cuda \
  --report-file "$RUN_ROOT/preflight/server_preflight.json"

python - <<'PY'
import json, os, platform, shutil, torch
from pathlib import Path
free = shutil.disk_usage(os.environ["RUN_ROOT"]).free
minimum = int(float(os.environ.get("DEEPSC_MIN_FREE_GB", "20")) * 1024**3)
if free < minimum:
    raise SystemExit("insufficient scratch: {:.2f} GiB free, {:.2f} GiB required".format(free/1024**3, minimum/1024**3))
report = {
    "status": "ok", "python": platform.python_version(),
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "scratch_free_bytes": free, "scratch_minimum_bytes": minimum,
}
root = Path(os.environ["RUN_ROOT"]) / "preflight"
(root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "PREFLIGHT_OK").write_text("ok\n")
PY
