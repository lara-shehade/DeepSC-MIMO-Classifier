#!/bin/bash
# Apply the narrowly scoped RNG-resume fix to an existing immutable run copy.
set -euo pipefail

if [[ $# -ne 1 || "$1" != /* ]]; then
  echo "Usage: bash $0 /absolute/path/to/existing/run" >&2
  exit 2
fi
RUN_ROOT="$(cd "$1" && pwd)"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$RUN_ROOT/code/main.py"
OLD_SHA=0503af035930aa55029da626b868932ad6a817336ef703ced0653adb230fe31b
NEW_SHA=c64be3586cf51a25c0e04587d282b29edc4fb60e3065eff3d5de39706fea8617

[[ -f "$RUN_ROOT/PIPELINE_SUBMITTED" && -f "$RUN_ROOT/source_snapshot.sha256" && -f "$TARGET" ]] || {
  echo "Not a submitted DeepSC-MIMO run: $RUN_ROOT" >&2
  exit 2
}
[[ "$(sha256sum "$SOURCE_ROOT/main.py" | awk '{print $1}')" == "$NEW_SHA" ]] || {
  echo "The uploaded hotfix main.py is not the expected version" >&2
  exit 2
}
current_sha="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$current_sha" == "$NEW_SHA" ]]; then
  echo "RNG-resume fix already installed; no files changed"
  exit 0
fi
[[ "$current_sha" == "$OLD_SHA" ]] || {
  echo "Run main.py differs from the expected original; no files changed" >&2
  exit 2
}

AMENDMENT="$RUN_ROOT/amendments/rng_resume_20260928"
mkdir -p "$AMENDMENT"
[[ ! -e "$AMENDMENT/main.py.before" ]] || {
  echo "Existing amendment backup needs inspection; no files changed" >&2
  exit 2
}
cp -p "$TARGET" "$AMENDMENT/main.py.before"
temporary="$(mktemp "$RUN_ROOT/code/main.py.fix.XXXXXX")"
trap 'rm -f -- "$temporary"' EXIT
cp "$SOURCE_ROOT/main.py" "$temporary"
chmod --reference="$TARGET" "$temporary"
mv -f -- "$temporary" "$TARGET"
[[ "$(sha256sum "$TARGET" | awk '{print $1}')" == "$NEW_SHA" ]] || {
  echo "Patched main.py failed checksum verification" >&2
  exit 1
}
printf 'utc_time\told_sha256\tnew_sha256\treason\n%s\t%s\t%s\t%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OLD_SHA" "$NEW_SHA" \
  'CPU RNG ByteTensor restoration for resumed GPU training' \
  > "$AMENDMENT/record.tsv"
echo "Installed and recorded RNG-resume fix; original source snapshot was retained"
