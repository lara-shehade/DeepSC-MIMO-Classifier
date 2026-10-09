#!/bin/bash
set -euo pipefail
count=1000
if [[ -f "$MOCK_COUNT" ]]; then count="$(<"$MOCK_COUNT")"; fi
count=$((count + 1))
printf '%s\n' "$count" > "$MOCK_COUNT"
printf '%s\n' "$*" >> "$MOCK_LOG"
if [[ "${MOCK_FAIL_AT:-}" == "$count" ]]; then
  echo 'mock sbatch failure' >&2
  exit 42
fi
if [[ "${MOCK_BAD_AT:-}" == "$count" ]]; then
  echo 'Submitted batch job ???'
  exit 0
fi
printf '%s%s\n' "$count" "${MOCK_SUFFIX:-}"
