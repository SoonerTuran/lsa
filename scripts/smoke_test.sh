#!/usr/bin/env bash
# Smoke test: unit tests, then a <=2 min CPU training run of configs/phase1_smoke.yaml
# that must write a .ply and a slice .png into the out dir.
#   bash scripts/smoke_test.sh [out_dir]        (PYTHON=... to pick an interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
else PY=python; fi

OUT="${1:-runs/smoke_$(date +%Y%m%d_%H%M%S)}"
LIMIT=120

echo "== pytest"
"$PY" -m pytest -q

echo "== smoke training (CPU) -> $OUT"
start=$(date +%s)
"$PY" -m lsa.train --config configs/phase1_smoke.yaml --out "$OUT" --set device=cpu
elapsed=$(( $(date +%s) - start ))

ply=$(find "$OUT" -name '*.ply' | head -n 1)
png=$(find "$OUT" -name '*.png' | head -n 1)
if [[ -z "$ply" || -z "$png" ]]; then
  echo "FAIL: no .ply / .png written under $OUT"; exit 1
fi
echo "artifacts: $ply"
echo "           $png"
if (( elapsed > LIMIT )); then
  echo "FAIL: smoke run took ${elapsed}s (> ${LIMIT}s)"; exit 1
fi
echo "SMOKE OK in ${elapsed}s  (metrics: $OUT/metrics.json)"
