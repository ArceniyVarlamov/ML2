#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Python not found at: $PY"
  echo "Set PYTHON_BIN=/path/to/python or create .venv first."
  exit 1
fi

echo "[a100_preflight] root=$ROOT"
echo "[a100_preflight] python=$PY"

need_files=(
  "Data/Main/train_target.parquet"
  "Data/Main/sample_submit.parquet"
  "Data/Train/train_main_features.parquet"
  "Data/Train/train_extra_features.parquet"
  "Data/Test/test_main_features.parquet"
  "Data/Test/test_extra_features.parquet"
  "configs/top1_a100_h9_full_xgb_v1.json"
  "configs/top1_a100_h6_targetaware_strong_gpu_v1.json"
  "configs/manifests/champion_current_best_sources.json"
)

for f in "${need_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[a100_preflight] MISSING: $f"
    exit 1
  fi
done

echo "[a100_preflight] files: OK"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[a100_preflight] nvidia-smi:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[a100_preflight] nvidia-smi not found (OK for CPU-only prep)"
fi

"$PY" - <<'PY'
import importlib, sys
mods = ["numpy", "pandas", "polars", "pyarrow", "sklearn", "lightgbm", "xgboost", "catboost"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    print("[a100_preflight] missing python modules:", ", ".join(missing))
    sys.exit(1)
print("[a100_preflight] python modules: OK")
PY

echo "[a100_preflight] write-meta dry run: H9"
"$PY" top1_pipeline.py --config configs/top1_a100_h9_full_xgb_v1.json write-meta

echo "[a100_preflight] write-meta dry run: H6 strong"
"$PY" top1_pipeline.py --config configs/top1_a100_h6_targetaware_strong_gpu_v1.json write-meta

echo "[a100_preflight] DONE"
