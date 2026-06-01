#!/usr/bin/env bash
# Full replication pipeline for Paperspace / headless GPU machines.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "==> Project root: $ROOT"
echo "==> Python: $(python --version)"

echo "==> Install dependencies (Paperspace: no torch in requirements-paperspace.txt)"
pip install -q -U pip wheel
pip install -q -r requirements-paperspace.txt

echo "==> Fix / install PyTorch (GPU)"
chmod +x scripts/fix_paperspace_torch.sh
./scripts/fix_paperspace_torch.sh

DATA_SOURCE="${DATA_SOURCE:-bnci}"
if [ "$DATA_SOURCE" = "synthetic" ]; then
  echo "==> Generate synthetic data (DATA_SOURCE=synthetic)"
  python scripts/generate_synthetic_data.py
  echo "==> Preprocess synthetic"
  python scripts/run_preprocessing.py --input data/sample --output data/processed
elif [ "$DATA_SOURCE" = "bnci" ]; then
  echo "==> Import BNCI2014_001 (MOABB; first run downloads data)"
  python scripts/import_bnci2014_001.py --output data/processed
else
  echo "Unknown DATA_SOURCE=$DATA_SOURCE (use bnci or synthetic)" >&2
  exit 1
fi

QUICK="${QUICK_EPOCHS:-10}"
echo "==> Train proposed model (all subjects, epochs=$QUICK)"
python scripts/run_training_all_subjects.py --model proposed --quick-epochs "$QUICK"

echo "==> Baselines"
python scripts/run_baselines.py --all-subjects --quick-epochs "$QUICK"

echo "==> Report"
python scripts/generate_report.py

echo "==> Tests"
pytest tests/ -q

echo "==> Done. See results/reports/"
