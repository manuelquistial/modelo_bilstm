#!/usr/bin/env bash
# Full replication pipeline for Paperspace / headless GPU machines.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "==> Project root: $ROOT"
echo "==> Python: $(python --version)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), getattr(torch.cuda, 'get_device_name', lambda *_: 'n/a')(0) if torch.cuda.is_available() else '')" || true

echo "==> Install dependencies (skip if already installed)"
pip install -q -r requirements.txt

echo "==> Generate synthetic data"
python scripts/generate_synthetic_data.py

echo "==> Preprocess"
python scripts/run_preprocessing.py --input data/sample --output data/processed

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
