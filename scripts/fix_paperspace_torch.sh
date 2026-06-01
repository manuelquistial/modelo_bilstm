#!/usr/bin/env bash
# Fix PyTorch / CUDA wheel mismatch on Paperspace (libnvJitLink / libcusparse).
# Run from project root with venv activated.
set -euo pipefail

echo "==> Reinstalling PyTorch CUDA 12.1 wheels..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install -U pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Aligning NVIDIA runtime libraries..."
pip install -U "nvidia-nvjitlink-cu12>=12.4" "nvidia-cusparse-cu12>=12.3" 2>/dev/null || true

echo "==> Verify import"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
