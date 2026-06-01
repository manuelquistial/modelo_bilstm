#!/usr/bin/env bash
# Fix PyTorch / CUDA wheel mismatch on Paperspace (libnvJitLink / libcusparse).
#
# Usage (venv activated, project root):
#   ./scripts/fix_paperspace_torch.sh
#
# CPU-only fallback (no GPU, but import/train works):
#   TORCH_CPU=1 ./scripts/fix_paperspace_torch.sh
#
# Recommended Paperspace order:
#   pip install -r requirements-paperspace.txt
#   ./scripts/fix_paperspace_torch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

USE_CPU="${TORCH_CPU:-0}"

# Stale LD_LIBRARY_PATH from the host image often shadows pip's NVIDIA libs.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
unset PYTHONPATH

pip install -U pip wheel setuptools

uninstall_torch_stack() {
  echo "==> Removing broken torch / NVIDIA pip wheels..."
  pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  # Remove all nvidia-* wheels (version skew causes libnvJitLink symbol errors)
  while IFS= read -r pkg; do
    [ -n "$pkg" ] || continue
    pip uninstall -y "$pkg" 2>/dev/null || true
  done < <(pip freeze 2>/dev/null | grep -E '^nvidia-' | sed 's/==.*//' || true)
}

verify_torch() {
  # Clean env for the check
  env -u LD_LIBRARY_PATH \
    python -c "
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
"
}

install_cpu() {
  echo "==> Installing CPU-only PyTorch..."
  uninstall_torch_stack
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  verify_torch
}

install_cuda_index() {
  local idx="$1"
  echo "==> Installing PyTorch (${idx})..."
  uninstall_torch_stack
  pip install torch torchvision --index-url "https://download.pytorch.org/whl/${idx}"

  echo "==> Aligning NVIDIA runtime wheels (nvjitlink must match cusparse)..."
  pip install -U \
    nvidia-nvjitlink-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-nccl-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-curand-cu12 \
    nvidia-nvtx-cu12 \
    2>/dev/null || true

  verify_torch
}

if [ "$USE_CPU" = "1" ]; then
  install_cpu
  echo "==> OK (CPU). Training will not use GPU."
  exit 0
fi

echo "==> Fixing GPU PyTorch (trying cu124 → cu121 → cu118)..."

for idx in cu124 cu121 cu118; do
  if install_cuda_index "$idx"; then
    echo "==> OK with PyTorch index ${idx}"
    exit 0
  fi
  echo "==> ${idx} failed, trying next..."
done

echo ""
echo "ERROR: Could not install a working GPU build of PyTorch."
echo "Options:"
echo "  1) Use the Paperspace **PyTorch** template and recreate the venv:"
echo "     deactivate && rm -rf .venv"
echo "     python3 -m venv .venv --system-site-packages"
echo "     source .venv/bin/activate"
echo "     pip install -r requirements-paperspace.txt"
echo "  2) CPU-only: TORCH_CPU=1 ./scripts/fix_paperspace_torch.sh"
echo "  3) Skip venv and use system Python on the notebook image."
exit 1
