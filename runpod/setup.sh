#!/usr/bin/env bash
# One-time setup on a RunPod pod: Unsloth (and its transformers / peft) in /workspace/venv, kept apart
# from the pod's own Python so the pod's Jupyter is never modified. Safe to run again: it skips what exists.
set -euo pipefail
cd /workspace

[ -f dataset.zip ] && [ ! -f dataset/train.jsonl ] && python3 -m zipfile -e dataset.zip .

if [ ! -x venv/bin/python ]; then
    echo "one-time setup: installing Unsloth into /workspace/venv (a few minutes)..."
    python3 -m pip install -q uv
    python3 -m uv venv venv --python "$(command -v python3)"
    # A PyTorch build newer than the pod's GPU driver cannot use the GPU, so match it to the driver.
    DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || echo 0)
    if [ "$DRIVER_CUDA" -lt 13 ]; then
        python3 -m uv pip install --python venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
    fi
    python3 -m uv pip install --python venv/bin/python unsloth pandas matplotlib
fi

venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'this PyTorch build cannot use the GPU driver'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
