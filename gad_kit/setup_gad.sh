#!/usr/bin/env bash
# One-time setup on the RTX 3060 machine (Linux, NVIDIA driver 570 or newer for CUDA 12.8). About 15-30 minutes,
# mostly the ~9 GB model download.
#   cd gad_bundle/gad_kit && bash setup_gad.sh
# Same library versions as the v4 kit (Qwen3.5 needs transformers 5.2).
set -euo pipefail
cd "$(dirname "$0")"
VENV="${GAD_VENV:-$HOME/venv_gad}"
if [ ! -d "$VENV" ]; then
  python3 -m pip install -q --user uv || true
  python3 -m uv venv "$VENV" --python 3.11 || python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install -q uv
uv pip install -q "torch==2.8.0" torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "triton>=3.3.0" numpy pillow pymupdf bitsandbytes "xformers==0.0.32.post2" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install -q --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" "trl==0.22.2"
uv pip install -q "transformers==5.2.0" huggingface_hub

python - <<'EOF'
import torch
print("torch", torch.__version__, "CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("GPU memory", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB; bf16:", torch.cuda.is_bf16_supported())
EOF

MODEL="${GAD_MODEL:-unsloth/Qwen3.5-4B}"
echo "downloading $MODEL (about 9 GB) ..."
python - <<EOF
from huggingface_hub import snapshot_download
print("model at", snapshot_download("$MODEL"))
EOF
echo "setup done. Next: bash run_gad.sh"
