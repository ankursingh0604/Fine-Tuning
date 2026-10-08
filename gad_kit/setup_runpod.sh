#!/usr/bin/env bash
# One-time setup on a RunPod pod (A100 / H100 80 GB, network volume at /workspace). About 10-15 minutes,
# mostly the ~9 GB model download.
#   cd /workspace/gad_bundle/gad_kit && bash setup_runpod.sh
# Same library versions as the 3060 setup (Qwen3.5 needs transformers 5.2).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME" /workspace/gad_run

if [ ! -d /workspace/venv_gad ]; then
  python -m pip install -q uv
  uv venv /workspace/venv_gad --python 3.11
fi
source /workspace/venv_gad/bin/activate

uv pip install -q "torch==2.8.0" torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "triton>=3.3.0" numpy pillow pymupdf bitsandbytes "xformers==0.0.32.post2" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install -q --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" "trl==0.22.2"
uv pip install -q "transformers==5.2.0" huggingface_hub

python - <<'EOF'
import torch
print("torch", torch.__version__, "CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("GPU memory", round(torch.cuda.get_device_properties(0).total_memory / 1e9), "GB; bf16:", torch.cuda.is_bf16_supported())
EOF

MODEL="${GAD_MODEL:-unsloth/Qwen3.5-4B}"
echo "downloading $MODEL to $HF_HOME (about 9 GB) ..."
python - <<EOF
from huggingface_hub import snapshot_download
print("model at", snapshot_download("$MODEL"))
EOF
echo "setup done. Next: bash run_runpod.sh"
