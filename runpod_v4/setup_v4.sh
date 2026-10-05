#!/usr/bin/env bash
# One-time setup on the RunPod pod (network volume mounted at /workspace). About 15-25 minutes, mostly the model download.
#   cd /workspace/v4 && bash setup_v4.sh
# Versions follow Unsloth's Qwen3.5 vision notebook (torch 2.8 / CUDA 12.8, transformers 5.2, trl 0.22.2).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME" /workspace/v4_run

if [ ! -d /workspace/venv4 ]; then
  python -m pip install -q uv
  uv venv /workspace/venv4 --python 3.11
fi
source /workspace/venv4/bin/activate

uv pip install -q "torch==2.8.0" torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "triton>=3.3.0" numpy pillow bitsandbytes "xformers==0.0.32.post2" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install -q --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" "trl==0.22.2"
uv pip install -q "transformers==5.2.0" huggingface_hub
uv pip install -q pymupdf                     # the assistant's read_sheet / look tools (benchmark after training)

python - <<'EOF'
import torch
print("torch", torch.__version__, "CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("GPU memory", round(torch.cuda.get_device_properties(0).total_memory / 1e9), "GB")
EOF

# the assistant and the benchmark scoring work here (no GPU needed): perfect fake model 281/281, useless 0
python app/assistant_v4/run_benchmark.py --selftest --out /workspace/v4_run/benchmark_selftest | tail -n 1

MODEL="${V4_MODEL:-unsloth/Qwen3.5-27B}"
echo "downloading $MODEL to $HF_HOME (about 55 GB) ..."
python - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL")
print("model at", p)
EOF
echo "setup done. Next: bash run_v4.sh --price <your pod price per hour>"
