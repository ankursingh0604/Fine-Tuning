#!/usr/bin/env bash
# Start L-section v5 training on RunPod in the background (keeps running if the browser / terminal closes).
#   bash run_runpod.sh                     # batch 8 x 2 (effective 16), GPU-side checkpointing, 8 loader workers, 2 epochs, stops at 4 h (best checkpoint kept), then scores the validation rows
#   bash run_runpod.sh --max-hours 2       # a tighter time limit
#   bash run_runpod.sh --resume            # continue from the last checkpoint after a restart
# Anything after the script name goes to train_lsec.py (later options win over these defaults).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf
source /workspace/venv_lsec/bin/activate
mkdir -p /workspace/lsec_run
nohup python train_lsec.py --out /workspace/lsec_run --batch 8 --grad-accum 2 --grad-ckpt gpu --workers 8 --max-hours 4 "$@" \
    > /workspace/lsec_run/train.log 2>&1 &
echo "training started (pid $!)."
echo "follow it:   tail -f /workspace/lsec_run/progress.log"
echo "full log:    tail -f /workspace/lsec_run/train.log"
echo "GPU:         watch -n 10 nvidia-smi"
