#!/usr/bin/env bash
# Start GAD training on RunPod in the background (keeps running if the browser / terminal closes).
#   bash run_runpod.sh                     # batch 8, 8 loader workers, 2 epochs, stops at 2.5 h (best checkpoint kept), then scores the held-out GADs
#   bash run_runpod.sh --max-hours 2       # a tighter time limit
#   bash run_runpod.sh --resume            # continue from the last checkpoint after a restart
# Anything after the script name goes to train_gad.py (later options win over these defaults).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf
source /workspace/venv_gad/bin/activate
mkdir -p /workspace/gad_run
nohup python train_gad.py --out /workspace/gad_run --batch 8 --grad-accum 1 --workers 8 --max-hours 2.5 "$@" \
    > /workspace/gad_run/train.log 2>&1 &
echo "training started (pid $!)."
echo "follow it:   tail -f /workspace/gad_run/progress.log"
echo "full log:    tail -f /workspace/gad_run/train.log"
echo "GPU:         watch -n 10 nvidia-smi"
