#!/usr/bin/env bash
# Start training in the background (keeps running if the browser / terminal closes).
#   bash run_v4.sh --price 1.09                 # price of your pod per hour, for the cost estimate
#   bash run_v4.sh --price 1.09 --budget 35     # stop after the smoke test if the estimate is above $35
#   bash run_v4.sh --resume                     # continue from the last checkpoint after a restart
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf
source /workspace/venv4/bin/activate
mkdir -p /workspace/v4_run
nohup python train_v4.py --out /workspace/v4_run "$@" > /workspace/v4_run/train.log 2>&1 &
echo "training started (pid $!)."
echo "follow it:   tail -f /workspace/v4_run/progress.log"
echo "full log:    tail -f /workspace/v4_run/train.log"
echo "GPU:         watch -n 10 nvidia-smi"
