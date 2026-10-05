#!/usr/bin/env bash
# Start GAD training in the background (keeps running if the terminal closes).
#   bash run_gad.sh                 # 2 epochs (default)
#   bash run_gad.sh --resume        # continue from the last checkpoint after a stop / restart
#   bash run_gad.sh --epochs 3
set -euo pipefail
cd "$(dirname "$0")"
source "${GAD_VENV:-$HOME/venv_gad}/bin/activate"
mkdir -p gad_run
nohup python train_gad.py "$@" > gad_run/train.log 2>&1 &
echo "training started (pid $!)."
echo "follow it:   tail -f gad_run/progress.log"
echo "full log:    tail -f gad_run/train.log"
echo "GPU:         watch -n 10 nvidia-smi"
