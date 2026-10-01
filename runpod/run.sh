#!/usr/bin/env bash
# Train on a RunPod pod without touching the pod's own Python, which Jupyter runs on.
# (Installing Unsloth into that Python upgrades packages under a running Jupyter and breaks it.)
#
#   bash /workspace/run.sh          start, or resume, training in the background
#   tail -f /workspace/train.log    follow it; Ctrl+C stops following, not training
set -euo pipefail
cd /workspace

if pgrep -f "venv/bin/python -u train.py" >/dev/null; then
    echo "training is already running - follow it with: tail -f /workspace/train.log"
    exit 0
fi

bash /workspace/setup.sh

nohup venv/bin/python -u train.py > train.log 2>&1 &
echo "training started in the background (pid $!)"
echo "follow it with:  tail -f /workspace/train.log"
