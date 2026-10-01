#!/usr/bin/env bash
# Start the question-answering app for a trained adapter on a RunPod pod.
#
#   bash /workspace/serve.sh            uses /workspace/adapter.zip, port 7860
#   bash /workspace/serve.sh --share    also prints a temporary public gradio.live link
#
# Open it from the pod's Connect tab (HTTP port 7860 must be exposed on the pod), or use --share.
# Ctrl+C stops the app.
set -euo pipefail
cd /workspace

bash /workspace/setup.sh
venv/bin/python -c "import gradio" 2>/dev/null || python3 -m uv pip install --python venv/bin/python -q gradio

export HF_HOME=/workspace/hf                 # the model downloaded during training is reused, not fetched again
ADAPTER=/workspace/adapter.zip
[ -f /workspace/railway_vlm/adapter.zip ] && ADAPTER=/workspace/railway_vlm/adapter.zip   # just trained on this pod
[ -f "$ADAPTER" ] || { echo "no adapter: train first, or upload adapter.zip to /workspace"; exit 1; }
echo "using adapter $ADAPTER"
venv/bin/python -u /workspace/app.py --adapter "$ADAPTER" --port 7860 "$@"
