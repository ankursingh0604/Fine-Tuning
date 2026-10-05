"""Build the whole v4 dataset in order, so every part comes from the same annotations.

    .venv\\Scripts\\python scripts\\build_v4_all.py

1. annotate_v4.py            data/v4/annotations/ + sheets.csv
2. build_dataset_v4.py       normal + reasoning datasets (train/val/test)
3. augment_v4.py             low-DPI copies, 200 dpi crops, low-DPI test set
4. generic_tables_v4.py      generic table reading
5. agent_conversations_v4.py tool-use conversations
6. qwen35_format.py          checks every box against its image and the Qwen3.5 conversion
7. preview_dataset_v4.py     data/v4/dataset/preview.html
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["annotate_v4.py", "build_dataset_v4.py", "augment_v4.py", "generic_tables_v4.py",
         "agent_conversations_v4.py", "qwen35_format.py", "preview_dataset_v4.py"]

if __name__ == "__main__":
    t0 = time.time()
    for step in STEPS:
        t = time.time()
        print(f"\n===== {step}", flush=True)
        r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / step)], cwd=ROOT)
        print(f"===== {step}: exit {r.returncode}, {(time.time() - t) / 60:.1f} min", flush=True)
        if r.returncode:
            sys.exit(f"stopped: {step} failed")
    print(f"\nall steps done in {(time.time() - t0) / 60:.1f} min")
