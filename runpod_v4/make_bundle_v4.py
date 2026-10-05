"""Pack everything the pod needs into runpod_v4/bundle_v4.zip (run on the laptop after check_v4.py).

    .venv\\Scripts\\python runpod_v4\\make_bundle_v4.py

Contents (unzip on the pod as /workspace/v4):
    v4/*.py, *.sh, keep_ids.json, check_report.txt, RUNPOD_V4_STEPS.txt, tools_v4.json, qwen35_format.py
    v4/dataset/*.jsonl and v4/dataset/images/ (only images some row uses)
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "runpod_v4"
DS = ROOT / "data" / "v4" / "dataset"
OUT = KIT / "bundle_v4.zip"


def main():
    if not (KIT / "keep_ids.json").exists():
        raise SystemExit("run check_v4.py first (it writes keep_ids.json)")
    used = set()
    for f in DS.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for p in r.get("images") or ([r["image"]] if r.get("image") else []):
                used.add(Path(p).name)
    missing = [n for n in used if not (DS / "images" / n).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images used by the dataset are missing (e.g. {missing[:3]}): rebuild the dataset first")
    with zipfile.ZipFile(OUT, "w") as z:
        for p in KIT.iterdir():
            if p.suffix in (".py", ".sh", ".txt", ".json") and p.name != OUT.name:
                z.write(p, f"v4/{p.name}", zipfile.ZIP_DEFLATED)
        z.write(ROOT / "docs" / "tools_v4.json", "v4/tools_v4.json", zipfile.ZIP_DEFLATED)
        z.write(ROOT / "scripts" / "qwen35_format.py", "v4/qwen35_format.py", zipfile.ZIP_DEFLATED)
        for f in DS.glob("*.jsonl"):
            z.write(f, f"v4/dataset/{f.name}", zipfile.ZIP_DEFLATED)
        for n in sorted(used):
            z.write(DS / "images" / n, f"v4/dataset/images/{n}", zipfile.ZIP_STORED)    # PNGs are already compressed
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB): {len(used)} images")


if __name__ == "__main__":
    main()
