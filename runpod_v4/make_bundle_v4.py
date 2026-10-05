"""Pack everything the pod needs into runpod_v4/bundle_v4.zip (run on the laptop after check_v4.py).

    .venv\\Scripts\\python runpod_v4\\make_bundle_v4.py

Contents (unzip on the pod as /workspace/v4):
    v4/*.py, *.sh, keep_ids.json, check_report.txt, RUNPOD_V4_STEPS.txt, tools_v4.json, qwen35_format.py
    v4/dataset/*.jsonl and v4/dataset/images/ (only images some row uses)
    v4/app/  the assistant and the benchmark in the project's own layout (run after training by train_v4.py):
             assistant_v4/, the scripts it imports, runpod_v4/data_v4.py, docs/tools_v4.json (+ docs/library),
             data/v4/annotations/, data/v4/benchmark/benchmark_v4.jsonl and the source PDFs
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "runpod_v4"
DS = ROOT / "data" / "v4" / "dataset"
OUT = KIT / "bundle_v4.zip"
APP_CODE = ["assistant_v4/agent.py", "assistant_v4/store.py", "assistant_v4/tools.py", "assistant_v4/run_benchmark.py",
            "scripts/annotate.py", "scripts/annotate_v4.py", "scripts/agent_conversations_v4.py", "scripts/build_dataset.py",
            "scripts/build_dataset_v4.py", "scripts/build_reasoning.py", "scripts/qwen35_format.py",
            "runpod_v4/data_v4.py", "docs/tools_v4.json", "data/v4/benchmark/benchmark_v4.jsonl"]


def app_files():
    """(source, path inside app/) for everything the assistant and the benchmark need."""
    files = [(ROOT / p, p) for p in APP_CODE]
    anns = sorted((ROOT / "data" / "v4" / "annotations").glob("*.json"))
    files += [(f, f"data/v4/annotations/{f.name}") for f in anns]
    pdfs = sorted({json.loads(f.read_text(encoding="utf-8"))["source_pdf"] for f in anns})
    files += [(ROOT / p, p) for p in pdfs]
    lib = ROOT / "docs" / "library"
    files += [(f, f.relative_to(ROOT).as_posix()) for f in lib.rglob("*") if f.is_file()] if lib.exists() else []
    missing = [p for f, p in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing for app/: {missing}")
    return files


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
    app = app_files()
    with zipfile.ZipFile(OUT, "w") as z:
        for f, p in app:
            z.write(f, f"v4/app/{p}", zipfile.ZIP_DEFLATED)
        for p in KIT.iterdir():
            if p.suffix in (".py", ".sh", ".txt", ".json") and p.name != OUT.name:
                z.write(p, f"v4/{p.name}", zipfile.ZIP_DEFLATED)
        z.write(ROOT / "docs" / "tools_v4.json", "v4/tools_v4.json", zipfile.ZIP_DEFLATED)
        z.write(ROOT / "scripts" / "qwen35_format.py", "v4/qwen35_format.py", zipfile.ZIP_DEFLATED)
        for f in DS.glob("*.jsonl"):
            z.write(f, f"v4/dataset/{f.name}", zipfile.ZIP_DEFLATED)
        for n in sorted(used):
            z.write(DS / "images" / n, f"v4/dataset/images/{n}", zipfile.ZIP_STORED)    # PNGs are already compressed
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB): {len(used)} images, app/ {len(app)} files")


if __name__ == "__main__":
    main()
