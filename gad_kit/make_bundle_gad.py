"""Pack everything the RTX 3060 machine needs into gad_kit/gad_bundle.zip (run on the laptop after check_gad.py).

    .venv\\Scripts\\python gad_kit\\make_bundle_gad.py

Unzip anywhere on the 3060 machine:
    gad_bundle/gad_kit/   training kit + dataset/ (jsonl + images) + keep_ids.json
    gad_bundle/gad_tools/ the GAD reader (annotate_gad.py, gad_kinds.py, facts.py) used by ask_gad.py
    gad_bundle/test_gad/  the five held-out GAD PDFs for your own test (never trained on)
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "gad_kit"
DS = ROOT / "data" / "gad" / "dataset"
OUT = KIT / "gad_bundle.zip"


def main():
    if not (KIT / "keep_ids.json").exists():
        raise SystemExit("run check_gad.py first (it writes keep_ids.json)")
    used = set()
    for f in DS.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("image"):
                used.add(r["image"])
    missing = [p for p in used if not (DS / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing (e.g. {missing[:3]}): rebuild the dataset")
    tests = [l.split("\t") for l in (DS / "test_gad.txt").read_text(encoding="utf-8").splitlines() if "\t" in l]
    with zipfile.ZipFile(OUT, "w") as z:
        for p in KIT.iterdir():
            if p.suffix in (".py", ".sh", ".txt", ".json") and p.name != OUT.name:
                z.write(p, f"gad_bundle/gad_kit/{p.name}", zipfile.ZIP_DEFLATED)
        for name in ("annotate_gad.py", "gad_kinds.py", "facts.py", "image_page.py", "gad_vocab.txt", "glossary.py"):
            z.write(ROOT / "gad_tools" / name, f"gad_bundle/gad_tools/{name}", zipfile.ZIP_DEFLATED)
        for f in DS.glob("*.jsonl"):
            z.write(f, f"gad_bundle/gad_kit/dataset/{f.name}", zipfile.ZIP_DEFLATED)
        z.write(DS / "test_gad.txt", "gad_bundle/gad_kit/dataset/test_gad.txt")
        for p in sorted(used):
            z.write(DS / p, f"gad_bundle/gad_kit/dataset/{p}", zipfile.ZIP_STORED)
        for _, test_pdf in tests:
            z.write(ROOT / "GAD" / test_pdf, f"gad_bundle/test_gad/{test_pdf}", zipfile.ZIP_DEFLATED)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB): {len(used)} images; test GADs {', '.join(g for g, _ in tests)}")


if __name__ == "__main__":
    main()
