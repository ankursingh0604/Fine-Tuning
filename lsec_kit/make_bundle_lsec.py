"""Pack what the RunPod training needs into lsec_kit/lsec_bundle.zip (run on the laptop after check_lsec.py).

    .venv\\Scripts\\python lsec_kit\\make_bundle_lsec.py

    lsec_bundle/lsec_kit/   training kit + dataset/ (jsonl + the images the rows use) + keep_ids.json
    lsec_bundle/runpod/     the Railsight readers the question tool answers with (find_crop, sheet_objects, band_table,
                            bridge_list, bridge_table, bridge_card, pdf_book)
    lsec_bundle/scripts/    lsec_facts.py (the facts given with a question), annotate*.py (reading a PDF)
    lsec_bundle/gad_tools/  glossary.py (meanings of terms: the sheet's list, standard, web)
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "lsec_kit"
DS = ROOT / "data" / "v5" / "dataset"
OUT = KIT / "lsec_bundle.zip"


def main():
    if not (KIT / "keep_ids.json").exists():
        raise SystemExit("run check_lsec.py first (it writes keep_ids.json)")
    used = set()
    for split in ("train", "val"):
        for line in open(DS / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("image"):
                used.add(r["image"])
    with zipfile.ZipFile(OUT, "w") as z:
        for p in sorted(KIT.iterdir()):
            if p.is_file() and p.suffix in (".py", ".sh", ".txt", ".json") and p.name != OUT.name:
                z.write(p, f"lsec_bundle/lsec_kit/{p.name}", zipfile.ZIP_DEFLATED)
        for name in ("find_crop.py", "sheet_objects.py", "band_table.py", "bridge_list.py", "bridge_table.py", "bridge_card.py", "pdf_book.py", "layout.py"):
            if (ROOT / "runpod" / name).exists():
                z.write(ROOT / "runpod" / name, f"lsec_bundle/runpod/{name}", zipfile.ZIP_DEFLATED)
        for name in ("lsec_facts.py", "annotate.py", "annotate_v4.py", "annotate_v5.py"):       # (ask_lsec reads a PDF with these)
            z.write(ROOT / "scripts" / name, f"lsec_bundle/scripts/{name}", zipfile.ZIP_DEFLATED)
        z.write(ROOT / "runpod" / "model_v5.py", "lsec_bundle/runpod/model_v5.py", zipfile.ZIP_DEFLATED)
        z.write(ROOT / "gad_tools" / "glossary.py", "lsec_bundle/gad_tools/glossary.py", zipfile.ZIP_DEFLATED)
        for f in ("train.jsonl", "val.jsonl", "stats.json"):
            z.write(DS / f, f"lsec_bundle/lsec_kit/dataset/{f}", zipfile.ZIP_DEFLATED)
        for p in sorted(used):
            z.write(DS / p, f"lsec_bundle/lsec_kit/dataset/{p}", zipfile.ZIP_STORED)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB): {len(used)} images")


if __name__ == "__main__":
    main()
