"""Annotate every Plan & L-Section PDF in the project folder for the v4 dataset.

    .venv\\Scripts\\python scripts\\annotate_v4.py

Writes data/v4/annotations/<id>.json (one per sheet) and data/v4/sheets.csv (an index). Sheet numbers repeat
across the drawing sets (3rd line, 4th line), so each sheet is identified by its drawing-number series and sheet
number, e.g. MKN-3RD_100 and MKN-4TH_100. The v3 annotations in data/annotations/ are left untouched.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import annotate as A           # noqa: E402

OUT = ROOT / "data" / "v4"


def sheet_id(ann, pdf, i):
    info = ann["sheet_info"]
    m = re.search(r"/([A-Z0-9-]+)/P&P/(\d+)", info.get("drawing_no") or "")
    if m:
        return f"{m.group(1)}_{int(m.group(2)):03d}"
    return f"{re.sub(r'[^A-Za-z0-9]+', '_', Path(pdf).stem)}_p{i + 1:02d}"


def findings(ann):
    """Things on the drawing worth telling the engineers (not annotation problems)."""
    out = []
    info, band = ann["sheet_info"], ann.get("bands") if isinstance(ann.get("bands"), dict) else {}
    if info.get("chainage_source"):
        out.append("chainage range taken from the data bands: " + info["chainage_source"])
    rail = [c for c in band.get("columns", []) if c.get("rail_level_ok") is False and c["checks_ok"]]
    if rail:
        vals = sorted({c["rl_minus_fl"] for c in rail})
        out.append(f"{len(rail)} band columns with RL - FL = {', '.join(map(str, vals))} instead of 0.762 (rail-level note)")
    lines = band.get("lines") or {}
    series = re.search(r"/MKN-(\w+)/", info.get("drawing_no") or "")
    if series and lines.get("proposed") and series.group(1) not in lines["proposed"]:
        out.append(f"drawing number is in the {series.group(1)} series but the band rows are for {lines['proposed']}")
    if "error" in (ann.get("bands") or {}):
        out.append("data bands not read: " + ann["bands"]["error"])
    return out


def main():
    (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    index = []
    for pdf in sorted(ROOT.glob("*.pdf")):
        for i, ann in enumerate(A.annotate_pdf(pdf)):
            sid = sheet_id(ann, pdf, i)
            ann["sheet_id"] = sid
            ann["findings"] = findings(ann)
            (OUT / "annotations" / f"{sid}.json").write_text(json.dumps(ann, indent=1, ensure_ascii=False), encoding="utf-8")
            info, band = ann["sheet_info"], ann.get("bands") if isinstance(ann.get("bands"), dict) else {}
            cols = band.get("columns", [])
            index.append({
                "sheet_id": sid, "pdf": pdf.name, "page": i + 1, "sheet_no": info.get("sheet_no"),
                "title": info.get("title"), "chainage_from": info.get("chainage_from"), "chainage_to": info.get("chainage_to"),
                "proposed_line": (band.get("lines") or {}).get("proposed"), "existing_line": (band.get("lines") or {}).get("existing"),
                "bridges": len(ann["bridges"]), "level_blocks": sum(1 for b in ann["bridges"] if b.get("lsection_levels")),
                "curves": len(ann["curves"]), "band_columns": len(cols), "band_columns_ok": sum(c["checks_ok"] for c in cols),
                "tbms": len(ann["tbm_benchmarks"]), "bridge_table_rows": len(ann.get("bridge_table") or []),
                "notes": len(ann["notes"]), "findings": " | ".join(ann["findings"]),
            })
    with open(OUT / "sheets.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"{len(index)} sheets from {len({r['pdf'] for r in index})} PDFs -> {OUT / 'annotations'}")
    return index


if __name__ == "__main__":
    main()
