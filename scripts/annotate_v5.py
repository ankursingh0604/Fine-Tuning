"""Annotate every Plan & L-Section PDF in the project folder for the v5 dataset (Qwen3.5-4B).

    .venv\\Scripts\\python scripts\\annotate_v5.py

Writes data/v5/annotations/<id>.json (one per sheet) and data/v5/sheets.csv. Each sheet is read twice over:
  - the v4 reader (scripts/annotate.py): sheet info, regions, bridges and their levels, curves, data bands, TBMs,
    notes, abbreviations, legend, transition / grade points, gradients, VPIs, bearings, KM marks, stations ... and
    every text item on the sheet (all_text) - kept as it is;
  - the readers the Railsight kit answers with (runpod/: sheet_objects, bridge_list, bridge_table, band_table), which
    know the later layouts: proposals written "PROP AS ..." or after the existing bridge's chainage, the DETAILS OF
    BRIDGES table, curves printed as TP1 / J1 / J2 / TP2 blocks, multi-line band headings. Stored under "v5".
A PDF that is a byte copy of another (xyz.pdf) is read once. Officers' names and designations (the signature panel)
are left out, from all_text too: the model never learns or repeats personal names.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "runpod")]
import pymupdf                 # noqa: E402
import annotate as A           # noqa: E402
import annotate_v4 as V4       # noqa: E402
import band_table              # noqa: E402
import bridge_list             # noqa: E402
import bridge_table            # noqa: E402
import find_crop as F          # noqa: E402
import sheet_objects as SO     # noqa: E402

OUT = ROOT / "data" / "v5"


def label_of(k):
    """A level block's label as printed, without the bridge number's own UP / DN ("EXG. BR. NO. 259 DN" / "RL 4TH LINE")."""
    return re.sub(r"^(?:UP|DN)\s+(?=\S)", "", re.sub(r"\s+", " ", k).strip(), flags=re.I)


def box(parts):
    w = SO.joined(parts)
    return [round(v, 1) for v in w.box]


def v5_reading(page):
    """What the Railsight readers find on the sheet (PDF points, as the v4 boxes)."""
    words = F._pdf_words(page, 1.0)
    objs = SO.SheetObjects(words)
    out = {"bridges": [], "crossings": [], "curves": [], "bridge_table": [], "bands": []}
    for num, bs in bridge_list.groups_of(objs).items():
        text = lambda b: SO.joined(b.best()).text
        paired = all(SO.SPAN.search(text(b)) for b in bs)
        if paired:
            ex = next((b for b in bs if b.status == "existing"), None)
            pr = next((b for b in bs if b.status == "proposed"), None)
            row = bridge_list.parse_pair(text(ex) if ex else None, text(pr) if pr else None, num)
        else:
            row = bridge_list.parse_callout(text(bs[0]))
        lv = next((b.levels[0] for b in bs if b.levels), None)
        levels = [{"label": label_of(k), "value": v} for k, v in SO.BLOCK_KV.findall(SO.joined(lv).text)] if lv else []
        out["bridges"].append({
            "num": num, "callouts": [{"status": b.status, "text": text(b), "bbox": box(b.best())} for b in bs],
            "fields": {k: v for k, v in row.items() if v}, "level_block": {"text": SO.joined(lv).text, "bbox": box(lv), "values": levels} if lv else None})
    for c in objs.crossings:
        lv = c.levels[0] if c.levels else None
        out["crossings"].append({"label": c.label, "kind": c.kind, "num": c.num, "text": SO.joined(c.best()).text, "bbox": box(c.best()),
                                 "level_block": {"text": SO.joined(lv).text, "bbox": box(lv)} if lv else None})
    # a level block whose bridge's callout is printed on the neighbouring sheet ("EXG. BR. NO. 270UP" / FL ... only)
    have = {re.sub(r"(UP|DN)$", "", b["num"]) for b in out["bridges"]}
    for w in words:
        m = SO.BR_RE.search(w.text)
        if not m or not SO.BR_HEAD.match(w.text):
            continue
        num = SO.bridge_num(m)
        if re.sub(r"(UP|DN)$", "", num) in have:
            continue
        parts = objs.lines_from(w, lambda t: bool(SO.LEVEL_LINE.match(t)), lambda t: False, 12)
        if len(parts) > 1:
            have.add(re.sub(r"(UP|DN)$", "", num))
            levels = [{"label": label_of(k), "value": v} for k, v in SO.BLOCK_KV.findall(SO.joined(parts).text)]
            out["bridges"].append({"num": num, "callouts": [], "fields": {"br_no": num}, "callout_elsewhere": True,
                                   "level_block": {"text": SO.joined(parts).text, "bbox": box(parts), "values": levels}})
    # the sheet's chainage range (from its band): a "curve" far outside it is a misread label (a KM figure)
    chs = [c for b in band_table.find_bands(words) for _, c in b.cols]
    lo, hi = (min(chs) - 5000, max(chs) + 5000) if chs else (-1e12, 1e12)
    for c in objs.curves:
        if not c.points or not (lo <= c.span[0] and c.span[1] <= hi):
            continue
        deg, _ = objs.degree(c)
        out["curves"].append({"name": c.name, "num": c.block_num or c.num, "track": c.track, "status": c.status, "hand": c.hand,
                              "points": {p: round(ch, 3) for p, (ch, _) in sorted(c.points.items(), key=lambda t: t[1][0])},
                              "degree": deg, "details": [SO.joined(d).text for d in c.details[:1]],
                              "bbox": box([w for _, w in c.points.values()])})
    out["main_track"] = objs.main_track
    for br, row in bridge_table.read(words, page, 1.0).items():
        out["bridge_table"].append({"bridge": br, "cells": {k: bridge_table.text(v) for k, v in row.items()},
                                    "bbox": [round(v, 1) for v in SO.union([w.box for ws in row.values() for w in ws])]})
    for b in band_table.find_bands(words):
        rows = []
        for r in b.rows:
            if r is b.chain or len(r.words) < band_table.MIN_VALUES * 2:
                continue
            rows.append({"heading": r.heading, "y": round(r.y, 1),
                         "values": [[round(ch, 3), w.text] for ch, w in b.values(r)]})
        out["bands"].append({"chainage_heading": b.chain.heading, "from": b.cols[0][1], "to": b.cols[-1][1],
                             "columns": len(b.cols), "rows": rows})
    return out


# ======================================================================== the right panel, found by its headings
# (v4's readers look in fixed places on the 3770 pt sheets; the 4365 pt sheets of 5 / 6 / 7 / 8 / abc.pdf put the panel
# elsewhere: these find each part by its heading, on any layout, and fill what v4 left empty)
HEAD = re.compile(r"^(?:NOTES?\s*:?|LEGENDS?\s*:?|ABBREVIATIONS?\s*:?|TBM\s+DETAILS|REFERENCE\s+DRAWINGS|ISSUE\s+RECORD|DETAILS\s+OF\s+\w+.*|"
                  r"CLIENT\s*:|PROJECT\s*:|TITLE\s*:|CONSULTANT\s*:)$", re.I)
mid = lambda b: (b[1] + b[3]) / 2


def rows_of(ls, tol=3.5):
    out = []
    for l in sorted(ls, key=lambda l: (mid(l["bbox"]), l["bbox"][0])):
        if out and abs(mid(out[-1][0]["bbox"]) - mid(l["bbox"])) < tol:
            out[-1].append(l)
        else:
            out.append([l])
    return [sorted(r, key=lambda l: l["bbox"][0]) for r in out]


def below(lines, head, width=650, depth=500):
    """The horizontal lines under a heading, down to the next heading (or a wide gap) - its panel section."""
    hx, hy = head["bbox"][0], head["bbox"][3]
    # (from just above the heading's bottom: the first item can sit level with it - "NOTE:" / "1. ALL DIMENSIONS ...")
    cand = sorted((l for l in lines if abs(l["dir"][0]) > 0.9 and hy - 4 <= l["bbox"][1] <= hy + depth and hx - 25 <= l["bbox"][0] <= hx + width
                   and l is not head), key=lambda l: l["bbox"][1])
    out, last = [], hy
    for l in cand:
        if HEAD.match(l["text"].strip()) or l["bbox"][1] - last > 60:
            break
        out.append(l)
        last = l["bbox"][3]
    return out


def generic_notes(lines):
    head = next((l for l in lines if re.match(r"^NOTES?\s*:?$", l["text"].strip(), re.I) and abs(l["dir"][0]) > 0.9), None)
    if not head:
        return []
    out, cur = [], None
    for row in rows_of(below(lines, head)):
        t = re.sub(r"\s+", " ", " ".join(l["text"] for l in row)).strip()
        m = re.match(r"^(\d{1,2})\.\s*(.*)$", t)        # ("1." printed as its own item: the text follows on the next row)
        if m:
            cur = {"no": int(m.group(1)), "text": m.group(2)}
            out.append(cur)
        elif cur:
            cur["text"] = (cur["text"] + " " + t).strip()
    return [n for n in out if n["text"].strip()]


def generic_abbreviations(lines):
    head = next((l for l in lines if re.match(r"^ABBREVIATIONS?\s*:?$", l["text"].strip(), re.I)), None)
    if not head:
        return {}
    sec = below(lines, head, width=600, depth=220)
    out = {}
    for l in sec:
        m = re.match(r"^[=:\-]\s*(.+)$", l["text"].strip())
        if not m:
            continue
        left = [o for o in sec if o is not l and abs(mid(o["bbox"]) - mid(l["bbox"])) < 4 and 0 <= l["bbox"][0] - o["bbox"][2] < 40]
        if left:
            out[max(left, key=lambda o: o["bbox"][2])["text"].strip()] = m.group(1).strip()
    return out


def generic_tbms(lines):
    head = next((l for l in lines if l["text"].strip().upper() == "TBM DETAILS"), None)
    if not head:
        return []
    hy = head["bbox"][3]
    near = [l for l in lines if abs(l["dir"][0]) > 0.9 and hy - 2 <= l["bbox"][1] <= hy + 30 and abs(l["bbox"][0] - head["bbox"][0]) < 450]
    hdr = {l["text"].strip().upper(): l for l in near if l["text"].strip().upper() in ("TBM ID", "CHAINAGE", "EASTING", "NORTHING", "MSL", "DESCRIPTION")}
    if not {"EASTING", "NORTHING", "MSL"} <= set(hdr):
        return []
    x0 = min(l["bbox"][0] for l in hdr.values()) - 20
    x1 = (hdr["DESCRIPTION"]["bbox"][2] + 160) if "DESCRIPTION" in hdr else max(l["bbox"][2] for l in hdr.values()) + 300
    hb = max(l["bbox"][3] for l in hdr.values())
    cells = [l for l in lines if abs(l["dir"][0]) > 0.9 and hb < l["bbox"][1] < hb + 400 and x0 <= l["bbox"][0] <= x1]
    cx = lambda l: (l["bbox"][0] + l["bbox"][2]) / 2
    idx = cx(hdr["TBM ID"]) if "TBM ID" in hdr else x0 + 30
    ids = sorted((l for l in cells if abs(cx(l) - idx) < 40 and re.search(r"\d", l["text"]) and re.search(r"[A-Za-z]", l["text"])),
                 key=lambda l: l["bbox"][1])
    rows = []
    for i, idl in enumerate(ids):
        yc = mid(idl["bbox"])
        nxt = mid(ids[i + 1]["bbox"]) if i + 1 < len(ids) else yc + 30
        if rows and yc - mid(rows[-1]["bbox"]) > 60:            # (the table ended)
            break
        row = {"tbm_id": idl["text"].strip()}
        for key, name in (("CHAINAGE", "chainage_m"), ("EASTING", "easting"), ("NORTHING", "northing"), ("MSL", "msl_m")):
            if key in hdr:
                c = [l for l in cells if abs(mid(l["bbox"]) - yc) < 5 and abs(cx(l) - cx(hdr[key])) < 50 and A.num(l["text"]) is not None]
                row[name] = A.num(c[0]["text"]) if c else None
            else:
                row[name] = None
        # the description: text starting under its heading (left-aligned a little before it), not the abbreviation list
        # or legend printed beside the table
        desc = sorted((l for l in cells if "DESCRIPTION" in hdr and hdr["DESCRIPTION"]["bbox"][0] - 110 <= l["bbox"][0] <= hdr["DESCRIPTION"]["bbox"][0] + 60
                       and A.num(l["text"]) is None and yc - 5 <= mid(l["bbox"]) < (yc + nxt) / 2 + 2), key=lambda l: l["bbox"][1])
        row["description"] = " ".join(l["text"] for l in desc).strip()
        row["bbox"] = [round(v, 1) for v in A.union([idl["bbox"]] + [l["bbox"] for l in desc])]
        if sum(row[k] is not None for k in ("easting", "northing", "msl_m")) >= 2:
            rows.append(row)
    return rows


def generic_title(lines):
    """Labelled title-block fields (CLIENT : / PROJECT : / TITLE : / CONSULTANT :): the text after and under each label."""
    labels = [l for l in lines if re.match(r"^(CLIENT|PROJECT|TITLE|CONSULTANT)\s*:", l["text"].strip(), re.I) and abs(l["dir"][0]) > 0.9]
    out = {}
    for lab in labels:
        key = re.match(r"^(\w+)", lab["text"].strip()).group(1).lower()
        nxt = min([o["bbox"][1] for o in labels if o["bbox"][1] > lab["bbox"][1] + 2] + [lab["bbox"][1] + 70])
        same = re.sub(r"^\w+\s*:\s*", "", lab["text"].strip())
        body = [l for l in lines if l is not lab and abs(l["dir"][0]) > 0.9 and lab["bbox"][1] - 3 <= l["bbox"][1] < nxt - 2
                and lab["bbox"][0] - 5 <= l["bbox"][0] <= lab["bbox"][0] + 560 and not HEAD.match(l["text"].strip())]
        text = " ".join([same] + [" ".join(o["text"] for o in r) for r in rows_of(body)]).strip()
        if text:
            out[key] = re.sub(r"\s+", " ", text)
    return out


PERSON = re.compile(r"^\(\s*[A-Z][A-Z.]*(?:\s+[A-Z][A-Z.]*){1,3}\s*\)$")


def private_lines(lines):
    """Names on the sheet, on any layout: '(ANIL KALRA)' under a signature and the line next to it (the designation),
    and the issue record's prepared / checked / approved columns."""
    out = set()
    right = max(l["bbox"][2] for l in lines) if lines else 0
    designation = lambda t: bool(re.fullmatch(r"[A-Z][A-Z.()&\s]*(?:/[A-Z.()&\s]+){1,4}", t.strip()))      # "JE/DRG/C/KOTA"
    for l in lines:
        if not PERSON.match(l["text"].strip()):
            continue
        near = [o for o in lines if o is not l and abs(o["bbox"][0] - l["bbox"][0]) < 120 and 0 < abs(mid(o["bbox"]) - mid(l["bbox"])) < 14]
        # a name in brackets with a designation by it, or in the sheet's right-hand signature panel ("(NOT FOR STABLING)" in
        # the plan is not a name)
        if any(designation(o["text"]) for o in near) or l["bbox"][0] > 0.8 * right:
            out.add(id(l))
            out.update(id(o) for o in near if designation(o["text"]))
    # a designation on its own in the signature panel (right-hand fifth of the sheet): "DY.CE/C-II/KOTA"
    for l in lines:
        if l["bbox"][0] > 0.8 * right and designation(l["text"]) and not re.search(r"\d|P&P|DRG|FLS", l["text"]):
            out.add(id(l))
    rec = next((l for l in lines if l["text"].strip().upper() == "ISSUE RECORD"), None)
    if rec:
        cols = [l for l in lines if re.match(r"^(PREPARED|CHECKED|APPROVED)\s+BY$", l["text"].strip(), re.I)]
        for c in cols:
            for o in lines:
                if c["bbox"][3] < o["bbox"][1] < c["bbox"][3] + 60 and abs((o["bbox"][0] + o["bbox"][2]) / 2 - (c["bbox"][0] + c["bbox"][2]) / 2) < 45:
                    out.add(id(o))
    return out


def fill_panel(ann, lines):
    """What v4 left empty on this layout, from the headings; officers' names dropped."""
    if not ann.get("notes"):
        ann["notes"] = generic_notes(lines)
    if not ann.get("abbreviations"):
        ann["abbreviations"] = generic_abbreviations(lines)
    if not ann.get("tbm_benchmarks"):
        ann["tbm_benchmarks"] = generic_tbms(lines)
    info = ann["sheet_info"]
    t = generic_title(lines)
    for k in ("title", "client", "project", "consultant"):
        bad = info.get(k) and re.search(r"CHAINAGE|DETAILS OF|VERTICAL", info[k] or "")
        if t.get(k) and (not info.get(k) or bad):
            info[k] = t[k]
    hide = private_lines(lines)
    # a title-block field whose fixed box caught the signature panel on this layout ("client: <names>, PREPARED BY,
    # CHECKED BY ...") or any hidden name: dropped, never passed on
    hidden = {l["text"].strip().upper() for l in lines if id(l) in hide and l["text"].strip()}
    for k, v in list(info.items()):
        if isinstance(v, str) and (re.search(r"PREPARED\s*BY|CHECKED\s*BY|APPROVED\s*BY|SIGNATURE", v, re.I)
                                   or any(h in v.upper() for h in hidden if len(h) > 4)):
            info[k] = None
    return {(l["text"], tuple(round(v, 1) for v in l["bbox"])) for l in lines if id(l) in hide}


def private_texts(ann):
    """The officers' names and designations (and anything printed on them): not for the dataset."""
    names = {(o.get("name") or "").strip().upper() for o in ann.get("officers") or []} | \
            {(o.get("designation") or "").strip().upper() for o in ann.get("officers") or []}
    for r in (ann.get("sheet_info") or {}).get("issue_record") or []:          # prepared / checked / approved by: names
        names |= {(r.get(k) or "").strip().upper() for k in ("prepared_by", "checked_by", "approved_by")}
    boxes = [o["bbox"] for o in ann.get("officers") or [] if o.get("bbox")]
    over = lambda b, o: not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3])
    return lambda t: t["text"].strip().upper() in names - {""} or any(over(t["bbox"], o) for o in boxes)


def main():
    (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    pdfs, seen = [], {}
    for pdf in sorted(ROOT.glob("*.pdf")):
        h = hashlib.md5(pdf.read_bytes()).hexdigest()
        if h in seen:
            print(f"{pdf.name}: a copy of {seen[h]} - skipped", flush=True)
            continue
        seen[h] = pdf.name
        pdfs.append(pdf)
    from concurrent.futures import ProcessPoolExecutor
    index = []
    with ProcessPoolExecutor(max_workers=4) as ex:                 # one PDF per worker
        for rows in ex.map(one_pdf, pdfs):
            index += rows
    write_index(index)


def one_pdf(pdf):
    index = []
    if True:
        doc = pymupdf.open(pdf)
        for i, ann in enumerate(A.annotate_pdf(pdf)):
            page = doc[i]
            if page.rotation:
                page.remove_rotation()
            # (the drawing-number series repeats across sets - two PDFs both had MKN-3RD_012 - so the PDF is in the id)
            sid = f"{V4.sheet_id(ann, pdf, i)}@{re.sub(r'[^A-Za-z0-9]+', '', pdf.stem)[:14]}"
            ann["sheet_id"] = sid
            ann["findings"] = V4.findings(ann)
            try:
                ann["v5"] = v5_reading(page)
            except Exception as e:                        # (a reader failing on one sheet must not stop the rest)
                ann["v5"] = {"error": f"{type(e).__name__}: {e}"}
            names = fill_panel(ann, A.page_lines(page))
            hide = private_texts(ann)
            ann["all_text"] = [t for t in ann["all_text"] if not hide(t)
                               and (t["text"], tuple(round(v, 1) for v in t["bbox"])) not in names]
            ann.pop("officers", None)
            (OUT / "annotations" / f"{sid}.json").write_text(json.dumps(ann, indent=1, ensure_ascii=False), encoding="utf-8")
            info, band, v5 = ann["sheet_info"], ann.get("bands") if isinstance(ann.get("bands"), dict) else {}, ann["v5"]
            index.append({
                "sheet_id": sid, "pdf": pdf.name, "page": i + 1, "sheet_no": info.get("sheet_no"),
                "chainage_from": info.get("chainage_from"), "chainage_to": info.get("chainage_to"),
                "bridges_v4": len(ann["bridges"]), "bridges_v5": len(v5.get("bridges", [])),
                "with_proposal_v5": sum(1 for b in v5.get("bridges", []) if b["fields"].get("prop_structure")),
                "level_blocks_v5": sum(1 for b in v5.get("bridges", []) if b["level_block"]),
                "crossings_v5": len(v5.get("crossings", [])), "curves_v4": len(ann["curves"]), "curves_v5": len(v5.get("curves", [])),
                "curve_degree_v5": sum(1 for c in v5.get("curves", []) if c["degree"]),
                "bridge_table_rows": len(v5.get("bridge_table", [])), "band_rows_v5": sum(len(b["rows"]) for b in v5.get("bands", [])),
                "band_columns_v4": len(band.get("columns", [])), "all_text": len(ann["all_text"]),
                "notes": len(ann["notes"]), "tbms": len(ann["tbm_benchmarks"]), "error": v5.get("error", ""),
            })
            print(f"{sid:<16} {pdf.name[:40]:<40} p{i + 1:<3} bridges {index[-1]['bridges_v4']:>2}/{index[-1]['bridges_v5']:>2} "
                  f"curves {index[-1]['curves_v4']:>2}/{index[-1]['curves_v5']:>2} table {index[-1]['bridge_table_rows']:>2} "
                  f"band rows {index[-1]['band_rows_v5']:>2} texts {index[-1]['all_text']:>5}" + (f"  ERROR {index[-1]['error']}" if index[-1]["error"] else ""),
                  flush=True)
    return index


def write_index(index):
    index.sort(key=lambda r: (r["pdf"], r["page"]))
    with open(OUT / "sheets.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"{len(index)} sheets from {len({r['pdf'] for r in index})} PDFs -> {OUT / 'annotations'}")


if __name__ == "__main__":
    main()
