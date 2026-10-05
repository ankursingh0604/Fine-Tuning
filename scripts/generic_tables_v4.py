"""Generic data-band table reading for v4: tables whose rows, labels and spacing differ from the trained layout.

    .venv\\Scripts\\python scripts\\generic_tables_v4.py

From real band crops, variants are assembled row by row: rows re-ordered, one or two dropped, some relabelled
with a synonym of the same meaning (the label cell is blanked and the new text drawn), and sometimes only every
second column kept (40 m spacing). The answer names rows by their printed label, so the model learns to read any
band table by its labels instead of by row position:

    {"columns": [1256280, 1256300, ...], "rows": {"N.G.L. (m)": [172.308, 172.303, ...], ...}}

Writes data/v4/dataset/generic_{train,val,test}.jsonl (+ images/), same sheet split as the main dataset.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as D           # noqa: E402
import build_dataset_v4 as V        # noqa: E402

DS = ROOT / "data" / "v4" / "dataset"
SEED = 31
VARIANTS_PER_SHEET = 4
WINDOW = 8                           # columns per table (16 when every second column is kept)
rng = random.Random(SEED)
FIELDS = ["cut_fill", "fl_difference", "prop_rl", "prop_fl", "track_distance", "exg_up_fl", "ground_level"]
SYNONYMS = {   # labels of the same meaning, as other drawing sets print them
    "cut_fill": ["BANK HEIGHT (+) / CUTTING (-)", "CUTTING (-) / FILLING (+) (m)", "HEIGHT OF BANK / DEPTH OF CUT"],
    "fl_difference": ["DIFF. IN FORMATION LEVEL (PROP. - EXG.)", "FL DIFFERENCE (m)"],
    "prop_rl": ["PROPOSED RAIL LEVEL", "PROP. R.L. (m)", "RAIL LEVEL (PROPOSED)"],
    "prop_fl": ["PROPOSED FORMATION LEVEL", "PROP. F.L. (m)", "FORMATION LEVEL (PROPOSED)"],
    "track_distance": ["TRACK CENTRE (m)", "DISTANCE BETWEEN TRACK CENTRES", "C/C OF TRACKS (m)"],
    "exg_up_fl": ["EXISTING {L} FORMATION LEVEL", "EXG. {L} F.L. (m)"],
    "ground_level": ["NATURAL GROUND LEVEL", "N.G.L. (m)", "EXISTING GROUND LEVEL (m)"],
}
Q_TABLE = ["Read this data-band table: give every row by its printed label, with its value in each chainage column, as JSON.",
           "Transcribe this L-section data table row by row (use the row labels as printed) as JSON.",
           "Give all rows and columns of this band table as JSON, naming each row as it is labelled."]


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap(text, width=18):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def relabel(cell, text, color):
    """Blank a label cell (keeping its border lines) and draw new label text, centred."""
    img = cell.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([5, 4, w - 6, h - 5], fill="white")
    lines = wrap(text)
    f = font(max(14, min(22, int(h / (len(lines) + 1.2)))))
    lh = f.getbbox("Ag")[3] + 3
    y = (h - lh * len(lines)) / 2
    for ln in lines:
        tw = d.textlength(ln, font=f)
        d.text(((w - tw) / 2, y), ln, fill=color, font=f)
        y += lh
    return img


def rule_rows(img):
    """y (px) of the horizontal table rules in a label-strip image: rows dark across nearly the whole width."""
    g = np.asarray(img.convert("L")) < 160
    ys = np.flatnonzero(g.mean(axis=1) > 0.85)
    out = []
    for y in ys:
        if out and y - out[-1][-1] <= 2:
            out[-1].append(y)
        else:
            out.append([y])
    return [(min(r), max(r)) for r in out]


def printed_label(ann, box):
    """The label text printed inside a cell (several text lines joined)."""
    t = [l for l in ann["all_text"] if box[0] - 2 <= l["bbox"][0] and l["bbox"][2] <= box[2] + 2
         and box[1] <= (l["bbox"][1] + l["bbox"][3]) / 2 <= box[3]]
    return re.sub(r"\s+", " ", " ".join(l["text"] for l in sorted(t, key=lambda l: (l["bbox"][1], l["bbox"][0])))).strip()


def tables_for_sheet(page, ann, sid, split):
    bd = ann.get("bands")
    if not isinstance(bd, dict) or not bd.get("columns") or len(bd.get("row_y", [])) != 8:
        return []
    cols = sorted((c for c in bd["columns"] if c["checks_ok"]), key=lambda c: c["chainage"])
    if len(cols) < 2 * WINDOW + 2:
        return []
    line = (bd.get("lines") or {}).get("existing") or "UP LINE"
    ry = bd["row_y"]
    # cell boundaries halfway between the rows' text
    bounds = [ry[0][0] - 6] + [(ry[i][1] + ry[i + 1][0]) / 2 for i in range(7)] + [ry[-1][1] + 6]
    ls = bd["label_strip"]
    Z = D.ZOOM
    rows_out = []
    for v in range(VARIANTS_PER_SHEET):
        every2 = rng.random() < 0.3
        n = WINDOW * (2 if every2 else 1)
        start = rng.randrange(0, len(cols) - n)
        win = cols[start:start + n]
        if any(b["chainage"] - a["chainage"] != 20 for a, b in zip(win, win[1:])):
            continue                                    # a failed column inside: keep tables regular
        shown = win[::2] if every2 else win
        x0, x1 = win[0]["bbox"][0] - 0.6, win[-1]["bbox"][2] + 0.6
        top = bounds[0] - 18                            # margin so the table's top and bottom rules are inside the crop
        band_img, _ = D.render(page, [x0, top, x1, bounds[-1] + 18])
        strip_img, _ = D.render(page, [ls[0] - 4, top, ls[2] + 6, bounds[-1] + 18])
        # value rows to show: drop 0-2, re-order sometimes; the chainage row stays (top or bottom)
        fields = FIELDS[:]
        for _ in range(rng.choice([0, 0, 1, 2])):
            fields.remove(rng.choice(fields))
        if rng.random() < 0.6:
            rng.shuffle(fields)
        order = (["chainage"] + fields) if rng.random() < 0.2 else (fields + ["chainage"])
        parts, labels = [], {}
        rules = rule_rows(strip_img)
        for f in order:
            i = 7 if f == "chainage" else FIELDS.index(f)
            # cut along the table's own rules around this row's numbers (labels are taller than the numbers)
            ta, tb = (ry[i][0] - top) * Z, (ry[i][1] - top) * Z
            above = [r for r in rules if r[1] <= ta + 2]
            below = [r for r in rules if r[0] >= tb - 2]
            ya = above[-1][0] if above else round((bounds[i] - top) * Z)
            yb = below[0][1] + 1 if below else round((bounds[i + 1] - top) * Z)
            cell = strip_img.crop((0, ya, strip_img.width, yb))
            label = printed_label(ann, [ls[0] - 4, top + ya / Z, ls[2] + 6, top + yb / Z])
            if f != "chainage" and rng.random() < 0.45:
                label = rng.choice(SYNONYMS[f]).format(L=line)
                cell = relabel(cell, label, rng.choice([(0, 0, 0), (220, 0, 0)]))
            row_img = band_img.crop((0, ya, band_img.width, yb))
            if every2:                                  # keep columns 0, 2, 4, ... cut in the gaps between columns
                pieces = []
                for c in shown:
                    cx0 = round((c["bbox"][0] - 0.6 - x0) * Z)
                    cx1 = round((c["bbox"][2] + 0.6 - x0) * Z)
                    pieces.append(row_img.crop((cx0, 0, cx1, row_img.height)))
                gap = round(11.34 * Z) - pieces[0].width
                wsum = sum(p.width for p in pieces) + gap * (len(pieces) - 1)
                row_img2 = Image.new("RGB", (wsum, row_img.height), "white")
                x = 0
                for p in pieces:
                    row_img2.paste(p, (x, 0))
                    x += p.width + gap
                row_img = row_img2
            parts.append((cell, row_img))
            labels[f] = label
        W = max(c.width for c, _ in parts) + max(r.width for _, r in parts)
        H = sum(c.height for c, _ in parts)
        img = Image.new("RGB", (W, H), "white")
        y = 0
        sw = max(c.width for c, _ in parts)
        for c, r in parts:
            img.paste(c, (0, y))
            img.paste(r, (sw, y))
            y += c.height
        img = D.pad28(img)
        path = D.save(img, f"gt_{sid}_{v}")
        chs = [D.ch_text(c["chainage"]) for c in shown]
        chs = [int(x) if float(x).is_integer() else float(x) for x in chs]
        table = {"columns": chs, "rows": {labels[f]: [c["values"][f] for c in shown] for f in order if f != "chainage"}}
        rows_out.append(D.row(sid, "generic_table_json", path, img.size, rng.choice(Q_TABLE), D.fenced(table)))
        f = rng.choice([f for f in order if f != "chainage"])
        c = rng.choice(shown)
        rows_out.append(D.row(sid, "generic_table_qa", path, img.size,
                              f"In this table, what is the {labels[f]} at chainage {D.ch_text(c['chainage'])}?",
                              f"The {labels[f]} at chainage {D.ch_text(c['chainage'])} is {c['values'][f]}."))
        if rng.random() < 0.4:
            rows_out.append(D.row(sid, "generic_table_rows", path, img.size,
                                  "Which rows does this data-band table have, from top to bottom, and how far apart are its columns?",
                                  "Rows, as printed: " + "; ".join(labels[f] for f in order) +
                                  f". Columns are {40 if every2 else 20} m apart, from chainage {D.ch_text(shown[0]['chainage'])} "
                                  f"to {D.ch_text(shown[-1]['chainage'])}."))
    return rows_out


def main():
    D.OUT, D.IMG = DS, DS / "images"
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(V.ANN.glob("*.json"))]
    out = {"train": [], "val": [], "test": []}
    pdfs = {}
    for a in anns:
        sid = a["sheet_id"]
        split = "test" if (a["source_pdf"] == V.TEST_PDF or sid in V.TEST) else "val" if sid in V.VAL else "train"
        page = pdfs.setdefault(a["source_pdf"], pymupdf.open(ROOT / a["source_pdf"]))[a["page_index"]]
        rows = tables_for_sheet(page, a, sid, split)
        for i, r in enumerate(rows):
            r["id"], r["split"] = f"{sid}_gt{i:03d}", split
        out[split] += rows
    for split, rows in out.items():
        with open(DS / f"generic_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("id", "sheet", "split", "task", "image", "width", "height", "messages")},
                                   ensure_ascii=False) + "\n")
    print({s: dict(Counter(r["task"] for r in rows)) for s, rows in out.items()})


if __name__ == "__main__":
    main()
