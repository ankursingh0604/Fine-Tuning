"""Generic text-block reading for v4: bridge callouts and level blocks in styles the model has not seen.

    .venv\\Scripts\\python scripts\\text_blocks_v4.py

Real values from every sheet, drawn in varied styles: field order, labels ("BR. NO." / "BRIDGE No:"), level wording
("F.L. =" / "Formation Level :" / table rows), separators, fonts, colours, sizes, boxed or not, rotated like L-section
text, with clutter lines around. The answer gives every block as printed and the bridge fields recognisable in it:

    {"blocks": [{"text": "BRIDGE No: 594 / ...", "fields": {"bridge_id": "594", "chainage_m": 1256295.109, ...}}]}

Writes data/v4/dataset/textblocks_{train,val,test}.jsonl (+ images/), same sheet split as the main dataset.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as D           # noqa: E402
import build_dataset_v4 as V        # noqa: E402

DS = ROOT / "data" / "v4" / "dataset"
rng = random.Random(53)
VARIANTS = {"train": 3, "val": 1, "test": 1}
FONTS = ["arial.ttf", "arialbd.ttf", "cour.ttf", "times.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf"]
NO_LABEL = ["BR. NO. {v}", "BRIDGE No: {v}", "Br-{v}", "BRIDGE {v}", "EXG. BR. {v}", "No. {v}"]
EXIST = ["EXIST: {t} {s}", "EXG. {t} - {s}", "{t} ({s})", "EXISTING {t} {s}"]
CROSS = ["OVER {c}", "X-ING: {c}", "{c}", "CROSSING - {c}"]
PROP = ["PROP.: {p}", "PROPOSED - {p}", "TO BE EXTENDED AS {p}", "NEW: {p}"]
CAT = ["({c})", "CAT: {c}", "{c} BRIDGE"]
CH = ["CH: {ch}", "AT CH {ch}", "Chainage = {km}", "KM {km}"]
LEVEL_LABEL = {
    "existing_formation_level": ["EXG FL", "EXIST. F.L.", "Existing Formation Level"],
    "min_formation_level_required": ["MIN FL REQ.", "MIN. F.L. REQD", "Min. Formation Level"],
    "proposed_formation_level": ["FL", "PROP. F.L.", "Formation Level"],
    "bed_level": ["B.L", "BED LVL", "Bed Level"],
    "high_flood_level": ["HFL", "H.F.L.", "High Flood Level"],
    "free_board": ["FB", "F.B.", "Free Board"],
}
LEVEL_SEP = [" = ", " : ", " - ", " "]
Q = ["Read every text block in this crop as printed, and give the bridge fields you can recognise in each, as JSON.",
     "This drawing uses an unfamiliar callout style. Transcribe each text block and map what you can to bridge fields (JSON).",
     "Give each block of text in this image as written, plus its bridge number, structure, levels and chainage where shown, as JSON."]


def font(size):
    for name in rng.sample(FONTS, len(FONTS)):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def km_text(ch):
    return f"{int(ch // 1000)}+{ch % 1000:07.3f}"


def block_lines(b):
    """The printed lines of one callout + level block in a random style, and the fields they show."""
    fields, lines = {"bridge_id": b["bridge_id"]}, []
    parts = [("no", rng.choice(NO_LABEL).format(v=b["bridge_id"]))]
    if b.get("existing_type"):
        txt = rng.choice(EXIST).format(t=b["existing_type"], s=b["existing_span"]) if b.get("existing_span") else f"EXIST: {b['existing_type']}"
        parts.append(("exist", txt))
        fields["existing_type"] = b["existing_type"]
        if b.get("existing_span"):
            fields["existing_span"] = b["existing_span"]
    if b.get("crossing"):
        parts.append(("cross", rng.choice(CROSS).format(c=b["crossing"])))
        fields["crossing"] = b["crossing"]
    if b.get("proposal") and rng.random() < 0.85:
        parts.append(("prop", rng.choice(PROP).format(p=b["proposal"])))
        fields["proposal"] = b["proposal"]
    if b.get("category") and rng.random() < 0.6:
        parts.append(("cat", rng.choice(CAT).format(c=b["category"])))
        fields["category"] = b["category"]
    if b.get("chainage_m"):
        parts.append(("ch", rng.choice(CH).format(ch=b["chainage_m"], km=km_text(b["chainage_m"]))))
        fields["chainage_m"] = b["chainage_m"]
    head, rest = parts[0], parts[1:]
    if rng.random() < 0.5:
        rng.shuffle(rest)
    lines += [p for _, p in [head] + rest]
    lv = {k: v for k, v in (b.get("lsection_levels") or {}).items() if k in LEVEL_LABEL and v is not None}
    if lv and rng.random() < 0.8:
        style = rng.randrange(3)
        sep = rng.choice(LEVEL_SEP)
        keys = list(lv)
        if rng.random() < 0.4:
            rng.shuffle(keys)
        for k in keys:
            lab = rng.choice(LEVEL_LABEL[k])
            lines.append(f"{lab}{sep}{lv[k]}" if style < 2 else f"{lab} | {lv[k]}")
        fields["levels"] = {k: lv[k] for k in keys}
    return lines, fields


def draw_blocks(blocks):
    """Render 1-3 blocks with clutter. Returns the image."""
    size = rng.choice([16, 18, 20, 22, 24])
    f = font(size)
    rendered = []
    for lines, _ in blocks:
        w = max(int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(l, font=f)) for l in lines) + 24
        h = (size + 6) * len(lines) + 20
        im = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(im)
        color = rng.choice([(0, 0, 0), (200, 0, 0), (0, 0, 160)])
        for i, l in enumerate(lines):
            d.text((12, 10 + i * (size + 6)), l, fill=color, font=f)
        if rng.random() < 0.6:
            d.rectangle([2, 2, w - 3, h - 3], outline=color, width=2)
        if rng.random() < 0.3:
            im = im.rotate(90, expand=True, fillcolor="white")
        rendered.append(im)
    gap = 40
    W = sum(im.width for im in rendered) + gap * (len(rendered) + 1)
    H = max(im.height for im in rendered) + 2 * gap
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for _ in range(rng.randint(2, 8)):                         # clutter: drawing lines
        x0, y0 = rng.randrange(W), rng.randrange(H)
        d.line([x0, y0, x0 + rng.randint(-W, W), y0 + rng.randint(-60, 60)], fill=(120, 120, 120), width=rng.choice([1, 2]))
    x = gap
    for im in rendered:
        canvas.paste(im, (x, gap + rng.randint(0, H - 2 * gap - im.height) if H - 2 * gap > im.height else gap))
        x += im.width + gap
    return D.pad28(canvas)


def main():
    D.OUT, D.IMG = DS, DS / "images"
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(V.ANN.glob("*.json"))]
    out = {"train": [], "val": [], "test": []}
    for a in anns:
        sid = a["sheet_id"]
        split = "test" if (a["source_pdf"] == V.TEST_PDF or sid in V.TEST) else "val" if sid in V.VAL else "train"
        bs = [b for b in a["bridges"] if b.get("complete") and b.get("chainage_m") and b.get("belongs_to") == "this sheet"]
        for v in range(VARIANTS[split]):
            for i in range(0, len(bs), 3):
                grp = bs[i:i + rng.choice([1, 2, 3])]
                if not grp:
                    continue
                blocks = [block_lines(b) for b in grp]
                img = draw_blocks(blocks)
                path = D.save(img, f"tb_{sid}_{v}_{i}")
                ans = {"blocks": [{"text": " / ".join(lines), "fields": fields} for lines, fields in blocks]}
                row = D.row(sid, "text_blocks_json", path, img.size, rng.choice(Q), D.fenced(ans))
                row["id"], row["split"] = f"{sid}_tb{v}_{i:03d}", split
                out[split].append(row)
    for split, rows in out.items():
        with open(DS / f"textblocks_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("id", "sheet", "split", "task", "image", "width", "height", "messages")},
                                   ensure_ascii=False) + "\n")
    print({s: len(r) for s, r in out.items()})


if __name__ == "__main__":
    main()
