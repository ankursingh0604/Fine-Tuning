"""Image-scale augmentation for the v4 dataset (run after build_dataset_v4.py).

    .venv\\Scripts\\python scripts\\augment_v4.py

1. Low-DPI copies: reading-task images scaled down to 60-120 dpi and back up to their 150 dpi size (what the
   whole-sheet reader feeds the model from a low-DPI image), half of them also JPEG-compressed like a scan.
   Same questions and answers (boxes keep their pixel positions).
2. True 200 dpi crops: bridges, bands, curves, title block and TBM table rendered again from the PDFs at 200 dpi
   for a share of the training sheets (more detail, not an upscale).
3. A low-DPI test set (75 and 100 dpi) from the test rows, to measure reading accuracy at low DPI.

Writes data/v4/dataset/aug_train.jsonl, aug_reasoning_train.jsonl, test_lowdpi.jsonl (+ images/).
"""
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as D           # noqa: E402
import build_dataset_v4 as V        # noqa: E402

DS = ROOT / "data" / "v4" / "dataset"
IMG = DS / "images"
SEED = 23
LOW_DPI = [60, 75, 90, 100, 120]
READING = {"bridge_json", "bridge_level_qa", "bridge_level_check", "band_json", "band_qa", "band_bridge", "curve_json", "curve_qa",
           "tbm_json", "tbm_qa", "title_json", "title_qa", "tile_structures", "bridge_ground", "notes_qa", "abbr_qa",
           "bridge_table_json", "bridge_table_qa"}
REASON_LOW = {"reason_bridge_bands", "reason_level_check", "reason_band_cutfill", "reason_band_check"}
LOW_SHARE = 0.25                    # share of eligible training rows that get a low-DPI copy
Z200_SHEET_SHARE = 0.4              # share of training sheets rendered again at 200 dpi
TEST_LOW_PER_DPI = 400
rng = random.Random(SEED)
_cache = {}


def degrade(rel_path, dpi, jpeg):
    """The image as it would look read from a `dpi` scan and brought back to 150 dpi."""
    key = (rel_path, dpi, jpeg)
    if key in _cache:
        return _cache[key]
    src = Image.open(DS / rel_path).convert("RGB")
    w, h = src.size
    s = dpi / 150
    small = src.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BOX)
    if jpeg:
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=jpeg)
        small = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    out = small.resize((w, h), Image.BICUBIC)
    name = f"low{dpi}{'j' if jpeg else ''}_{Path(rel_path).stem}"
    _cache[key] = D.save(out, name)
    return _cache[key]


def low_copy(row, dpi, jpeg, reasoning=False):
    r = json.loads(json.dumps(row))
    if reasoning:
        r["images"] = [degrade(p, dpi, jpeg) for p in row["images"]]
        r["image"] = r["images"][0]
        k = 0
        for c in r["messages"][0]["content"]:
            if c.get("type") == "image":
                c["image"] = r["images"][k]
                k += 1
    else:
        r["image"] = degrade(row["image"], dpi, jpeg)
        for c in r["messages"][0]["content"]:
            if c.get("type") == "image":
                c["image"] = r["image"]
    r["id"] = f"{row['id']}_low{dpi}{'j' if jpeg else ''}"
    r["aug"] = f"low_dpi_{dpi}" + ("_jpeg" if jpeg else "")
    return r


def low_dpi_rows(rows, eligible, share, reasoning=False):
    out = []
    for row in rows:
        if row["task"] in eligible and rng.random() < share:
            out.append(low_copy(row, rng.choice(LOW_DPI), rng.choice([None, rng.randint(55, 85)]), reasoning))
    return out


def z200_rows():
    """Bridges, bands, curves and panel crops rendered at 200 dpi for a share of the training sheets."""
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(V.ANN.glob("*.json"))]
    split_of = {a["sheet_id"]: "test" if (a["source_pdf"] == V.TEST_PDF or a["sheet_id"] in V.TEST) else
                "val" if a["sheet_id"] in V.VAL else "train" for a in anns}
    held, held_ch = {}, {}
    for a in anns:
        if split_of[a["sheet_id"]] != "train":
            s = V.series(a["sheet_id"])
            held.setdefault(s, set()).update(b["bridge_id"] for b in a["bridges"])
            held_ch.setdefault(s, set()).update(c["chainage"] for c in (a.get("bands") or {}).get("columns", []))
    train = [a for a in anns if split_of[a["sheet_id"]] == "train"]
    chosen = rng.sample(train, round(len(train) * Z200_SHEET_SHARE))
    D.ZOOM = 200 / 72                                  # every crop, box and size below is at 200 dpi
    save150 = D.save
    D.save = lambda img, name: save150(img, "z200_" + name)
    rows, pdfs = [], {}
    for a in chosen:
        sid, s = a["sheet_id"], V.series(a["sheet_id"])
        facts = V.sheet_facts(a)
        V.rename_lines(a, facts)
        D.SHEET.clear()
        D.SHEET.update(facts)
        page = pdfs.setdefault(a["source_pdf"], pymupdf.open(ROOT / a["source_pdf"]))[a["page_index"]]
        sheet_rows = []
        for job in (lambda: D.bridge_crops(page, a, sid, 1, held.get(s, set())),
                    lambda: D.curve_crops(page, a, sid),
                    lambda: D.panel_crops(page, a, sid),
                    lambda: D.band_crops(page, a, sid, held_ch.get(s, set())) if (a.get("bands") or {}).get("columns") else []):
            try:
                sheet_rows += job()
            except Exception as e:                 # noqa: BLE001
                print(f"  {sid}: {type(e).__name__}: {e}")
        for i, r in enumerate(sheet_rows):
            r["id"], r["split"], r["aug"] = f"{sid}_z200_{i:04d}", "train", "dpi_200"
        rows += sheet_rows
        print(f"  200 dpi {sid}: {len(sheet_rows)} rows")
    D.ZOOM, D.save = 150 / 72, save150
    return rows


def write(name, rows):
    keys = ("id", "sheet", "split", "task", "image", "images", "sizes", "width", "height", "aug", "messages")
    with open(DS / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keys if k in r}, ensure_ascii=False) + "\n")


def main():
    D.OUT, D.IMG = DS, IMG
    load = lambda f: [json.loads(l) for l in open(DS / f, encoding="utf-8")]      # noqa: E731
    train, rtrain, test = load("train.jsonl"), load("reasoning_train.jsonl"), load("test.jsonl")
    low = low_dpi_rows(train, READING, LOW_SHARE)
    print(f"low-DPI copies: {len(low)} normal rows")
    rlow = low_dpi_rows(rtrain, REASON_LOW, LOW_SHARE, reasoning=True)
    print(f"low-DPI copies: {len(rlow)} reasoning rows")
    z = z200_rows()
    print(f"200 dpi: {len(z)} rows")
    write("aug_train.jsonl", low + z)
    write("aug_reasoning_train.jsonl", rlow)
    eligible = [r for r in test if r["task"] in READING]
    tl = []
    for dpi in (75, 100):
        tl += [low_copy(r, dpi, None) for r in rng.sample(eligible, min(TEST_LOW_PER_DPI, len(eligible)))]
    for r in tl:
        r["split"] = "test_lowdpi"
    write("test_lowdpi.jsonl", tl)
    stats = json.loads((DS / "stats.json").read_text(encoding="utf-8"))
    stats["augmented"] = {"aug_train": {"rows": len(low) + len(z), "low_dpi": len(low), "dpi_200": len(z),
                                        "tasks": dict(sorted(Counter(r["task"] for r in low + z).items()))},
                          "aug_reasoning_train": {"rows": len(rlow), "tasks": dict(sorted(Counter(r["task"] for r in rlow).items()))},
                          "test_lowdpi": {"rows": len(tl), "per_dpi": dict(Counter(r["aug"] for r in tl))}}
    (DS / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats["augmented"], indent=1)[:1500])


if __name__ == "__main__":
    main()
