"""Build the pilot report page and the full answer sheet from a run's results.

    python scripts/make_report.py

Reads  results/predictions_base.jsonl, results/predictions_finetuned.jsonl, results/loss.png and
       results/v1_test_set/test.jsonl (the test questions those predictions answer)
Writes report/railway-drawing-reader-pilot.html  (from report/template.html)
       results/test_answers_v1.csv               (every test question with all three answers)
"""
import base64
import html
import io
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
IMAGES = ROOT / "data" / "dataset" / "images"

# The 15 questions shown in the report: one or two of every kind, including two misses.
SHOWN = ["s100_0147", "s100_0116", "s100_0119", "s100_0104", "s100_0145", "s100_0149", "s100_0164",
         "s100_0168", "s100_0166", "s100_0018", "s100_0006", "s100_0002", "s100_0005", "s100_0032",
         "s100_0171"]

# Verdicts for the 15 shown questions, checked by reading each answer against the drawing.
# The automatic scorer counts values that appear anywhere in an answer, which is fair for scoring
# many answers but misjudges single ones (right values in the wrong fields, a long answer that
# happens to contain the key words), so the cards use these instead.
CHECKED = {  # id: ((base class, base verdict), (fine-tuned class, fine-tuned verdict))
    "s100_0147": (("bad", "wrong: values put in the wrong fields, a TBM label given as the category"), ("ok", "correct")),
    "s100_0116": (("bad", "wrong: chainage and dimensions invented"), ("part", "values right, but calls the 3 slab spans \"cells\" as if it were a box")),
    "s100_0119": (("bad", "wrong: 181.122 is the existing FL, not the proposed FL"), ("ok", "correct")),
    "s100_0104": (("ok", "correct"), ("ok", "correct")),
    "s100_0145": (("ok", "correct"), ("ok", "correct")),
    "s100_0149": (("part", "values right, its own field names"), ("ok", "correct")),
    "s100_0164": (("ok", "correct"), ("ok", "correct")),
    "s100_0168": (("ok", "correct"), ("ok", "correct")),
    "s100_0166": (("ok", "correct"), ("ok", "correct")),
    "s100_0018": (("bad", "wrong: invented a bridge"), ("ok", "correct")),
    "s100_0006": (("bad", "wrong box"), ("ok", "correct box")),
    "s100_0002": (("bad", "wrong box"), ("bad", "wrong box: on bridge 555, the one next to it")),
    "s100_0005": (("bad", "describes the crop but does not name the part of the sheet"), ("ok", "correct")),
    "s100_0032": (("bad", "no clear answer"), ("bad", "wrong: said alignment plan, it is the L-section")),
    "s100_0171": (("ok", "correct"), ("ok", "correct")),
}

KIND = {"bridge_json": "bridge → JSON", "bridge_explain": "bridge → words", "bridge_level_qa": "one level",
        "bridge_level_check": "level check", "curve_json": "curve → JSON", "curve_qa": "one curve value",
        "tbm_qa": "benchmark", "title_qa": "title block", "tile_structures": "find all bridges",
        "bridge_ground": "find one bridge", "tile_region": "part of sheet", "notes_qa": "general notes",
        "tbm_json": "TBM table", "title_json": "title block → JSON", "sheet_layout": "sheet layout",
        "abbr_qa": "abbreviation"}

# One headline metric per task for the score table: (group, label, detail, task, metric, flag).
SCORE_ROWS = [
    ("Reading bridge data", None),
    ("Read a bridge callout as JSON", "Values correct. Fields exact under the right key: {field_acc}", "bridge_json", "value_recall", None),
    ("Explain a bridge in plain words", "Values correct in the explanation", "bridge_explain", "value_recall", None),
    ("Read one level (EXG FL, FL, B.L, HFL…)", "Value correct", "bridge_level_qa", "value_recall", None),
    ("Check proposed FL against MIN FL REQ.", "Verdict and both levels correct", "bridge_level_check", "correct", "all refs Yes"),
    ("Reading other sheet data", None),
    ("Read a horizontal curve data box", "Values correct. Fields exact: {field_acc}", "curve_json", "value_recall", None),
    ("Read one curve parameter", "Value correct", "curve_qa", "value_recall", None),
    ("Read the TBM benchmark table", "Values correct. Fields exact: {field_acc}", "tbm_json", "value_recall", None),
    ("Read one benchmark", "Value correct", "tbm_qa", "value_recall", None),
    ("Read the title block", "Values correct. Fields exact: {field_acc}", "title_json", "value_recall", None),
    ("Read the chainage range", "Value correct", "title_qa", "value_recall", None),
    ("Finding things", None),
    ("Say “no bridge” on an empty crop", "Correct when nothing is there to find", "tile_structures", "empty_ok", "empty crops only"),
    ("Box a named bridge", "Boxes found (IoU ≥ 0.5)", "bridge_ground", "box_recall", None),
    ("Box the regions of a whole sheet", "Regions found (IoU ≥ 0.5)", "sheet_layout", "box_recall", None),
    ("Name the part of the sheet a crop is from", "Key words matched", "tile_region", "word_recall", None),
    ("Memory, not reading", None),
    ("Answer from the general notes", "Key words matched", "notes_qa", "word_recall", "same on every sheet"),
    ("Expand an abbreviation", "Key words matched", "abbr_qa", "word_recall", "same on every sheet"),
]


def scorer():
    """The notebook's own scoring code, so the report and the training run agree."""
    nb = json.loads((ROOT / "notebooks" / "train_qwen25vl_qlora.ipynb").read_text(encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "def score(" in "".join(c["source"]))
    ns = {"json": json}
    exec(src, ns)
    return ns


def data_uri(img, width, fmt="JPEG"):
    img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, fmt, quality=85) if fmt == "JPEG" else img.save(buf, fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def verdict(s):
    """(css class, text) for one scored answer."""
    if "empty_ok" in s:
        return ("ok", "correct") if s["empty_ok"] == 1 else ("bad", "invented a bridge")
    if "tp" in s:
        return ("ok", "correct box") if s["tp"] and not s["fp"] else ("bad", "wrong box")
    if "correct" in s:
        return ("ok", "correct") if s["correct"] == 1 else ("bad", "wrong")
    vr, wr, fa = s.get("value_recall"), s.get("word_recall"), s.get("field_acc")
    main = vr if vr is not None else wr
    if main is None:
        return ("part", "not scored")
    if main == 1 and (fa is None or fa == 1):
        return ("ok", "correct")
    if main == 1:
        return ("part", "values present, not in the requested fields")
    if main >= 0.5:
        return ("part", f"partly right ({round(main * 100)}% of values)")
    return ("bad", "wrong")


def pretty(text, limit=700):
    """Answer text for display: JSON reflowed one field per line, long free text cut."""
    m = re.search(r"```json\s*(.*?)```", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            body = "\n".join(json.dumps(o, ensure_ascii=False) for o in obj) if isinstance(obj, list) else \
                "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v}" for k, v in obj.items())
            return f"<pre>{html.escape(body)}</pre>"
        except json.JSONDecodeError:
            pass
    text = text.strip()
    cut = len(text) > limit
    text = text[:limit].rsplit(" ", 1)[0] + " …" if cut else text
    tag = "pre" if "\n" in text and ("{" in text or "[" in text) else "div"
    return f"<{tag}>{html.escape(text)}</{tag}>" + ('<div class="why">(cut short here)</div>' if cut else "")


def main():
    ns = scorer()
    test = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "v1_test_set" / "test.jsonl", encoding="utf-8")}
    base = pd.read_json(RES / "predictions_base.jsonl", lines=True, dtype={"id": str}).set_index("id")
    ft = pd.read_json(RES / "predictions_finetuned.jsonl", lines=True, dtype={"id": str}).set_index("id")

    # Full answer sheet: every test question with all three answers.
    sheet = []
    for i in ft.index:
        r = test[i]
        ref = r["messages"][1]["content"][0]["text"]
        sb, sf = ns["score"](r, base.loc[i, "pred"]), ns["score"](r, ft.loc[i, "pred"])
        sheet.append({"id": i, "task": r["task"], "image": r["image"], "question": r["messages"][0]["content"][1]["text"],
                      "correct_answer": ref, "base_answer": base.loc[i, "pred"], "base_verdict_auto": verdict(sb)[1],
                      "finetuned_answer": ft.loc[i, "pred"], "finetuned_verdict_auto": verdict(sf)[1],
                      "base_verdict_checked": CHECKED.get(i, ((None, ""),))[0][1],
                      "finetuned_verdict_checked": CHECKED[i][1][1] if i in CHECKED else ""})
    pd.DataFrame(sheet).to_csv(RES / "test_answers_v1.csv", index=False, encoding="utf-8-sig")

    # The 15 question cards.
    cards = []
    for i in SHOWN:
        r = test[i]
        ref = r["messages"][1]["content"][0]["text"]
        pb, pf = base.loc[i, "pred"], ft.loc[i, "pred"]
        img = Image.open(ROOT / "data" / "dataset" / r["image"]).convert("RGB")
        if r["task"] in ("bridge_ground", "tile_structures"):
            d = ImageDraw.Draw(img)
            for b in ns["boxes"](ref):
                d.rectangle(b, outline=(0, 150, 80), width=8)
            for b in ns["boxes"](pf):
                d.rectangle(b, outline=(200, 40, 30), width=5)
        (cb, tb), (cf, tf) = CHECKED[i]
        alt = f"Test crop {r['image'].split('/')[-1]} from sheet 100"
        cards.append(f"""      <article class="qa">
        <img src="{data_uri(img, 440)}" alt="{html.escape(alt)}" loading="lazy">
        <div class="body">
          <div class="qhead"><span class="qtext">{html.escape(r['messages'][0]['content'][1]['text'])}</span><span class="kind">{KIND[r['task']]}</span></div>
          <div class="rows">
            <span class="who">Correct answer</span><div class="ans">{pretty(ref)}</div>
            <span class="who">Untouched model</span><div class="ans">{pretty(pb)}<span class="v {cb}">{tb}</span></div>
            <span class="who">Fine-tuned</span><div class="ans">{pretty(pf)}<span class="v {cf}">{tf}</span></div>
          </div>
        </div>
      </article>""")

    # Score table from the same scorer.
    def summary(df):
        return ns["summary"](pd.DataFrame([{**ns["score"](test[i], p), "id": i} for i, p in zip(df.index, df["pred"])]))
    sb, sf = summary(base), summary(ft)
    pct = lambda v: f"{round(v * 100)}%"
    rows = []
    for item in SCORE_ROWS:
        if item[1] is None:
            rows.append(f'          <tr class="group"><td colspan="3">{html.escape(item[0])}</td></tr>')
            continue
        label, detail, task, metric, flag = item
        b, f = float(sb.loc[task, metric]), float(sf.loc[task, metric])
        if "{field_acc}" in detail:
            detail = detail.format(field_acc=f"{pct(sb.loc[task, 'field_acc'])} → {pct(sf.loc[task, 'field_acc'])}")
        bar = lambda cls, v: (f'<div class="bar {cls}"><div class="track"><div class="fill" style="width:{v * 100:.1f}%"></div></div>'
                              f'<span class="num">{pct(v)}</span></div>')
        fl = f'<span class="flag">{html.escape(flag)}</span>' if flag else ""
        rows.append(f'          <tr><td class="task">{html.escape(label)}{fl}<small>{html.escape(detail)}</small></td>'
                    f'<td class="n">{int(sb.loc[task, "n"])}</td><td><div class="bars">{bar("base", b)}{bar("ft", f)}</div></td></tr>')

    # Fixed illustrations: the bridge 558 example and the find-all-bridges demo tile.
    tile = Image.open(IMAGES / "s100_tile_r0c0.png").convert("RGB")
    d, fnt = ImageDraw.Draw(tile), font(26)
    for box, lab in [([573, 186, 680, 654], "554"), ([699, 244, 806, 712], "555"), ([806, 265, 921, 734], "555A")]:
        d.rectangle(box, outline=(0, 150, 90), width=6)
        tb_ = d.textbbox((box[0], box[1] - 34), lab, font=fnt)
        d.rectangle([tb_[0] - 4, tb_[1] - 4, tb_[2] + 4, tb_[3] + 4], fill=(0, 150, 90))
        d.text((box[0], box[1] - 34), lab, fill="white", font=fnt)

    page = (ROOT / "report" / "template.html").read_text(encoding="utf-8")
    page = (page.replace("{{ROWS}}", "\n".join(rows)).replace("{{QA}}", "\n".join(cards))
                .replace("{{IMG_bridge558}}", data_uri(Image.open(IMAGES / "s100_br558_lsec0.png"), 720))
                .replace("{{IMG_tile}}", data_uri(tile, 720))
                .replace("{{IMG_loss}}", data_uri(Image.open(RES / "loss.png"), 640, "PNG")))
    assert "{{" not in page, "unfilled placeholder in template"
    out = ROOT / "report" / "railway-drawing-reader-pilot.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page) // 1024} KB) and results/test_answers_v1.csv ({len(sheet)} questions)")


if __name__ == "__main__":
    main()
