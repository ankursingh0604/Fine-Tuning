"""A self-contained HTML preview of the v4 datasets: a few rows of every task, normal and reasoning.

    .venv\\Scripts\\python scripts\\preview_dataset_v4.py      -> data/v4/dataset/preview.html

Images are embedded (downscaled JPEG), so the page opens anywhere without the image folder.
"""
import base64
import html
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "v4" / "dataset"
PER_TASK = 2
rng = random.Random(3)


def img_tag(rel, max_w=620):
    im = Image.open(DS / rel).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}" alt="{html.escape(rel)}">'


def text_of(msg):
    return "\n".join(c["text"] for c in msg["content"] if c.get("type") == "text")


def answer_html(ans):
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)", ans, re.S)
    if not m:
        return f'<pre class="ans">{html.escape(ans)}</pre>'
    return (f'<div class="think"><div class="lbl">Reasoning (inside &lt;think&gt;)</div><pre>{html.escape(m.group(1).strip())}</pre></div>'
            f'<div class="lbl">Final answer</div><pre class="ans">{html.escape(m.group(2).strip())}</pre>')


def section(title, files, intro):
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(DS / f, encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["task"]][r["split"]] += 1
    out = [f"<h2>{title}</h2><p class='intro'>{intro}</p>",
           "<table class='counts'><tr><th>Task</th><th>Train</th><th>Val</th><th>Test</th></tr>"]
    for t in sorted(counts):
        c = counts[t]
        out.append(f"<tr><td>{t}</td><td>{c['train']}</td><td>{c['val']}</td><td>{c['test']}</td></tr>")
    tot = defaultdict(int)
    for t in counts:
        for s, n in counts[t].items():
            tot[s] += n
    out.append(f"<tr class='tot'><td>Total</td><td>{tot['train']}</td><td>{tot['val']}</td><td>{tot['test']}</td></tr></table>")
    for t in sorted(by):
        out.append(f"<h3>{t}</h3>")
        for r in rng.sample(by[t], min(PER_TASK, len(by[t]))):
            user, asst = r["messages"][0], r["messages"][1]
            imgs = r.get("images") or [r["image"]]
            out.append("<div class='card'>"
                       f"<div class='meta'>{html.escape(r['id'])} &middot; sheet {html.escape(r['sheet'])} &middot; {r['split']}</div>"
                       f"<div class='imgs'>{''.join(img_tag(i) for i in imgs)}</div>"
                       f"<div class='lbl'>Question</div><pre class='q'>{html.escape(text_of(user))}</pre>"
                       f"{answer_html(text_of(asst))}</div>")
    return "\n".join(out)


def main():
    stats = json.loads((DS / "stats.json").read_text(encoding="utf-8"))
    body = [
        "<h1>v4 datasets &mdash; preview</h1>",
        f"<p class='intro'>Built from 114 sheets in 14 PDFs. Test: the whole PDF MKN_PnP_1203-1244_4th Line plus "
        f"{', '.join(s for s in stats['splits']['test'] if not s.startswith('MKN-4TH_1') or s == 'MKN-4TH_101')}; "
        f"validation: {', '.join(stats['splits']['val'])}. Every answer is generated from the PDF's own text, so the "
        f"values are exactly what is printed. {PER_TASK} random rows per task are shown.</p>",
        section("Normal dataset (direct answers)", ["train.jsonl", "val.jsonl", "test.jsonl"],
                "Reading tasks: the model answers directly, without reasoning."),
        section("Reasoning dataset (chain of thought)", ["reasoning_train.jsonl", "reasoning_val.jsonl", "reasoning_test.jsonl"],
                "Questions that combine, check or decide: numbered steps inside &lt;think&gt;, then the final answer. "
                "Some rows use two images (e.g. a bridge callout and the data bands)."),
    ]
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v4 dataset preview</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--accent:#2f5d8a;--think:#f3f6fa}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#ecebe6;--muted:#a3a29b;--card:#242422;--line:#3a3936;--accent:#8db6e0;--think:#1f2630}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:26px}h2{margin-top:40px;border-bottom:2px solid var(--accent);padding-bottom:4px}h3{margin:28px 0 8px;color:var(--accent)}
.intro{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0}
.meta{font-size:12px;color:var(--muted);margin-bottom:6px}.imgs{display:flex;flex-wrap:wrap;gap:8px}
.imgs img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px}
.lbl{font-size:12px;font-weight:600;color:var(--muted);margin-top:8px;text-transform:uppercase;letter-spacing:.04em}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font:13px/1.45 ui-monospace,Consolas,monospace}
.think{background:var(--think);border-radius:6px;padding:6px 10px;margin-top:8px}
table.counts{border-collapse:collapse;margin:8px 0;font-size:13px}table.counts td,table.counts th{border:1px solid var(--line);padding:3px 10px;text-align:right}
table.counts td:first-child,table.counts th:first-child{text-align:left}tr.tot td{font-weight:700}
</style></head><body>""" + "\n".join(body) + "</body></html>"
    (DS / "preview.html").write_text(page, encoding="utf-8")
    print(f"wrote {DS / 'preview.html'} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
