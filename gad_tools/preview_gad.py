"""A page with sample rows of every task of the GAD dataset (question, facts given, answer, image).

    .venv\\Scripts\\python gad_tools\\preview_gad.py        -> data/gad/dataset/preview.html
"""
import html
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "gad" / "dataset"
PER_TASK = 3


def text(content):
    return content if isinstance(content, str) else "\n".join(c.get("text", "") for c in content if c.get("type") == "text")


def main():
    rows = [json.loads(l) for l in open(DS / "train.jsonl", encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    rng = random.Random(4)
    parts = []
    for task in sorted(by):
        parts.append(f"<h2>{task} <small>({len(by[task])} rows)</small></h2>")
        for r in rng.sample(by[task], min(PER_TASK, len(by[task]))):
            q = text(r["messages"][0]["content"])
            facts, _, question = q.rpartition("\n\nQuestion: ")
            if not _:
                facts, question = "", q
            img = f'<img src="{r["image"]}" loading="lazy">' if r.get("image") else ""
            fx = f"<details><summary>GAD facts given to the model</summary><pre>{html.escape(facts)}</pre></details>" if facts else ""
            parts.append(f'<div class="card"><div class="meta">{r["id"]}</div>{img}<p class="q">Q: {html.escape(question)}</p>{fx}'
                         f'<pre class="a">{html.escape(text(r["messages"][1]["content"]))}</pre></div>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GAD dataset preview</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--accent:#2f5d8a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1b1b1a;--fg:#ecebe6;--muted:#a3a29c;--card:#252524;--line:#3a3a37;--accent:#8fb4dc}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:16px}}
h2{{color:var(--accent);margin-top:2em}} small{{color:var(--muted);font-weight:normal}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}}
.meta{{color:var(--muted);font-size:12px}} .q{{font-weight:600}} pre{{white-space:pre-wrap;word-break:break-word;font-size:13px}}
.a{{background:rgba(47,93,138,.08);padding:8px;border-radius:6px}} img{{max-width:100%;border:1px solid var(--line)}}
</style></head><body><h1>GAD dataset preview</h1>
<p>{len(rows)} training rows. {PER_TASK} random rows per task. Answers come from the drawings' own text; meanings from gad_tools/gad_kinds.py.</p>
{''.join(parts)}</body></html>"""
    (DS / "preview.html").write_text(page, encoding="utf-8")
    print(DS / "preview.html")


if __name__ == "__main__":
    main()
