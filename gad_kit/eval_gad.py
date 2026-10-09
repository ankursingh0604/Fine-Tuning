"""Score a trained GAD adapter on the held-out GAD (test.jsonl) and validation GADs.

    python eval_gad.py --adapter gad_run/adapter [--split test] [--per-task 0]

Scoring per row (pass / fail with the reason):
  questions      every number in the reference answer appears in the model's answer (same decimals); for "not on the
                 drawing" questions the answer must say so and give no number that is not in the question's facts
  img_view       the view's title is named
  img_tile/table JSON: share of reference values found (a row passes at >= 90 %)
Writes <out>/test_scores.csv, test_summary.txt, test_predictions.jsonl.
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_gad as DG      # noqa: E402

NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def generate(model, processor, msgs, images, max_new=800):   # (the longest test answers are ~610 tokens)
    import torch
    text = DG.render(processor, msgs, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def nums(text):
    out = set()
    for x in NUM.findall(text or ""):
        out.add(x)
        if "." in x:
            out.add(f"{float(x):.3f}")
    return out


def flat(obj):
    """All scalar values of a JSON answer, as strings."""
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in flat(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in flat(x)]
    return [str(obj)]


def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def score(row, pred):
    task = row["task"]
    ref = DG.text_of(row["messages"][1]["content"])
    if task in ("img_tile", "img_table"):
        r, p = parse_json(ref), parse_json(pred)
        if p is None:
            return False, "not valid JSON", 0.0
        want = [v for v in flat(r) if v not in ("", "None") and len(v) < 60 and not v.startswith("the ")]
        got = set(flat(p))
        got_n = nums(" ".join(got))
        hit = sum(1 for v in want if v in got or (NUM.fullmatch(v) and (v in got_n or f"{float(v):.3f}" in got_n)))
        share = hit / max(1, len(want))
        return share >= 0.9, f"{hit}/{len(want)} values", share
    if task == "img_view":
        title = re.search(r"This is the (.+?) \(scale", ref)
        ok = bool(title) and title.group(1).lower() in pred.lower()
        return ok, "" if ok else "view not named", float(ok)
    if task == "qa_absent":
        q = DG.text_of(row["messages"][0]["content"])
        new = {x for x in NUM.findall(pred) if x not in q}
        says = re.search(r"\b(not|no|does not|doesn't|isn't|there is no|I don't)\b", pred, re.I)
        ok = bool(says) and not new
        return ok, "" if ok else ("invented " + ",".join(sorted(new)) if new else "did not say it is absent"), float(ok)
    if task == "qa_component":
        name = re.search(r"^The (.+?) is ", ref)
        ok = bool(name) and name.group(1).split(" (")[0].lower() in pred.lower()
        if not ok and not name:                                # the "all components" answer: most parts named
            parts = re.findall(r"(?:^|\. |: )([A-Z][a-z /()]+?): ", ref)
            got = sum(1 for p in parts if p.lower() in pred.lower())
            ok = got >= 0.8 * max(1, len(parts))
            return ok, "" if ok else f"named {got}/{len(parts)} parts", got / max(1, len(parts))
        return ok, "" if ok else "component not named", float(ok)
    if task == "qa_other":                                     # the same meaning case: known / web fits / web off-topic / unknown
        def case(t):
            if re.search(r"does not fit|doesn't fit", t, re.I):
                return "web off-topic"
            if re.search(r"web search found nothing|can't say what it denotes|cannot say what it denotes|meaning is not known", t, re.I):
                return "unknown"
            if re.search(r"web search|from the web|found on the web", t, re.I):
                return "web fits"
            return "known"
        if case(ref) != case(pred):
            return False, f"meaning case {case(pred)} (reference: {case(ref)})", 0.0
    if task == "qa_callout":                                   # where it is written: the same views, and "not on that view" when so
        neg = lambda t: bool(re.search(r"\bis not written on\b|\bnot (?:shown|written) on\b|\bnot on the\b", t, re.I))
        if neg(ref) != neg(pred):
            return False, "said it is on the view" if neg(ref) else "said it is not on the view", 0.0
        m = re.search(r"(?:written on|it is on) the (.+?)(?: \(drawn|\.|$)", ref.split(";")[-1] if neg(ref) else ref)
        if m and not all(v.strip().lower() in pred.lower() for v in m.group(1).split(", ")[:2]):
            return False, "view not named", 0.0
    if task == "qa_cl" and "no value of its own" in ref and not re.search(r"no value|not a value|reference line", pred, re.I):
        return False, "gave a centre line a value", 0.0
    if task in ("qa_reason", "qa_reason_flag"):                # the same verdict as the reference, and its numbers
        def verdict(t):
            # a flag in any wording: "NOT what the note asks", "worth checking", "a different grade", "does not follow" ...
            return "flag" if re.search(r"\bNOT\b|\bnot\s+(?:what|agree|match|consistent|follow)|worth checking|needs? checking|"
                                       r"less than \d|does not (?:agree|follow|match)|doesn't|\bdifferent\b|\bmismatch|"
                                       r"\binconsistent\b|should be checked", t, re.I) else "agree"
        if verdict(ref) != verdict(pred):
            return False, f"verdict {verdict(pred)} (reference: {verdict(ref)})", 0.0
    want = {x for x in NUM.findall(ref) if len(x.replace(".", "")) >= 2 or "." in x}
    q = DG.text_of(row["messages"][0]["content"])
    want -= {x for x in NUM.findall(q.split("Question:")[-1])}          # numbers the question itself contains
    got = nums(pred)
    missing = [x for x in want if x not in got and (("." not in x) or f"{float(x):.3f}" not in got)]
    ok = not missing
    return ok, "" if ok else "missing " + ",".join(sorted(missing)[:6]), 1 - len(missing) / max(1, len(want))


def evaluate(model, processor, ds, out, split="test", per_task=0, log=print):
    rows = DG.load_rows(ds, split, limit_per_task=per_task or None, seed=5)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res, by, by_gad = [], defaultdict(list), defaultdict(list)
    t0 = time.time()
    with open(out / f"{split}_predictions.jsonl", "w", encoding="utf-8") as fp:
        for i, row in enumerate(rows):
            msgs, imgs = DG.to_chat(row, ds)
            pred = generate(model, processor, msgs[:-1], imgs)
            ok, why, part = score(row, pred)
            by[row["task"]].append(ok)
            by_gad[row.get("gad", "?")].append(ok)
            res.append({"id": row["id"], "task": row["task"], "pass": ok, "why": why, "partial": round(part, 3)})
            fp.write(json.dumps({"id": row["id"], "task": row["task"], "pass": ok, "why": why,
                                 "question": DG.text_of(row["messages"][0]["content"]).split("Question:")[-1].strip()[:300],
                                 "reference": DG.text_of(row["messages"][1]["content"]), "prediction": pred}, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                log(f"  {split} {i + 1}/{len(rows)}: {sum(r['pass'] for r in res)} passed, about {el / (i + 1) * (len(rows) - i - 1) / 60:.0f} min left")
    with open(out / f"{split}_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    lines = [f"{split}: {sum(r['pass'] for r in res)}/{len(res)} passed ({100 * sum(r['pass'] for r in res) / max(1, len(res)):.1f} %)"]
    lines += [f"  {t:<16} {sum(v)}/{len(v)}" for t, v in sorted(by.items())]
    lines += ["by GAD:"] + [f"  {g:<16} {sum(v)}/{len(v)} ({100 * sum(v) / max(1, len(v)):.0f} %)" for g, v in sorted(by_gad.items())]
    (out / f"{split}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for l in lines:
        log(l)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-task", type=int, default=0, help="rows per task (0 = all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(args.adapter, load_in_4bit=False)
    FastVisionModel.for_inference(model)
    evaluate(model, processor, DG.find_dataset(), args.out or Path(args.adapter).parent / "eval", args.split, args.per_task)
