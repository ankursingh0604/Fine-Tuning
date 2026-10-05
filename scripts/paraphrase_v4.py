"""Varied question wording for the v4 training rows (answers unchanged; validation / test keep their wording).

    .venv\\Scripts\\python scripts\\paraphrase_v4.py              # rule-based, runs anywhere (in-house)
    python scripts/paraphrase_llm_v4.py ...                        # optional: richer variants from a local open model;
                                                                   # writes data/v4/paraphrase_cache.json, used here if present

Every value is protected before rewording and checked afterwards: numbers, bridge / sheet / TBM ids, chainages, line
names ("3rd line"), abbreviations and printed labels in capitals, quoted text. A variant that changes any of them is
discarded. About 60 % of each training row's user turns get a variant; the original wording stays for the rest.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "v4" / "dataset"
CACHE = ROOT / "data" / "v4" / "paraphrase_cache.json"
FILES = ["train.jsonl", "reasoning_train.jsonl", "generic_train.jsonl", "aug_train.jsonl", "aug_reasoning_train.jsonl",
         "textblocks_train.jsonl", "agent_train.jsonl"]
SHARE = 0.6
OPEN, CLOSE = "⟦", "⟧"
rng = random.Random(61)

PROTECT = re.compile("|".join([
    r'"[^"]*"', r"'[^']*'",
    r"\b(?:3rd|4th) line\b",
    r"\b[A-Z][A-Z0-9.()&/'+-]*(?:\s+[A-Z0-9][A-Z0-9.()&/'+-]*)+\b",          # printed labels in capitals
    r"\b[A-Z](?:[A-Z0-9]|\.(?=[A-Z0-9]))+\.?(?:\s?\([a-z]+\))?(?![a-z])",      # abbreviations: HFL, MSL, N.G.L. (m), JSON
    r"\bMKN-\w+", r"\bROB(?: at CH [\d.]+|-\d+)\b", r"\bLC[- ]?\d+\w*", r"\bT?BMT?\s?\d+\w*",
    r"\b\d+[A-Z]*(?:UP|DN)?\b", r"-?\d+(?:\.\d+)?(?:\+\d+(?:\.\d+)?)?",
]))
SYNONYMS = [
    (r"\bWhat is the\b", ["What's the", "Tell me the", "What is the"]),
    (r"\bWhat is\b", ["What's", "What is"]),
    (r"\bchainage\b", ["CH", "ch.", "chainage"]),
    (r"\bbridge(?= \u27e6)", ["br.", "bridge no.", "bridge", "Br"]),        # only before an id
    (r"\bformation level\b", ["FL", "formation lvl", "formation level"]),
    (r"\bground level\b", ["GL", "NGL", "ground level"]),
    (r"\bminimum formation level\b", ["MIN FL", "min. FL", "minimum formation level"]),
    (r"\bdata[- ]bands?\b", ["bands", "L-section bands", "data band"]),
    (r"\bin this crop\b", ["in this image", "here", "in the picture", "in this crop"]),
    (r"\bthis crop\b", ["this image", "this picture", "this crop"]),
    (r"\breturn\b", ["give", "send", "return"]),
    (r"\bGive\b(?! me)", ["Provide", "Show me", "Give"]),
    (r"\bas JSON\b", ["in JSON", "as JSON please", "as JSON", "in json format"]),
    (r"^Read\b", ["Extract", "Transcribe", "Read"]),
    (r"\bexisting\b", ["exg.", "existing"]),
    (r"\bproposed\b", ["prop.", "proposed"]),
]
PREFIX_ASK = ["", "", "", "Quick one: ", "Hey, ", "Could you tell me: ", "Help me out - ", "Can you check: "]
PREFIX_DO = ["", "", "", "Please ", "Could you ", "Kindly ", "Quick one: ", "Hey, "]
PREFIX_ANY = ["", "", "", "Hey, ", "Quick one: "]
KEEP_CASE = re.compile(r"^(CH|GL|NGL|FL|MIN FL|L-section|Br|MSL|JSON)\b")
QUESTION = re.compile(r"\b(what|which|where|when|how|is|are|does|do|can|could|who|why)\b", re.I)
DO = re.compile(r"(read|give|find|locate|extract|detect|list|describe|explain|check|transcribe|provide|show|compare|"
                r"return|output|identify|count|draw|mark|work)\b(?![^?]*\?)", re.I)    # an instruction, not a question
SKIP = re.compile(r"\.(?:pdf|png|jpe?g|tiff?)\b|^The (?:3rd|4th) line\.$", re.I)   # file names / the "which line?" reply
DONE = DS / ".paraphrased"


def protect(text):
    keep = []

    def sub(m):
        keep.append(m.group(0))
        return f"{OPEN}{len(keep) - 1}{CLOSE}"
    return PROTECT.sub(sub, text), keep


def restore(text, keep):
    return re.sub(re.escape(OPEN) + r"(\d+)" + re.escape(CLOSE), lambda m: keep[int(m.group(1))], text)


def typo(text):
    words = [(m.start(), m.group()) for m in re.finditer(r"\b[a-z]{6,}\b", text)]
    if not words:
        return text
    pos, w = rng.choice(words)
    i = rng.randrange(1, len(w) - 2)
    return text[:pos] + w[:i] + w[i + 1] + w[i] + w[i + 2:] + text[pos + len(w):]


def reword(masked):
    low = rng.random() < 0.12                               # typed in lower case (protected values keep their case)
    out = masked.lower() if low else masked
    for pat, alts in SYNONYMS:
        if not re.search(pat, out, re.I) or rng.random() >= 0.55:
            continue
        alt = rng.choice(alts)

        def fit(m, alt=alt, s=out):                         # follow the case of the replaced spot
            if KEEP_CASE.match(alt):
                return alt
            at_start = not low and (m.start() == 0 or s[max(0, m.start() - 2):m.start()] in (". ", "? "))
            return alt[0].upper() + alt[1:] if at_start else alt[0].lower() + alt[1:]
        out = re.sub(pat, fit, out, count=1, flags=re.I)
    pre = rng.choice(PREFIX_ASK if QUESTION.match(out) else PREFIX_DO if DO.match(out) else PREFIX_ANY)
    if "tell me" in pre.lower() and "tell me" in out.lower():
        pre = ""
    if pre:
        out = (pre.lower() if low else pre) + out[0].lower() + out[1:]
    if rng.random() < 0.1:
        out = out.rstrip("?.")
    if rng.random() < 0.03:
        out = typo(out)
    return out


def vary(text, cache=None):
    masked, keep = protect(text)
    out = rng.choice(cache[masked]) if cache and masked in cache and rng.random() < 0.6 else reword(masked)
    if len(re.findall(re.escape(OPEN) + r"\d+" + re.escape(CLOSE), out)) != len(keep):
        return None
    new = restore(out, keep)
    if any(k not in new for k in keep):
        return None
    return new if new != text else None


def main():
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else None
    stats = Counter()
    done = set(DONE.read_text(encoding="utf-8").split()) if DONE.exists() else set()
    for f in FILES:
        p = DS / f
        if not p.exists() or f in done:                         # reworded already: a second run would reword twice
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        for r in rows:
            for m in r["messages"]:
                if m["role"] != "user" or rng.random() >= SHARE:
                    continue
                items = [m] if isinstance(m["content"], str) else [c for c in m["content"] if c.get("type") == "text"]
                for it in items:
                    key = "content" if it is m else "text"
                    if SKIP.search(it[key]):
                        continue
                    new = vary(it[key], cache)
                    if new:
                        it[key] = new
                        stats["varied"] += 1
                    else:
                        stats["kept"] += 1
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        done.add(f)
        DONE.write_text("\n".join(sorted(done)) + "\n", encoding="utf-8")
    print(dict(stats))


if __name__ == "__main__":
    main()
