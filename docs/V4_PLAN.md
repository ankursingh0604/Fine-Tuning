# v4 plan: an L-section assistant that works like an engineer's assistant

Decided: **fully in-house (option A)**. No drawings, values or questions go to outside AI services. One fine-tuned
open model is both the brain (conversation, planning, tool calls) and the eyes (reading drawings), with local tools.

## Model

- **Qwen3.5-35B-A3B** (mixture of experts: ~35B parameters, ~3B active per token, so large-model accuracy at
  small-model speed). Unsloth supports Qwen3.5 vision fine-tuning.
- **Pilot first: Qwen3.5-9B** to prove the data and pipeline cheaply (also runs on the RTX 3060 in 4-bit).
- 16-bit LoRA (Unsloth advises against 4-bit QLoRA training for Qwen3.5); 2-3 epochs, not more.
- Open hardware questions:
  1. Is a rented RunPod GPU acceptable for training under the in-house rule, or must training run on own hardware
     (80 GB A100/H100 for the 35B-A3B; 48 GB for the 9B)?
  2. Inference for the 35B-A3B needs a GPU with 24 GB or more (48 GB comfortable); the RTX 3060 (12 GB) can only run the 9B.

## Architecture: brain + tools in a loop

| Tool | What it does |
|---|---|
| `read_sheet(image or pdf)` | Reads a whole sheet into the store (exact for vector PDFs; the model on tiles for images) |
| `look(sheet, area, question)` | Zooms into any part of a sheet and asks the vision model about it |
| `query(...)` | Searches the store of every sheet read (bridges, bands, curves, gradients, TBMs, notes, ...) |
| `band_at(chainage)` | Nearest band column, never interpolated |
| `calc(expression)` | Exact arithmetic (the model never does arithmetic in its head) |
| `check(rule, ...)` | FL >= MIN FL, ruling gradient, track centres >= 4.725 m, free board, band arithmetic, curve formulas |

Memory: the store keeps every sheet read (questions across the whole line); the conversation keeps follow-ups
("its HFL?", "and 561?").

## Full coverage: everything on an uploaded sheet

Goal: any piece of information printed on an uploaded sheet can be answered. Two layers, because "everything"
splits into what can be listed in advance and a long tail that cannot.

**Today (v3 + whole-sheet reader, from an image):** bridge callouts, level blocks, title block, TBMs, and band values
only at bridge chainages (plus any chainage asked about). Curves, transition points, gradients, grade points, notes,
legend, abbreviations, km posts, stations, reference drawings, issue record and all other text are **not** extracted
from images yet. From a **vector PDF**, `annotate.py` already extracts all of these exactly (including `all_text`),
but that route is not connected to question answering yet.

**Layer 1 — full sweep into the store** (everything with a known structure), read once per sheet:

| Item | How |
|---|---|
| All data-band columns (~250 per sheet, not just at bridges) | ~16 band-window questions per sheet (each crop covers 16 columns) |
| Bridges: callouts and level blocks | as today |
| Curves (plan and L-section boxes), transition points ST/TTP1, TC/CTP1, CT/CTP2, TS/TTP2 | tiles + trained curve tasks |
| Gradients, grade points, vertical intersection points, km posts | tiles + trained tasks |
| Notes, legend, abbreviations | right-panel sections (found by headings) |
| Title block, TBM table, issue record, reference drawings, stations, officers | right-panel sections |
| Drawing checks | band arithmetic, FL vs MIN FL, curve formulas, plan vs L-section |

For vector PDFs the same store is filled exactly from `annotate.py` (no model needed).

**Layer 2 — the long tail** (anything else printed):

- **Whole-sheet text index:** every word on the sheet with its position, from the local OCR (RapidOCR / PP-OCRv4)
  for images, or the PDF text layer for vector PDFs. Answers "where does it say ...?" and "what is written near
  CH ...?".
- **`look`:** the model zooms into that spot and reads or interprets it, so any small label, remark or symbol can be
  answered on request even if it was not extracted in advance.

**Limits (stated in answers when they apply):**

- From images not every value will be perfect: tiny, overlapping or low-DPI text can be misread. Checks catch many
  misreads, but ordinary text (a note's wording, a station name) has nothing to check it against. Vector PDFs are exact.
- Graphics are not data: the model can say a curve is there and read its box, but the drawn shape of the ground line
  between columns is not printed data, and the nearest-column rule means no interpolation anyway.
- It knows only the sheets and documents given to it; reference drawings mentioned on a sheet are not known unless
  uploaded too.

**Test set:** includes long-tail questions (notes, legend items, small labels, remarks) scored separately from the
structured items, so coverage is measured, not assumed.

## Situation awareness

The model must recognise **what it has been given and what is being asked**, and choose its behaviour —
not just answer. Every situation below gets its own training examples (including the negative ones) and its own
score in the test set.

**The input**

| Situation | Expected behaviour |
|---|---|
| Whole sheet | `read_sheet`, then answer from the store |
| Crop of a sheet | Read it directly; say what part of the sheet it seems to be |
| Low DPI (< 100) or blurred | Read, but carry a warning into answers; flag doubtful digits |
| Layout known / similar / unknown (layout.py verdict) | Known: normal. Similar: answer and say to check. Unknown: answer only what was read reliably; list what was skipped |
| Not an L-section (GAD, schematic, photo, other document) | Say what it appears to be; do not force L-section fields onto it |
| Schematic with no values | Say no values are printed; never invent numbers |
| Same sheet uploaded again | Reuse the stored reading |

**The request**

| Situation | Expected behaviour |
|---|---|
| Simple lookup | Tool call, short answer with source (sheet, chainage, column) |
| Multi-step ("bridges between A and B with fill > 2 m") | Plan, several tool calls, `calc`, answer with working |
| Follow-up ("its HFL?", "and 561?") | Resolve from the conversation |
| Ambiguous ("bridge 56" when 556 and 565 exist; "the FL" with no bridge) | Ask back, offering the candidates |
| Report ("bridge schedule", "all flags") | Table / CSV |
| Chainage outside this sheet | Say which sheet covers it (from the store), or that it has not been read |
| Out of scope (design decisions, codes not provided, unrelated topics) | Say so plainly; do not improvise engineering decisions |

**The evidence**

| Situation | Expected behaviour |
|---|---|
| Value printed and checks pass | Answer |
| Value not printed on the sheet | "Not on the sheet" (not a guess, not interpolated) |
| Value read but a check fails (band arithmetic, level block vs band) | Give the value, say it failed the check and is probably misread |
| Between band columns | Nearest column only, with the distance; never interpolate |
| FL below MIN FL | "FLAG: For bridge X min. FL = .., and FL = .." |
| Two sources disagree (plan vs L-section, drawing inconsistencies) | Report both and the disagreement |

## Reasoning rules (part of situation awareness)

The model decides **when** to reason, just as it decides whether to answer, ask back or warn. Reasoning on every
question would make simple answers and sheet reading several times slower without making them more accurate.

| Question type | Reasoning | Example |
|---|---|---|
| Single lookup | **No** — one tool call, answer with source | "FL of bridge 558?" |
| Reading / transcription (the ~100 reading questions per sheet, labels, headings) | **No** | callout JSON, band column JSON |
| Combines several values into a judgement | **Yes** | "Is bridge 532 safe against flood?" (FL, MIN FL, HFL, free board -> conclusion) |
| Multi-step / filtering across sheets | **Yes** — plan, tool calls, filter, answer | "Bridges between 1240+000 and 1260+000 with fill > 2 m" |
| Checks and consistency | **Yes** | "Does this band column add up?", plan vs L-section curve boxes |
| Ambiguous or incomplete request | **Briefly** — notice the ambiguity, then ask back | "The FL?" with no bridge named |
| Failed check or doubtful reading | **Briefly** — explain why the value is doubtful | band arithmetic does not add up |
| Out of scope | **No** — say so plainly | design decisions, codes not provided |

Rules:

1. Think (Qwen3.5 thinking mode, `<think> ... </think>`) only for the "Yes" and "Briefly" rows; answer directly otherwise.
2. **Calculations always show their working in the final answer**, even without long reasoning
   (e.g. "Free board = FL - HFL = 193.059 - 192.370 = 0.689 m"). The arithmetic itself is done by `calc`, never in the
   model's head.
3. Reasoning uses only values from tools (the store, `look`, `band_at`) — never remembered or invented values — and
   cites where each came from (sheet, chainage, column).
4. Nearest band column only, never interpolation; FL below MIN FL always produces the flag sentence.
5. If the reasoning finds a gap or a conflict (value not printed, check fails, sources disagree), the answer says so
   instead of smoothing it over.

Training and scoring: thinking traces are generated by code for the "Yes"/"Briefly" situations and omitted for the
"No" ones, so the model learns the switch. The test set scores both directions — **reasoned when it should** and
**did not reason when it should not** (as v3's think-flag metric did) — plus whether the shown working is correct.

Note on v3 (current model): it reasons only on its 11 trained reasoning families (crop questions in `ask.py` / the
app's crop tab). The whole-sheet CLI answers (`read_sheet.py -q`) come from code (`sheet_qa.py`) over what the model
read, not from model reasoning.

## Phases

0. **Data and decisions**: new L-section PDFs (text layer, layout, annotation); held-back test sheets; hardware answers.
1. **Store and tools** (code only): SQLite store of everything on every sheet; the six tools; tested on vector PDFs.
2. **Whole-sheet reader, everything on the sheet** (see "Full coverage"): layer 1 full sweep (all band columns, curves,
   transition points, gradients, grade points, km posts, notes, legend, abbreviations, panel sections) and layer 2
   long tail (whole-sheet text index + `look`). Connect the vector-PDF route (`annotate.py`) to the same store.
   Layout detection is done (layout.py).
3. **Dataset v4**
   - Vision: everything on the sheet, in Qwen3.5's image and box format; low-DPI copies (rendered at 60-120 dpi and
     scaled up).
   - **Label and heading reading (new):** "read the row labels of this band strip" and "read the headings of this
     panel", generated from the annotations (every label's text is stored). The reader then asks the model for the
     labels and compares with the OCR (RapidOCR / PP-OCRv4): agreement confirms the layout, disagreement is flagged.
     OCR stays as the independent check.
   - Reasoning: existing chain-of-thought families, nearest column, MIN FL flag, plus thinking traces only where the
     reasoning rules say so (see "Reasoning rules").
   - **Agent behaviour:** code-generated conversations (question -> tool calls -> real tool results -> reasoning ->
     answer), covering every situation in the tables above, including multi-turn, clarifying, "not on the sheet",
     failed checks, unknown layouts and non-L-section inputs.
   - Varied wording from a **local** open model (answers always our checked ones).
4. **Test set**: about 300 questions with exact answers from held-back sheets, scored per category and per situation:
   correct answers, correct sources, correct clarifications, correct "not on the sheet", false-confidence rate
   (confident answers that are wrong), layout warnings raised when they should be.
5. **Training**: 9B pilot (2-3 epochs), then 35B-A3B (2-3 epochs, checkpoints, resume, best on validation).
6. **Serving**: vLLM in-house, batched questions (target: whole-sheet read in about 1-2 minutes); chat in CLI and app.
7. **Evaluate and improve**: v4 vs v3 on the test set; fix weak areas with data, not more epochs.

## Targets (aims, to be measured)

| Capability | Target |
|---|---|
| Values read from unseen sheets at 150-300 dpi | >= 98 % fields correct |
| Values at 75-100 dpi | >= 90 % (after low-DPI training) |
| Questions in any wording, incl. multi-step and follow-ups | >= 90 % correct on the test set |
| Situation handling (clarify / not on sheet / warn / refuse to guess) | each situation scored; false-confidence rate as low as possible |
| Data | never leaves own systems |

## What it will not do

Match a frontier model's general reasoning; know anything not in the drawings or documents given to it; make
engineering decisions (it flags and explains); read layouts or hand-drawn sheets it has no training examples of;
replace review (values marked CHECK and low-DPI sheets still need a person).
