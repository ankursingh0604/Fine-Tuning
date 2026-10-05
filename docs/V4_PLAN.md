# v4 plan: an L-section assistant that works like an engineer's assistant

Decided: **fully in-house (option A)**. No drawings, extracted values or project-specific questions go to outside
services; the only outside access is a controlled internet fallback for general knowledge (automatic, through a code-enforced leak gate; see "Knowledge beyond the drawing"). One fine-tuned
open model is both the brain (conversation, planning, tool calls) and the eyes (reading drawings), with local tools.

## Model — decided: accuracy over speed

- **Main model: Qwen3.5-27B (dense).** All 27B parameters work on every token, so it is stronger at reading small
  digits and at reasoning than the 35B-A3B mixture-of-experts model (which uses only ~3B parameters per token and was
  the choice when speed mattered). Unsloth supports Qwen3.5 vision fine-tuning; the 27B scores 89.4 on OCRBench.
- **No 9B pilot (decided: budget).** The risks a pilot would catch are caught instead by free CPU checks and a smoke
  test at the start of the main run (see "One training run").
- **Step up only if the 27B falls short on the test set:** Qwen3.5-122B-A10B (larger mixture of experts, ~10B active
  per token). Much heavier to train and run (several GPUs), so only with evidence that it is needed.
- Training: 16-bit LoRA (Unsloth advises against 4-bit QLoRA training for Qwen3.5), 2-3 epochs, not more; best
  checkpoint on validation. 27B 16-bit LoRA needs about 56 GB plus our image crops: **one 80 GB GPU (A100/H100)**.
- Running it: **16-bit (about 54 GB) on an 80 GB GPU, or 8-bit (about 28-30 GB) on a 48 GB GPU** — not 4-bit, which
  costs accuracy. The RTX 3060 (12 GB) cannot run it.
- **Training on RunPod — decided, acceptable.** One main run (27B) on one 96 GB RTX PRO 6000, roughly 1.5-3 days and
  $100-250 for up to 3 epochs (estimate; depends on the final dataset size); usually less with early stopping. Pods are private and deleted after training; the adapter is downloaded and kept in-house.
- Open hardware question: which in-house GPU will run it (80 GB for 16-bit, or 48 GB for 8-bit)?

## Frameworks — decided

**Training: Unsloth, 16-bit LoRA (not QLoRA).**

- Not QLoRA: it trains against a 4-bit copy of the model (accuracy loss), Unsloth advises against 4-bit training for
  Qwen3.5, and the 96 GB GPU makes it unnecessary. v3 also used 16-bit LoRA.
- Not full fine-tuning: needs several GPUs; LoRA gets close for a narrow domain at a fraction of the cost.
- Unsloth: supports Qwen3.5 vision fine-tuning, fastest on one GPU, and v3's script, data format and
  response-only training carry over. LLaMA-Factory / ms-swift only if multi-GPU training is ever needed (122B).
- LoRA settings (no pilot to compare, so one choice): vision + language layers (as v3); **r = 32, alpha = 32** — v4
  teaches much more than v3 (tools, situations, generic tables), the extra memory is small, and checkpoint selection
  on validation guards against overfitting. Up to 3 epochs, best checkpoint on validation, checkpoints on a RunPod
  network volume.
- RunPod GPU: **RTX PRO 6000, 96 GB ($2.09/hr)** — headroom
  over 80 GB cards, good availability (needs recent PyTorch cu128 builds, as already used). Top up the balance
  before the main run.

| Part | Tool |
|---|---|
| Fine-tuning | Unsloth (on Hugging Face TRL + PEFT); `transformers` v5 (required by Qwen3.5) |
| Serving in-house | vLLM: base + LoRA adapter, or merged into one 16-bit model; 8-bit (FP8) only on a 48 GB GPU |
| Agent loop and tools | Own small loop using Qwen3.5's built-in tool-calling format (no LangChain etc.) |
| Store | SQLite + local text-search index (reference library, sheet text) |
| OCR cross-check | RapidOCR (PP-OCRv4), CPU |
| Exact PDF reading | PyMuPDF (`annotate.py`) |
| Evaluation | Own test-set scripts, as for v3 |

## One training run (decided: no pilot, limited budget)

A pilot would catch broken data, a wrong chat template, out-of-memory errors and a model that does not learn. Those
are caught here instead, mostly for free:

1. **Free checks on CPU before renting anything:** every dataset row validated (images open, sizes, JSON answers parse,
   tool calls well-formed, thinking traces only where the reasoning rules say); the Qwen3.5 chat template applied with
   the tokenizer (CPU) to every row; token lengths measured so the sequence limit and batch size are set from real
   numbers; a sample of rows rendered for a visual check.
2. **Smoke test at the start of the main run (~30-60 min):** a few hundred steps, then automatic checks — loss going
   down, no out-of-memory, a handful of validation questions answered in the right format (JSON, tool calls,
   thinking switch). If anything fails, the pod is stopped after under an hour instead of after days. If it passes,
   training simply **continues from that checkpoint** — no time wasted.
3. **Validation during training:** scored every N steps on held-back validation rows; the best checkpoint is kept.
4. **Early stopping and a step cap:** training stops when validation stops improving (often before epoch 3), and never
   exceeds the planned number of steps, so the cost cannot run away.
5. **Checkpoints on a network volume and resume:** an interrupted pod resumes from the last checkpoint instead of
   starting over.
6. Balance topped up for the estimated cost before starting; the adapter is downloaded and the pod deleted at the end.

**GPU utilisation (v3 used only ~36% of the L40S).** Likely causes in v3's `train.py`: images loaded, resized and
patched in the main process every step (no data-loader workers), a small 3B model finishing each step quickly and
then waiting, and generation-based evaluations (before/after comparisons, checks every 150 steps) that use the GPU
lightly. Idle GPU time is paid time, so for v4:

- images pre-processed once to their final size before training;
- 4-8 data-loader workers with pinned memory (the RTX PRO 6000 pod has 16 vCPU);
- batch size set by measurement in the smoke test (raise until ~85% GPU memory), not a fixed rule;
- examples grouped by length to cut padding;
- evaluation during training kept light (validation loss + a small format check); the full generation-based
  evaluation runs afterwards in-house with vLLM, not on the rented GPU;
- the smoke test records GPU utilisation and memory and must reach about 80% utilisation before the run continues.
- The 27B model itself does ~9x more work per step than the 3B, so starvation is less likely to begin with.

## Accuracy mode (time traded for correctness)

Accuracy comes mainly from **reading each value more than once and checking it**, not from longer thinking on
transcription. The reader therefore:

1. **Reads every value twice** from different crops (tiles and band windows overlap so each callout, level block
   and band column appears in at least two crops); the two readings must agree.
2. **Cross-checks digits with the independent OCR** (RapidOCR / PP-OCRv4) on band values, level blocks and callout
   numbers.
3. **Re-reads automatically on any disagreement or failed check** (band arithmetic, level block vs band FL): a third
   read with a shifted / tighter crop and at 200 dpi; the reading that agrees with the others and passes the checks
   wins. If none does, the value is kept with a clear "doubtful — check on the drawing" flag, never silently.
4. **Trains and reads at two scales** (150 and 200 dpi crops), using 200 dpi when the source allows it.
5. **Thinks** on checks, judgements, multi-step questions and doubtful readings (see "Reasoning rules").
6. Vector PDFs need none of this: they are read exactly from the text layer.

Expected time per scanned sheet in accuracy mode: roughly 5-15 minutes (to be measured); vector PDFs: seconds.

## Architecture: brain + tools in a loop

| Tool | What it does |
|---|---|
| `read_sheet(image or pdf)` | Reads a whole sheet into the store (exact for vector PDFs; the model on tiles for images) |
| `look(sheet, area, question)` | Zooms into any part of a sheet and asks the vision model about it |
| `query(...)` | Searches the store of every sheet read (bridges, bands, curves, gradients, TBMs, notes, ...) |
| `band_at(chainage)` | Nearest band column, never interpolated |
| `calc(expression)` | Exact arithmetic (the model never does arithmetic in its head) |
| `check(rule, ...)` | FL >= MIN FL, ruling gradient, track centres >= 4.725 m, free board, band arithmetic, curve formulas |
| `search_library(query)` | Searches the local reference library (codes, manuals, standards, abbreviation lists) |
| `web_search(term)` | Automatic internet lookup of a new term's general meaning; the query is built and leak-checked by code, never by the model; answer labelled |

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

## L-section layouts the model was not trained on

Without training, any layout already gives: every printed word with its position (whole-sheet text index), the
tables found from their rules, labels and column spacing (`layout.py`), and `look` on any spot. What does not carry
over is **meaning and structure** (which number belongs to which bridge or row, what a new callout style means), and
rows the model has never seen (e.g. "BANK HEIGHT") are skipped today. v4 closes these gaps:

1. **Generic table reading (new task).** "Read this table: for each row give its printed label and the value in every
   column", answered as `{label: {chainage: value}}`, not only the 7 trained fields. Trained on deliberately varied
   layouts generated from our sheets: rows re-ordered, dropped and duplicated; labels renamed or reworded (e.g. "GROUND
   LEVEL" / "NGL" / "EXISTING GROUND"); extra rows with new labels; different row heights and column spacing; columns
   at 10/20/25/50 m; tables moved on the sheet. Unknown rows are then stored by their printed label and are
   answerable ("bank height at 1242+660") instead of skipped.
2. **Generic text-block reading (new task).** Group nearby text into blocks (callouts, level blocks, boxes) and read
   each block as printed with its position, even when the model does not know the style. Trained on our callouts
   re-drawn in varied styles (field order, separators, box shapes, abbreviations). Known fields are mapped when
   recognisable (bridge number, span, levels); everything else is kept as printed text.
3. **Label and heading reading** (above) plus label-to-field mapping by meaning, so "BANK HEIGHT", "FORMATION WIDTH" or
   "PROP. DN LINE FL" become named fields or stay as their own labelled rows.
4. **Store and answers by label.** The store keeps both the trained fields and generic `label -> values` tables and text
   blocks, each with its sheet position. Answers cite where on the sheet the value was found.
5. **Situation awareness.** On a new layout the reader says so; answers from generic tables/blocks are marked
   "new layout: check"; checks that still apply (FL - GL = cut/fill when those rows exist) are run on the identified rows.
6. **Becoming reliable on a new layout:** 5-10 annotated sheets of that layout in the next training round.
7. **Test set:** held-out layouts the model never saw (synthetic variants and any real new-layout sheets), scored
   separately: rows identified, values read, blocks read, correct "new layout" warnings.

## Knowledge beyond the drawing (local library first, then automatic internet lookup through a leak gate)

Some questions need knowledge that is not on any sheet ("what does CTP mean?", "minimum track centre per IRS?",
"free board required by the code?").

1. **Local reference library (first).** Documents supplied by the team (IRS codes and manuals, RDSO standards,
   Schedule of Dimensions, organisation abbreviation lists, specifications) are indexed in the store and searched
   locally with the new tool `search_library(query)`; answers cite document and section.
2. **Internet — automatic, no approval (decided), leak-proof by construction.** When something new is found (an
   unknown term, abbreviation or label) or a question needs general knowledge the sheet and library do not have, the
   assistant searches the internet by itself. Because no person checks each search, **the protection is enforced by
   code, not left to the model:**
   - **The model never writes the search text.** It can only hand over a *term* (e.g. "SFL"). Code builds the query
     from a fixed template: the term plus words from a fixed allowlist ("railway", "longitudinal section", "civil
     engineering", "abbreviation", "meaning") — e.g. `"SFL" railway longitudinal section abbreviation meaning`.
   - **The term itself must pass a leak gate (code) before anything leaves:** short (at most 4 words / 30 characters);
     no numbers that look like chainages, levels, coordinates, dates, bridge / sheet / drawing numbers; and not on the
     **denylist built from the store** — every identifying string on the sheets: drawing and sheet numbers, project
     and client names, station names and codes, place names, people's names (issue record, officers), TBM
     descriptions. A term that fails is **not searched**; the answer says the meaning could not be looked up safely.
   - **Only the query text goes out:** no images, no sheet text, no extracted values, no file names, no cookies or
     account; plain search requests through one module, results fetched read-only.
   - **Each term is searched once** and the result cached locally; later questions reuse the cache.
   - Every search is logged (term, query, time, source used); an administrator can switch internet search off.
   - Residual risk, stated honestly: a printed term can itself hint at the project (e.g. an unusual local
     abbreviation). The denylist removes the identifying strings known from the sheets; anything else that is short,
     general and number-free is treated as safe to search.
3. **Never sent out:** drawings, extracted values, sheet text, or questions containing project details.
   **Lookup order for anything new or unclear** (Ankur's rule): (1) the sheet's own right-hand panel — notes, legend,
   abbreviations; (2) the local library; (3) the internet, automatically, through the leak gate; (4) if still
   unclear, ask the user. On a new layout, unfamiliar terms found while reading are looked up the same way, after the
   panel and library, without interrupting the reading.
   A web result only explains a term's meaning; it never changes a value read from the drawing or which row a value
   belongs to — if the meaning stays uncertain, the row keeps its printed label.
4. **Answers say where each part comes from:** the drawing (sheet, chainage, column), the library (document, section)
   or the internet (source) — never blended without attribution. Values always come from the drawing; general rules
   from the library or the web.
5. **Training:** conversations where the model picks the right source (store / `look` / library / web), hands
   `web_search` only a bare general term (never sentences or project details), and labels the answer.
   **Test set:** knowledge questions scored for correct source and labelling; **leak test** of the gate itself — every
   identifying string from the test sheets (and synthetic ones mixed into terms) must be blocked, with zero project
   details in any query that leaves.

## Multi-sheet PDFs

Real uploads are whole PDFs (e.g. `MKN_PNP_1210-1251_3rd Line.pdf`: 9 pages, sheets 94-102, each continuing the
previous sheet's chainage). Today `read_sheet.py` and the app take one image of one sheet; `annotate.py` (used by
the on-hold PDF checker) already reads every page of a vector PDF exactly.

1. **Split and recognise each page** (situation awareness): L-section sheet / cover or index / GAD / other. Other
   pages are reported, never forced into L-section fields. Each L-section page gets its layout verdict.
2. **Route per page:** vector page (selectable text; all 17 pages of the current PDFs) -> `annotate.py`, exact,
   seconds, no model. Scanned page -> render at 150-200 dpi -> whole-sheet reader + model. Mixed PDFs are handled page
   by page.
3. **One store**, each sheet filed by sheet number and chainage range from its title block.
4. **Checks across sheets:**
   - continuity: each sheet's end chainage = the next sheet's start; previous/next sheet numbers in the title blocks
     agree; gaps, overlaps and missing sheets reported;
   - bridges on a sheet boundary (drawn on both sheets) merged into one, keeping the complete callout
     (`belongs_to` / `complete` from the annotator);
   - values that should match across a boundary (e.g. last band column of one sheet vs first of the next) checked.
5. **Questions across the whole PDF**, every answer citing its sheet: "which sheet covers CH 1242662.9?", "all
   bridges below MIN FL in this PDF", "highest fill between 1215+000 and 1240+000" across several sheets.
6. **Time and progress:** vector PDF, all sheets in seconds; scanned, roughly 5-15 minutes per sheet in accuracy mode
   (about 1-2 hours for 9 scanned sheets; to be measured). Progress page by page; each sheet can be questioned as soon
   as it is done.

**Versions — decided: a sheet number already in the store is kept as a separate version, never replaced.**

- A version is identified by sheet number + revision from the issue record (R0, R1, ...) + upload (file name, date,
  content hash).
- The **identical file uploaded again** (same content hash) reuses the stored reading; it is not a new version.
- Answers use the **latest revision** by default (by issue-record revision; if revisions are equal or missing, the
  latest upload) and say which version they used.
- If another stored version of that sheet **differs** on the value asked about, the answer says so
  (e.g. "sheet 100 R1: FL 180.724; R0 had 180.700").
- Users can ask for a specific version ("sheet 100 R0"), list versions ("which versions of sheet 100 do we have?")
  and compare them ("what changed between R0 and R1 of sheet 100?" -> differences in bridges, levels, band values,
  curves, notes).
- Nothing is deleted or overwritten by an upload.

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

The model decides **when** to reason, just as it decides whether to answer, ask back or warn. Even with accuracy
over speed, thinking on pure transcription does not make a digit more likely to be read correctly — re-reading and
cross-checking does (see "Accuracy mode"). Thinking is used where it adds accuracy: combining, checking, deciding.

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

0. **Data and decisions**: new L-section PDFs (text layer, layout, annotation); held-back test sheets; the in-house
   inference GPU (training on RunPod is decided).
1. **Store and tools** (code only): SQLite store of everything on every sheet, with sheet versions (see "Multi-sheet
   PDFs"); multi-page PDF intake (page split, page type, vector/scanned routing, cross-sheet checks); the six tools;
   tested on vector PDFs.
2. **Whole-sheet reader, everything on the sheet** (see "Full coverage"): layer 1 full sweep (all band columns, curves,
   transition points, gradients, grade points, km posts, notes, legend, abbreviations, panel sections) and layer 2
   long tail (whole-sheet text index + `look`). Connect the vector-PDF route (`annotate.py`) to the same store.
   Layout detection is done (layout.py).
3. **Dataset v4**
   - Vision: everything on the sheet, in Qwen3.5's image and box format, at 150 and 200 dpi; low-DPI copies
     (rendered at 60-120 dpi and scaled up).
   - **Label and heading reading (new):** "read the row labels of this band strip" and "read the headings of this
     panel", generated from the annotations (every label's text is stored). The reader then asks the model for the
     labels and compares with the OCR (RapidOCR / PP-OCRv4): agreement confirms the layout, disagreement is flagged.
     OCR stays as the independent check.
   - Reasoning: existing chain-of-thought families, nearest column, MIN FL flag, plus thinking traces only where the
     reasoning rules say so (see "Reasoning rules").
   - Unseen layouts: generic table reading and generic text-block reading on layout variants generated from our
     sheets (see "L-section layouts the model was not trained on").
   - Knowledge sources: choosing store / `look` / library / web, bare general terms for `web_search`, labelling.
   - **Agent behaviour:** code-generated conversations (question -> tool calls -> real tool results -> reasoning ->
     answer), covering every situation in the tables above, including multi-turn, clarifying, "not on the sheet",
     failed checks, unknown layouts and non-L-section inputs.
   - Varied wording from a **local** open model (answers always our checked ones).
4. **Test set**: about 300 questions with exact answers from held-back sheets, scored per category and per situation:
   correct answers, correct sources, correct clarifications, correct "not on the sheet", false-confidence rate
   (confident answers that are wrong), layout warnings raised when they should be.
5. **Training — one run** (see "One training run"): free CPU checks, then the **27B** on the RTX PRO 6000 with a smoke
   test at the start, validation during training, early stopping, checkpoints and resume. 122B-A10B only if the 27B
   falls short on the test set (a separate budget decision).
6. **Serving**: vLLM in-house, 16-bit (80 GB GPU) or 8-bit (48 GB GPU), accuracy mode on (double reads, OCR
   cross-check, re-read on disagreement); batching keeps the extra reads affordable. Chat in CLI and app.
7. **Evaluate and improve**: v4 vs v3 on the test set; fix weak areas with data, not more epochs.

## Targets (aims, to be measured)

| Capability | Target |
|---|---|
| Values read from unseen sheets at 150-300 dpi | >= 98 % fields correct |
| Values at 75-100 dpi | >= 90 % (after low-DPI training) |
| Questions in any wording, incl. multi-step and follow-ups | >= 90 % correct on the test set |
| Situation handling (clarify / not on sheet / warn / refuse to guess) | each situation scored; false-confidence rate as low as possible |
| Doubtful values | every value that failed its checks or disagreed between reads is flagged — none passed silently |
| Data | never leaves own systems |

## What it will not do

Match a frontier model's general reasoning; know anything not in the drawings or documents given to it; make
engineering decisions (it flags and explains); read layouts or hand-drawn sheets it has no training examples of;
replace review (values marked CHECK and low-DPI sheets still need a person).
