# GAD (General Arrangement Drawing) plan

Status: **Phase 1 (reading and questions) started for RCC box GADs** (2026-10-05). DXF output and parametric editing
(Phases 2-4) stay on hold. The L-section work is untouched by this.

### Phase 1 as built (RCC box, 67 vector-PDF GADs of the Itarsi-Nagpur 3rd line)

- `gad_tools/annotate_gad.py` reads a GAD exactly from its text layer: views (found from the drawing's own line work;
  rotated sheets and rotated views handled), every labelled level with its meaning, labelled dimensions (T/C track
  centres, barrel length, thicknesses, weep holes, slopes), unlabelled dimension figures (view and direction only),
  comparative table / hydraulic data, track details, depth of track structure, bore logs, all notes, specifications,
  design criteria, reference drawings, abbreviations, title block, and **disagreements** between places on the same
  drawing (e.g. box size in the title vs the table). The signature block (names) is not read.
  `gad_tools/overlay_gad.py` draws what was found on the sheet for checking.
- `gad_tools/gad_kinds.py`: what each view, level, table row and labelled dimension means. **For the engineers to
  review** - every "what does X represent / denote" answer comes from it.
- `gad_tools/facts.py` picks the facts relevant to a question; `gad_tools/build_dataset_gad.py` builds question/answer
  rows with those facts as context plus image rows (whole views, 150 dpi tiles, tables).
- `gad_kit/`: training on the RTX 3060 (Qwen3.5-4B, 16-bit LoRA, 2 epochs), automatic scoring, and `ask_gad.py`
  (questions about any GAD PDF). One GAD (810-1) is held back for the manual test.
- Decided: unlabelled dimension figures are given with their view and direction only (the drawings say "do not
  scale", and measured distances do not match the printed ones, so their meaning cannot be derived reliably).

## Goal

Someone uploads a bridge GAD in any form (DWG/DXF, vector PDF, image or scan, hand-drawn sketch) and can then:

1. **Ask questions about it**, the same way the L-section sheets can be queried.
2. **Get a parameterized DXF** of it when they ask for one, and change a parameter (e.g. wing wall length) to see how the GAD changes.

The model must decide by itself when to return a DXF and when to give a normal answer, and ask when the request is ambiguous.

## Core principle: the model reads, code draws

The model never writes DXF. A DXF is thousands of exact coordinates, and a language model would invent geometry.

1. The fine-tuned model reads the GAD and outputs a **parameter set** (JSON): bridge family, spans, cells, slab/wall thicknesses, haunch, cushion, wing walls, skew, bed level, FL, HFL, foundation, reinforcement, ...
2. Code checks it (e.g. bed level + bottom slab + clear height + top slab + cushion = FL; values in sensible ranges).
3. A deterministic **template per bridge family** (Python, ezdxf) computes every coordinate from the parameters and draws the DXF.

The same parameter set answers questions about the GAD, so querying and DXF output share one extraction.

## Outputs per GAD

| File | What it is |
|---|---|
| `GAD_<id>.dxf` | Clean, layered drawing that opens in any CAD program; parameters embedded |
| `GAD_<id>_params.json` | The parameters; edit and regenerate |
| `GAD_<id>_build.lsp`, or a constrained DWG/DXF | AutoCAD builds the drawing with live constraints and dynamic blocks (see below) |
| Check report + overlay image | Generated drawing laid over the uploaded GAD; mismatches flagged |

"Parameterized" was agreed to mean **both**:

- **(a) Parameter file → regenerate.** Pure Python, works anywhere.
- **(b) AutoCAD constraints and dynamic blocks inside the file.** No open-source library writes these, so AutoCAD (or BricsCAD) has to build them. We generate an AutoLISP script that inserts dynamic blocks, adds geometric constraints, named dimensional constraints (`clear_span`, `wall`, `top_slab`, ...) and formulas in the Parameters Manager. It is run either by the engineer in their own AutoCAD, or headless on a licensed Windows machine (accoreconsole or BricsCAD).
  - A dynamic block library (box cell, pier, abutment, wing wall, girder, bearing, pile cap, railing) is built once by hand in AutoCAD, preferably by a CAD person who knows the drawing standards.
  - AutoCAD LT cannot author dynamic blocks or constraints, which would limit (b).
  - Library check (re-verified): ezdxf only preserves dynamic blocks and constraints (they are undocumented in the DXF reference; copying such blocks can drop them); LibreDWG lists the constraint objects as unhandled; ACadSharp has no constraint support. No open-source library can author them.
  - **Excluded by the decision that project data stays in-house** (drawings would go to Autodesk's cloud); kept here only for reference. Use the engineers' own AutoCAD or an in-house licensed copy instead.
  - Server option without our own AutoCAD licence: Autodesk Platform Services (APS) Design Automation for AutoCAD — real AutoCAD running headless in Autodesk's cloud, which runs our AutoLISP / scripts / .NET plug-ins and returns the DWG/DXF. Billed per engine time (4 cloud credits per hour for AutoCAD; Autodesk's own example put a ~1-minute AutoCAD job at about $0.07 — re-check current pricing). Drawings are sent to Autodesk's cloud, so confirm that is acceptable for the project data.
  - Commercial alternative: ODA Drawings SDK (C++, paid membership) supports dynamic blocks and geometric/dimensional constraints; only worth it if we build our own CAD product.

## Interactive "what if" editing

1. Upload. The model reads the GAD and recognises the family.
2. Show the original and the regenerated drawing side by side, overlaid, so the user confirms the baseline. Mismatches are highlighted ("wing wall: drawing 3.5, read 3.2 — confirm").
3. A parameter panel lists every editable value. Changing one redraws in about a second: previous version in grey, changed parts in red, every view updated, plus a list of what moved.
4. The same change can be asked in chat ("make the wing wall 4.5 m"); the model turns it into a tool call.
5. Inputs and dependent values: dependents update by engineering rules (e.g. wing wall length from embankment height and side slope); a manual override of a dependent is kept and marked.
6. Engineering checks run on every change (wing wall too short for the slope, clear height below HFL + free board, FL below MIN FL, ...), with limits from the engineers and the codes.
7. Download the plain DXF, the parameter JSON, and the AutoCAD script or constrained DWG at any point.

Redrawing needs no model call: it only re-runs the template.

## What "exact" means for each input form

| Input | Exact visual copy | Dimensionally exact parameterized redraw |
|---|---|---|
| DWG / DXF | Yes (it is the drawing; DWG via the free ODA File Converter) | Yes |
| Vector PDF | Yes, almost perfectly (lines, arcs and text are in the PDF) | Yes |
| Image / scan of a CAD drawing | No (pixel tracing gives unusable wobbly lines) | Yes, from the printed dimensions, if legible |
| Hand-drawn sketch | No, and not wanted (not to scale) | Yes, from the written dimensions; show what was read and ask the user to confirm |

The printed dimension is the truth, so a redraw from it is correct by construction and is the only version that can be parameterized. Dimensions that are missing are filled from design rules or standard drawings and marked **ASSUMED**, or asked from the user.

Schematics with no dimensions at all (e.g. a textbook "Section of R.C.C. slab bridge" figure): the model must recognise that no dimensions are present and say so, never invent numbers. It can still answer questions about components, and produce a DXF only from user-given or ASSUMED values. Such schematics belong in the training data with the correct "no dimensions present" response.

## What is not parameterized, and how that improves

- Parts the template knows (slabs, walls, haunches, cells, wing walls, dimensions, levels) are redrawn and follow the parameters.
- Parts it does not know are carried over as fixed content: general notes, title block and revision table, unusual details (drainage spouts, existing pipes, hand corrections). From vector input they are copied in place; from images they are listed as "not reproduced".
- Notes that contain a parameter's value are linked so they update; notes that cannot be linked are flagged as possibly outdated.
- Drawing style follows a CAD standard (ours, or the organisation's if supplied), not the original draughtsman's.

Fixes:

- **Reinforcement:** a `REBAR` layer drawn from bar parameters (main bar diameter and spacing, distributors, cover, top steel, bent-up bars). Positions are computed (n = (width − 2·cover) / spacing + 1). A bar bending schedule (DXF table + Excel) is generated and updates with the parameters. Values are read from the drawing ("16Ø @ 150 c/c"), entered by the user, or ASSUMED. The system draws and checks simple rules (minimum steel, maximum spacing); it does **not** design reinforcement from loads, which needs an engineer's sign-off.
- **River bed and water:** on real GADs, read the surveyed bed levels and draw the profile through them. Without levels, the model outputs bed points relative to the abutment faces and bed level, and a smooth curve is fitted, so it stretches with the span. Water hatching fills the exact region between bed and HFL and moves with the HFL. Pitching and riprap are a standard symbol along the bed.
- **General mechanism:** free-form geometry attached to template parts. The model outputs points relative to a known part, so odd details move with the structure. Trainable from vector PDFs, where every line's exact shape is known. Details that keep recurring get promoted to proper template parts.

## Situational awareness (DXF or answer)

Tool calling (supported natively by Qwen3.5), trained with a mix of:

- normal answers ("what is the HFL on this GAD?"),
- tool calls (`make_dxf({...})` for "convert to DXF", "redraw with 3 spans"),
- clarifying questions for ambiguous requests ("show me the drawing" → "a DXF file, or a summary?"),
- look-alike cases where a normal answer is right ("what does the drawing say about the bearing?").

## Where fine-tuning fits

| Part | How |
|---|---|
| Reading a GAD image or scan | Fine-tuned model |
| Reading a vector PDF / DWG / DXF | Code (exact) |
| Storing extracted values across drawings and projects | Database / search index ("memory"), not fine-tuning |
| Answering questions | Fine-tuned model, with values looked up from the store |
| DXF or answer decision | Fine-tuned model (tool calling) |
| DXF and AutoCAD script | Code (templates) |
| Checks | Code (arithmetic, geometry, overlay) |

Retraining is needed only for a new kind of drawing or when a test shows misreading; new drawings go into the store without retraining.

## Comparison with the agent + one-skill-per-design approach

- Skills load on demand (only a one-line description stays in context), so context size is not the main problem. The real problems: similar skills get confused as their number grows, a general model is weak at reading dense GAD dimensions, and if the skill has the model write CAD code each time, the DXF is not deterministic.
- Any approach needs per-family knowledge somewhere (a template or a skill).
- Recommended combination: **one** GAD skill that looks up the family in a template registry; most "new designs" are parameter variants of a family (cells, skew, cushion), not new templates; the skill calls fixed tools (fine-tuned reader, templates, checks); the agent handles unusual questions, clarification, the DXF-or-answer decision, and families with no template yet (exact non-parameterized copy from vector input, flagged).
- Settle it with a test: 5 real GADs including one unseen design; compare value accuracy, DXF consistency across 3 runs, time and cost per GAD, and the effort to add a design.

## Phases

0. **Collect and classify**: GAD samples per bridge family; agree each family's parameter list with the engineers (the most important step).
1. **GAD reading and Q&A**: parameter extraction from vector PDFs, dataset, queryable GADs (reuses the L-section pipeline).
2. **DXF templates** per family, verified by regenerating real GADs and overlaying them on the originals. Start with the most common family (probably RCC box).
3. **AutoCAD parametric output** (dynamic block library + generated AutoLISP), reinforcement layer and bar bending schedule, interactive parameter editing.
4. **Tool-call training + app**: DXF-or-answer decision, DXF download in the CLI and app.
5. **Images and hand-drawn input**: tiling reader as for the L-section; hand-drawn samples need separate data collection.

## Open questions for the team

1. Which CAD software and version do the engineers use (AutoCAD year, AutoCAD LT, BricsCAD)?
2. Build constrained drawings automatically on a server (needs a licensed copy) or let engineers run the script in their own AutoCAD?
3. Is there a CAD person to build the dynamic block library, or should a first version be generated for them to refine?
4. Which values must be editable constraints for each family (keep the list short; too many constraints make drawings slow and fragile)?
5. Are original DWG/DXF files available for any GADs? (Best possible ground truth.)
6. Which bridge families come first?
7. Is there an organisational or RDSO CAD standard (layers, text styles, dimension style)?
8. Are reinforcement drawings available alongside the GADs?
9. Road and rail variants: confirm the components of each (track, ballast, sleepers vs roadway, parapet, hand rail).
