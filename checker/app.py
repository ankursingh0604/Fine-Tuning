"""PDF checker for Plan & L-Section sheets: upload a PDF, get every sheet extracted, checked and explained.

    .venv\\Scripts\\python checker\\app.py        then open http://localhost:7861

Runs on an ordinary laptop (no GPU, no model): the PDF's text layer is read exactly by
scripts/annotate.py, the checks are in checker/checks.py and questions are answered from the
extracted data by checker/qa.py.
"""
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr               # noqa: E402

import annotate as A              # noqa: E402
import checks                     # noqa: E402
import overlay                    # noqa: E402
import qa                         # noqa: E402

COLUMNS = ["#", "Severity", "Sheet", "Category", "Item", "Chainage", "Finding"]


def sheet_label(a):
    return f"Sheet {a['sheet_info']['sheet_no']}" if a["sheet_info"].get("sheet_no") else f"Page {a['page_index'] + 1}"


def run_check(uploaded, local_choice):
    path = uploaded or (str(ROOT / local_choice) if local_choice else None)
    if not path:
        return "Upload a PDF or pick one from the folder.", [], gr.update(choices=[], value=None), None, None, None, {}
    anns = A.annotate_pdf(path)
    found = checks.check_all(anns)
    for i, f in enumerate(found, 1):
        f["no"] = i
    rows = [[f["no"], f["severity"], f["sheet"], f["category"], f["item"], checks.g(f["chainage"]) if f["chainage"] else "", f["message"]]
            for f in found]
    report = Path(tempfile.gettempdir()) / f"{Path(path).stem}_check_report.csv"
    with open(report, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)
    labels = [sheet_label(a) for a in anns]
    state = {"path": path, "anns": anns, "found": found}
    first = overlay.sheet_overlay(path, anns[0], found) if anns else None
    return (qa.summary(anns, found), rows, gr.update(choices=labels, value=labels[0] if labels else None),
            first, None, str(report), state)


def show_sheet(label, state):
    if not state or not label:
        return None
    a = next(a for a in state["anns"] if sheet_label(a) == label)
    return overlay.sheet_overlay(state["path"], a, state["found"])


def pick_finding(evt: gr.SelectData, state):
    if not state:
        return None, gr.update(), None
    f = state["found"][evt.index[0]]
    a = next(a for a in state["anns"] if a["page_index"] == f["page_index"])
    return overlay.finding_crop(state["path"], a, f), gr.update(value=sheet_label(a)), overlay.sheet_overlay(state["path"], a, state["found"])


def ask(question, state):
    if not state:
        return "Check a PDF first."
    return qa.answer(question or "", state["anns"], state["found"])


def build():
    local = sorted(p.name for p in ROOT.glob("*.pdf"))
    with gr.Blocks(title="L-Section PDF checker") as ui:
        gr.Markdown("## L-Section PDF checker\nUpload a Plan & L-Section PDF. Every sheet is read from the PDF's text layer, "
                    "checked, and can then be questioned. Findings: **FLAG** = a value outside its requirement, "
                    "**CHECK** = values on the drawing that disagree, **INFO** = incomplete or worth a look.")
        with gr.Row():
            upload = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
            with gr.Column():
                local_pick = gr.Dropdown(choices=local, label="…or a PDF in the project folder", value=None)
                go = gr.Button("Check the PDF", variant="primary")
        summary = gr.Markdown()
        findings = gr.Dataframe(headers=COLUMNS, label="Findings (click a row to see it on the sheet)", wrap=True,
                                interactive=False, column_widths=["4%", "7%", "6%", "14%", "13%", "9%", "47%"])
        report = gr.File(label="Download the findings (CSV)")
        with gr.Row():
            with gr.Column(scale=3):
                sheet = gr.Dropdown(label="Sheet", choices=[])
                sheet_img = gr.Image(label="Sheet with findings marked (red FLAG, orange CHECK, blue INFO)", type="pil")
            with gr.Column(scale=2):
                crop = gr.Image(label="Selected finding, close up", type="pil")
        gr.Markdown("### Ask about this PDF")
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="e.g. bridge 560 · CH 1242662.9 · curve 377 · which bridges are flagged",
                                  scale=4)
            ask_btn = gr.Button("Ask", scale=1)
        answer_md = gr.Markdown()
        state = gr.State({})

        go.click(run_check, [upload, local_pick], [summary, findings, sheet, sheet_img, crop, report, state])
        sheet.change(show_sheet, [sheet, state], sheet_img)
        findings.select(pick_finding, [state], [crop, sheet, sheet_img])
        ask_btn.click(ask, [question, state], answer_md)
        question.submit(ask, [question, state], answer_md)
    return ui


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7861)
