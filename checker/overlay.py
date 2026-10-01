"""Sheet images for the checker: the whole sheet with findings marked, and a close-up of one finding."""
import pymupdf
from PIL import Image, ImageDraw, ImageFont

COLOR = {"FLAG": (220, 30, 30), "CHECK": (235, 140, 0), "INFO": (40, 110, 220)}
FAINT = (150, 160, 170)


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(pdf_path, page_index, zoom, clip=None):
    page = pymupdf.open(pdf_path)[page_index]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(*clip) if clip else None, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def sheet_overlay(pdf_path, ann, findings, zoom=0.6):
    """The whole sheet: extracted bridges and curves outlined faintly, findings in colour with their number."""
    img = render(pdf_path, ann["page_index"], zoom)
    d = ImageDraw.Draw(img)
    sc = lambda b: [v * zoom for v in b]
    for b in ann.get("bridges", []):
        for k in ("plan_callout", "lsection_callout", "lsection_levels"):
            if k in b:
                d.rectangle(sc(b[k]["bbox"]), outline=FAINT, width=1)
    for c in ann.get("curves", []):
        d.rectangle(sc(c["bbox"]), outline=FAINT, width=1)
    f_lbl = font(15)
    for f in findings:
        if f["page_index"] != ann["page_index"] or not f.get("bbox"):
            continue
        box = sc(f["bbox"])
        col = COLOR[f["severity"]]
        d.rectangle([box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3], outline=col, width=3)
        tag = f"#{f['no']}"
        tb = d.textbbox((box[0], box[1] - 20), tag, font=f_lbl)
        d.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=col)
        d.text((box[0], box[1] - 20), tag, fill="white", font=f_lbl)
    return img


def finding_crop(pdf_path, ann, f, zoom=2.0, margin=110):
    """A close-up around one finding, readable at full size."""
    if not f.get("bbox"):
        return None
    b = f["bbox"]
    clip = [max(b[0] - margin, 0), max(b[1] - margin, 0), b[2] + margin, b[3] + margin]
    w, h = ann["page_size_pt"]
    clip = [clip[0], clip[1], min(clip[2], w), min(clip[3], h)]
    img = render(pdf_path, ann["page_index"], zoom, clip)
    d = ImageDraw.Draw(img)
    x0, y0 = clip[0], clip[1]
    d.rectangle([(b[0] - x0) * zoom - 4, (b[1] - y0) * zoom - 4, (b[2] - x0) * zoom + 4, (b[3] - y0) * zoom + 4],
                outline=COLOR[f["severity"]], width=4)
    return img
