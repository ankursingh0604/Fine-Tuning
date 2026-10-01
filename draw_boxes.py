import json, re, sys
from PIL import Image, ImageDraw, ImageFont

IMG, RAW, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NORMALIZED_1000 = False   # set True if Gemma's boxes are on a 0-1000 scale
YX_ORDER = False          # set True if boxes are [ymin, xmin, ymax, xmax]

text = open(RAW, encoding="utf-8-sig").read()
data = json.loads(re.search(r"[\[{].*[\]}]", text, re.S).group(0))   # skips prose / ``` fences
if isinstance(data, dict):
    data = next((v for v in data.values() if isinstance(v, list)), [])

img = Image.open(IMG).convert("RGB")
W, H = img.size
draw = ImageDraw.Draw(img)
font = ImageFont.load_default(size=16)
drawn = 0
for i, item in enumerate(data):
    b = item.get("bbox_2d") or item.get("bbox") or item.get("box")
    if not b or len(b) != 4:
        continue
    if YX_ORDER:
        b = [b[1], b[0], b[3], b[2]]
    if NORMALIZED_1000:
        b = [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000]
    x0, x1 = sorted((b[0], b[2]))
    y0, y1 = sorted((b[1], b[3]))
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
    draw.text((x0 + 3, y0 + 3), str(item.get("name") or item.get("label") or i)[:40], fill=(255, 0, 0), font=font)
    drawn += 1
img.save(OUT)
print(f"{drawn} boxes drawn -> {OUT}")