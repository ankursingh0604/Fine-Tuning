"""Ask the fine-tuned model one question from the command line (no web page).

    python ask.py --adapter adapter.zip --image crop.png --question "Read the callout for bridge 560 as JSON."
    python ask.py --adapter adapter.zip --image bridge.png --image2 bands.png --question "..." --compare
    python ask.py ... --4bit          # GPUs with less than about 10 GB

--compare also prints the untouched model's answer to the same question.
"""
import argparse

from PIL import Image

from app import Model, prepare

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, help="adapter.zip or an unzipped adapter folder")
ap.add_argument("--image", required=True)
ap.add_argument("--image2", help="optional second image, e.g. the data bands for a bridge question")
ap.add_argument("--question", required=True)
ap.add_argument("--4bit", dest="four_bit", action="store_true")
ap.add_argument("--compare", action="store_true", help="also ask the untouched model")
args = ap.parse_args()

model = Model(args.adapter, args.four_bit)
imgs = [prepare(Image.open(args.image))] + ([prepare(Image.open(args.image2))] if args.image2 else [])
print("\n=== fine-tuned ===\n" + model.ask(imgs, args.question))
if args.compare:
    print("\n=== untouched ===\n" + model.ask(imgs, args.question, use_adapter=False))
