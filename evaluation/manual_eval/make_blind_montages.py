"""Build blinded montages + score sheet for the manual (human) evaluation.

For each of the 14 objects this writes blind_montages/<obj>.png laid out as

    INPUT (reference) | A | B | C | D | E | F | G

where A..G are the 7 pipeline renders in a per-object RANDOMISED order, so the
rater cannot infer the engine from the column position. The slot -> pipeline
mapping goes to blind_key.csv, which the rater must not open until scoring is
finished.

Also emits scores_TEMPLATE.csv — one row per (object, slot) — for the rater to
fill in. Copy it to scores_<yourname>.csv before filling it in; analyze_manual.py
picks up every scores_*.csv except the template.

Usage:
    python make_blind_montages.py
    python make_blind_montages.py --seed 99      # different blinding
"""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent / "trellis2_benchmark"
RENDERS_DIR = BENCH_DIR / "renders"
DATA_DIR = SCRIPT_DIR.parent / "benchmark_v2" / "data"
GT_CSV = SCRIPT_DIR.parent / "auto_scale_benchmark" / "ground_truth.csv"

OUT_DIR = SCRIPT_DIR / "blind_montages"
KEY_PATH = SCRIPT_DIR / "blind_key.csv"
TEMPLATE_PATH = SCRIPT_DIR / "scores_TEMPLATE.csv"

PIPELINES = [
    "trellis2_rembg",
    "trellis2_sam3",
    "trellis_rembg",
    "trellis_sam3",
    "hunyuan_rembg",
    "hunyuan_sam3",
    "sam3d_sam3",
]
SLOTS = ["A", "B", "C", "D", "E", "F", "G"]

CELL = 560
GAP = 10
TITLE_H = 62
HEADER_H = 54
BG = (255, 255, 255)
FG = (20, 20, 20)
ACCENT = (200, 30, 30)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def load_font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except Exception:
        return ImageFont.load_default()


def load_display_names() -> dict[str, str]:
    names = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            names[row["object_id"]] = row["display_name"]
    return names


def find_original(obj_id: str) -> Path | None:
    d = DATA_DIR / obj_id
    if not d.is_dir():
        return None
    for f in d.iterdir():
        if f.stem == "original":
            return f
    return None


def apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
    """Uniform brightness lift for dark objects.

    Applied identically to all 7 renders of an object and never to the input,
    so it cannot bias one pipeline against another — but it must be disclosed
    in the method if used.
    """
    if gamma == 1.0:
        return img
    lut = [min(255, int(255.0 * ((i / 255.0) ** (1.0 / gamma)) + 0.5)) for i in range(256)]
    return img.point(lut * len(img.getbands()))


def fit_square(img: Image.Image, size: int) -> Image.Image:
    """Letterbox onto a white square without distorting aspect ratio."""
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), BG)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def build_one(obj_id: str, display_name: str, order: list[str],
              gamma: float = 1.0) -> Image.Image | None:
    cells: list[tuple[str, Image.Image]] = []

    orig = find_original(obj_id)
    if orig is None:
        print(f"  ! {obj_id}: no original image, skipping")
        return None
    cells.append(("INPUT (reference)", fit_square(Image.open(orig), CELL)))

    for slot, pipe in zip(SLOTS, order):
        r = RENDERS_DIR / f"{obj_id}__{pipe}.png"
        if not r.exists():
            print(f"  ! {obj_id}/{pipe}: render missing, skipping object")
            return None
        cells.append((slot, apply_gamma(fit_square(Image.open(r), CELL), gamma)))

    n = len(cells)
    W = n * CELL + (n + 1) * GAP
    H = TITLE_H + HEADER_H + CELL + 2 * GAP
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    f_title = load_font("DejaVuSans-Bold.ttf", 34)
    f_slot = load_font("DejaVuSans-Bold.ttf", 38)
    f_input = load_font("DejaVuSans-Bold.ttf", 24)

    d.text((GAP, 16), f"{obj_id}  —  {display_name}", font=f_title, fill=FG)

    x = GAP
    y_hdr = TITLE_H
    y_img = TITLE_H + HEADER_H + GAP
    for label, im in cells:
        is_input = label.startswith("INPUT")
        font = f_input if is_input else f_slot
        colour = ACCENT if is_input else FG
        tw = d.textlength(label, font=font)
        d.text((x + (CELL - tw) / 2, y_hdr + (6 if is_input else 2)),
               label, font=font, fill=colour)
        canvas.paste(im, (x, y_img))
        d.rectangle([x, y_img, x + CELL - 1, y_img + CELL - 1],
                    outline=(215, 215, 215), width=1)
        x += CELL + GAP

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337,
                    help="blinding seed; record it if you rebuild")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="uniform brightness lift on renders only (e.g. 1.6 for "
                         "dark objects); applied identically to all pipelines")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = load_display_names()
    obj_ids = sorted(names)

    rng = random.Random(args.seed)
    key_rows: list[dict] = []
    template_rows: list[dict] = []
    built = 0

    for obj_id in obj_ids:
        order = PIPELINES[:]
        rng.shuffle(order)

        img = build_one(obj_id, names.get(obj_id, obj_id), order, args.gamma)
        if img is None:
            continue
        out = OUT_DIR / f"{obj_id}.png"
        img.save(out)
        built += 1
        print(f"  ok  {out.name}")

        for slot, pipe in zip(SLOTS, order):
            key_rows.append({"object_id": obj_id, "slot": slot, "pipeline": pipe})
            template_rows.append({
                "object_id": obj_id, "slot": slot,
                "geometry_1to5": "", "texture_1to5": "",
                "floaters": "", "holes": "", "flat_collapse": "",
                "janus_duplicate": "", "front_only_texture": "",
                "notes": "",
            })

    with open(KEY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["object_id", "slot", "pipeline"])
        w.writeheader()
        w.writerows(key_rows)

    with open(TEMPLATE_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(template_rows[0].keys()))
        w.writeheader()
        w.writerows(template_rows)

    print(f"\nBuilt {built} blinded montages -> {OUT_DIR}")
    print(f"Key (do not open until scored) -> {KEY_PATH}")
    print(f"Score sheet template           -> {TEMPLATE_PATH}")
    print(f"Blinding seed: {args.seed}")
    print("\nNext: cp scores_TEMPLATE.csv scores_<yourname>.csv  and fill it in.")


if __name__ == "__main__":
    main()
