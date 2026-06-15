"""Build slide-ready comparison figures from benchmark renders.

Produces:
- figure_6_5_1.png — 3 representative objects x (input + 5 conditions)
- figure_7_1_1.png — Hunyuan flat-mesh failure cases (motorcycle, car booth)
"""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
RENDERS = Path("/home/cj/HTX-3D/gallery/_bench_renders")
INPUTS = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = ["trellis_rembg", "trellis_sam3", "hunyuan_rembg", "hunyuan_sam3", "sam3d_sam3"]
COND_LABEL = {
    "trellis_rembg": "TRELLIS (rembg)",
    "trellis_sam3":  "TRELLIS (+SAM3)",
    "hunyuan_rembg": "Hunyuan (rembg)",
    "hunyuan_sam3":  "Hunyuan (+SAM3)",
    "sam3d_sam3":    "SAM 3D Objects",
}

CELL = 384      # cell size in pixels
GAP = 12
LABEL_H = 36

def load_input(obj_id: str, size: int) -> Image.Image:
    obj_dir = INPUTS / obj_id
    src = next((p for p in obj_dir.iterdir() if p.stem == "original"), None)
    if src is None:
        return Image.new("RGB", (size, size), "white")
    return Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS)

def load_render(obj_id: str, cond: str, size: int) -> Image.Image:
    p = RENDERS / f"{obj_id}__{cond}.png"
    if not p.exists():
        return Image.new("RGB", (size, size), "white")
    return Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)

def get_font(sz=18):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, sz)
    return ImageFont.load_default()

def grid(rows: list[tuple[str, list[tuple[str, Image.Image]]]],
         column_labels: list[str],
         out_path: Path,
         title: str = "") -> None:
    """rows = [(row_label, [(col_subtitle, image), ...]), ...]"""
    n_cols = len(column_labels)
    n_rows = len(rows)

    row_label_w = 130
    title_h = 44 if title else 0

    W = row_label_w + n_cols * (CELL + GAP) - GAP + 2 * GAP
    H = title_h + LABEL_H + n_rows * (CELL + GAP) - GAP + 2 * GAP

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    fnt_lbl = get_font(20)
    fnt_title = get_font(28)
    fnt_row = get_font(18)

    if title:
        draw.text((GAP, 8), title, fill="black", font=fnt_title)

    # column labels
    y0 = title_h
    x0 = row_label_w + GAP
    for i, lbl in enumerate(column_labels):
        x = x0 + i * (CELL + GAP)
        bbox = draw.textbbox((0, 0), lbl, font=fnt_lbl)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (CELL - text_w) // 2, y0 + 6), lbl, fill="black", font=fnt_lbl)

    # rows
    y = y0 + LABEL_H
    for row_lbl, cells in rows:
        # row label rotated would be best, but for simplicity put at left
        bbox = draw.textbbox((0, 0), row_lbl, font=fnt_row)
        text_w = bbox[2] - bbox[0]
        draw.text((row_label_w - text_w - 8, y + CELL // 2 - 12), row_lbl, fill="black", font=fnt_row)
        x = x0
        for sub, img in cells:
            canvas.paste(img, (x, y))
            if sub:
                draw.text((x + 6, y + CELL - 24), sub, fill="white",
                          font=get_font(16), stroke_width=2, stroke_fill="black")
            x += CELL + GAP
        y += CELL + GAP

    canvas.save(out_path)
    print(f"  -> {out_path}")


def figure_6_5_1():
    """Three representative objects across all 5 conditions, with input as col 0."""
    objects = [
        ("01_scdf_ambulance",      "Vehicle (clean)"),
        ("07_security_guard_house","Infrastructure"),
        ("13_apics_booth_barrier", "Multi-object scene"),
    ]
    column_labels = ["Input photo"] + [COND_LABEL[c] for c in CONDITIONS]
    rows = []
    for oid, lbl in objects:
        cells = [("", load_input(oid, CELL))]
        for c in CONDITIONS:
            cells.append(("", load_render(oid, c, CELL)))
        rows.append((lbl + "\n" + oid.split("_", 1)[1][:18], cells))
    grid(rows, column_labels, OUT_DIR / "figure_6_5_1.png",
         title="Figure 6.5.1 — Per-condition reconstructions (3 representative objects)")


def figure_7_1_1():
    """Hunyuan flat-mesh failure on side-profile shots."""
    objects = [
        ("04_police_motorcycle",   "Side profile (motorcycle)"),
        ("11_apics_red_car_booth", "Side profile (car booth)"),
    ]
    show_conds = ["trellis_sam3", "hunyuan_sam3", "sam3d_sam3"]
    column_labels = ["Input photo"] + [COND_LABEL[c] for c in show_conds]
    rows = []
    for oid, lbl in objects:
        cells = [("", load_input(oid, CELL))]
        for c in show_conds:
            cells.append(("", load_render(oid, c, CELL)))
        rows.append((lbl, cells))
    grid(rows, column_labels, OUT_DIR / "figure_7_1_1.png",
         title="Figure 7.1.1 — Hunyuan plane-collapse failure (X / Z axis ≈ 0)")


def main():
    print("Building figures...")
    figure_6_5_1()
    figure_7_1_1()


if __name__ == "__main__":
    main()
