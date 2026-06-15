"""Slide-ready outputs: a summary CSV and a PNG of the headline metrics table."""

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = Path(__file__).resolve().parent
AGG = SCRIPT_DIR / "metrics_aggregate.json"
RECORDS = SCRIPT_DIR / "metrics.json"
OUT_CSV = SCRIPT_DIR / "metrics_aggregate.csv"
OUT_PER_OBJ_CSV = SCRIPT_DIR / "metrics_per_object.csv"
OUT_TABLE_PNG = SCRIPT_DIR / "figures" / "table_6_5_1_slide.png"

CONDITIONS = ["trellis_rembg", "trellis_sam3", "hunyuan_rembg", "hunyuan_sam3", "sam3d_sam3"]
COND_LABEL = {
    "trellis_rembg": "TRELLIS (rembg)",
    "trellis_sam3":  "TRELLIS (+SAM3)",
    "hunyuan_rembg": "Hunyuan (rembg)",
    "hunyuan_sam3":  "Hunyuan (+SAM3)",
    "sam3d_sam3":    "SAM 3D Objects",
}

with open(AGG) as f:
    agg = json.load(f)
with open(RECORDS) as f:
    records = json.load(f)

# CSV — wide aggregate
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    cols = ["condition", "n", "gen_time_s_mean", "faces_mean", "vertices_mean",
            "file_size_mb_mean", "min_axis_dim_mean", "ssim_mean", "psnr_mean",
            "lpips_mean", "clip_mean"]
    w.writerow(cols)
    for c in CONDITIONS:
        a = agg.get(c, {})
        w.writerow([c] + [a.get(k, "") for k in cols[1:]])
print(f"  -> {OUT_CSV}")

# CSV — per-object
with open(OUT_PER_OBJ_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["obj_id", "condition", "engine", "use_sam3", "task_id",
                "gen_time_s", "faces", "vertices", "file_size_mb",
                "watertight", "bbox_x", "bbox_y", "bbox_z",
                "ssim", "psnr", "lpips", "clip"])
    for r in records:
        if r.get("status") != "completed":
            continue
        m = r.get("mesh_metrics", {})
        i = r.get("image_metrics", {})
        w.writerow([
            r["obj_id"], r["condition"], r["engine"], r["use_sam3"], r["task_id"],
            r.get("generation_time_s"),
            m.get("faces"), m.get("vertices"), m.get("file_size_mb"),
            m.get("watertight"), m.get("bbox_x"), m.get("bbox_y"), m.get("bbox_z"),
            i.get("ssim"), i.get("psnr"), i.get("lpips"), i.get("clip"),
        ])
print(f"  -> {OUT_PER_OBJ_CSV}")

# Headline table PNG via matplotlib
fig, ax = plt.subplots(figsize=(13, 3.6))
ax.axis("off")

headers = ["Condition", "n", "Gen time (s)", "Faces", "File MB",
           "SSIM ↑", "PSNR ↑", "LPIPS ↓", "CLIP ↑"]
rows = []
for c in CONDITIONS:
    a = agg.get(c, {})
    rows.append([
        COND_LABEL[c],
        a.get("n", "—"),
        f"{a.get('gen_time_s_mean', 0):.1f}",
        f"{int(a.get('faces_mean', 0)):,}",
        f"{a.get('file_size_mb_mean', 0):.2f}",
        f"{a.get('ssim_mean', 0):.3f}",
        f"{a.get('psnr_mean', 0):.2f}",
        f"{a.get('lpips_mean', 0):.3f}",
        f"{a.get('clip_mean', 0):.3f}",
    ])

# Highlight winners
ssim_vals = [agg.get(c, {}).get("ssim_mean", 0) for c in CONDITIONS]
clip_vals = [agg.get(c, {}).get("clip_mean", 0) for c in CONDITIONS]
lpips_vals = [agg.get(c, {}).get("lpips_mean", 1) for c in CONDITIONS]
gen_vals = [agg.get(c, {}).get("gen_time_s_mean", 999) for c in CONDITIONS]
ssim_best = ssim_vals.index(max(ssim_vals))
clip_best = clip_vals.index(max(clip_vals))
lpips_best = lpips_vals.index(min(lpips_vals))
gen_best = gen_vals.index(min(gen_vals))

table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.65)

# Style header
for col in range(len(headers)):
    cell = table[0, col]
    cell.set_facecolor("#1f3a5f"); cell.set_text_props(color="white", weight="bold")

# Highlight winners (cells are 1-indexed in row)
WIN = "#c8e6c9"
table[gen_best + 1, 2].set_facecolor(WIN)
table[ssim_best + 1, 5].set_facecolor(WIN)
table[lpips_best + 1, 7].set_facecolor(WIN)
table[clip_best + 1, 8].set_facecolor(WIN)

plt.title("Per-condition mean metrics — 14 objects × 5 conditions = 70 reconstructions\n(green = best in column)",
          fontsize=12, pad=14)
plt.tight_layout()
OUT_TABLE_PNG.parent.mkdir(exist_ok=True)
plt.savefig(OUT_TABLE_PNG, dpi=160, bbox_inches="tight", facecolor="white")
print(f"  -> {OUT_TABLE_PNG}")
