"""
Visual before/after of auto-scaling — shows unit-scale GLB, auto-scaled GLB, and a 1.7m human
reference all on the same meter-scaled axes, so the size difference is obvious.

For each benchmark object: produces a side-view silhouette comparison PNG.

Run from inside the htx-3d container:
    docker exec htx-3d python /app/evaluation/auto_scale_benchmark/visual_comparison.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull

sys.path.insert(0, "/app")
from app.services.auto_scale import auto_scale  # type: ignore

ROOT = Path(__file__).resolve().parent
INPUTS_DIR = Path("/app/gallery/_bench_input")
OUTPUTS_DIR = Path("/app/evaluation/benchmark_v2/outputs")
GT_CSV = ROOT / "ground_truth.csv"
RESULTS_DIR = ROOT / "results"
COMPARE_DIR = RESULTS_DIR / "comparisons"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

# Default pipeline used for the visual comparison (the one with most coverage)
DEFAULT_PIPELINE = "trellis_rembg"


def load_combined_mesh(glb_path: str) -> trimesh.Trimesh:
    obj = trimesh.load(glb_path, force="scene", process=False)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    geoms = []
    for name, g in obj.geometry.items():
        try:
            tf = obj.graph.get(name)[0]
        except Exception:
            tf = np.eye(4)
        g2 = g.copy(); g2.apply_transform(tf); geoms.append(g2)
    return trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]


def project_side_silhouette(glb_path: str) -> np.ndarray:
    """Side view (z=x_screen, y=y_screen). Returns hull points (N, 2) centered on ground (y_min=0)."""
    tri = load_combined_mesh(glb_path)
    verts = tri.vertices.astype(np.float32)
    # Center horizontally, place feet on y=0
    verts = verts.copy()
    verts[:, 0] -= verts[:, 0].mean()
    verts[:, 2] -= verts[:, 2].mean()
    verts[:, 1] -= verts[:, 1].min()
    pts2d = verts[:, [2, 1]]  # side view: x_screen = world_z, y_screen = world_y
    hull = ConvexHull(pts2d)
    return pts2d[hull.vertices]


def human_silhouette(height_m: float = 1.7) -> np.ndarray:
    """Cheesy stick-figure silhouette in meters. Centered horizontally at x=0, feet at y=0."""
    w = height_m * 0.25  # shoulders ≈ 0.4m wide
    pts = [
        # head
        ( -0.08, height_m * 1.00), ( 0.08, height_m * 1.00), ( 0.10, height_m * 0.93),
        ( 0.08, height_m * 0.87), ( -0.08, height_m * 0.87), ( -0.10, height_m * 0.93),
        ( -0.08, height_m * 1.00),  # close
    ]
    # body trapezoid
    body = [
        ( -0.08, height_m * 0.87), (  0.08, height_m * 0.87),
        (  w/2,  height_m * 0.55),               # shoulder
        (  w/2 - 0.02, height_m * 0.55),         # arm out
        (  w/2 + 0.05, height_m * 0.10),         # arm down
        (  w/2 - 0.05, height_m * 0.10),
        (  w/2 - 0.10, height_m * 0.55),         # back to torso
        (  0.15, height_m * 0.42),               # waist
        (  0.18, 0),                              # foot
        (  0.05, 0),                              # foot
        (  0.02, height_m * 0.42),
        ( -0.02, height_m * 0.42),
        ( -0.05, 0),
        ( -0.18, 0),
        ( -0.15, height_m * 0.42),
        ( -(w/2 - 0.10), height_m * 0.55),
        ( -(w/2 - 0.05), height_m * 0.10),
        ( -(w/2 + 0.05), height_m * 0.10),
        ( -(w/2 - 0.02), height_m * 0.55),
        ( -w/2, height_m * 0.55),
    ]
    return np.array(pts + body, dtype=np.float32)


def make_comparison(object_id: str, display_name: str, glb_path: str,
                    scaled_glb_path: str, gt_longest_m: float, pred_longest_m: float,
                    iou: float, save_path: Path) -> None:
    unit_hull = project_side_silhouette(glb_path)
    scaled_hull = project_side_silhouette(scaled_glb_path)
    human = human_silhouette(1.7)

    fig, ax = plt.subplots(figsize=(13, 7))

    # left: unit-scale GLB
    unit_w = unit_hull[:, 0].max() - unit_hull[:, 0].min()
    unit_h = unit_hull[:, 1].max() - unit_hull[:, 1].min()
    unit_offset = -unit_hull[:, 0].min()  # shift to start at x=0
    ax.add_patch(Polygon(unit_hull + np.array([unit_offset, 0]),
                         alpha=0.55, facecolor="#90A4AE", edgecolor="#37474F",
                         linewidth=1.2, label=f"Unit-scale GLB (longest ≈ {max(unit_w, unit_h):.2f} 'units')"))
    ax.text(unit_offset + unit_w / 2, unit_h + 0.4,
            f"BEFORE\nunit scale", ha="center", fontsize=10, color="#37474F", fontweight="bold")

    # gap
    gap = max(2.0, unit_w * 0.5)
    scaled_w = scaled_hull[:, 0].max() - scaled_hull[:, 0].min()
    scaled_h = scaled_hull[:, 1].max() - scaled_hull[:, 1].min()
    scaled_x0 = unit_offset + unit_w + gap - scaled_hull[:, 0].min()
    ax.add_patch(Polygon(scaled_hull + np.array([scaled_x0, 0]),
                         alpha=0.55, facecolor="#66BB6A", edgecolor="#1B5E20",
                         linewidth=1.2, label=f"Auto-scaled GLB (pred {pred_longest_m:.2f} m)"))
    ax.text(scaled_x0 + scaled_w / 2, scaled_h + 0.4,
            f"AFTER\nauto-scaled", ha="center", fontsize=10, color="#1B5E20", fontweight="bold")

    # human
    human_gap = 1.5
    human_x0 = scaled_x0 + scaled_w + human_gap
    ax.add_patch(Polygon(human + np.array([human_x0 + 0.3, 0]),
                         alpha=0.85, facecolor="#212121", edgecolor="#000",
                         linewidth=0.5, label="1.7 m human reference"))

    # ground line
    total_extent = human_x0 + 0.8
    ax.axhline(0, color="#5D4037", linewidth=1.2, alpha=0.7)
    # height ruler on right
    for h_m in range(0, int(max(scaled_h, 5)) + 2):
        ax.axhline(h_m, color="black", linewidth=0.3, alpha=0.2)
        ax.text(-0.4, h_m, f"{h_m}m", fontsize=8, va="center", ha="right", color="#666")

    ax.set_xlim(-0.8, total_extent)
    ax.set_ylim(-0.5, max(scaled_h, 5) + 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    err_pct = abs(pred_longest_m - gt_longest_m) / gt_longest_m * 100 if gt_longest_m > 0 else 0
    title = (f"{display_name}\n"
             f"Ground truth longest dim: {gt_longest_m:.2f} m   ·   "
             f"Predicted: {pred_longest_m:.2f} m   ·   "
             f"Error: {err_pct:.1f}%   ·   View-match IoU: {iou:.2f}")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    with open(GT_CSV) as f:
        rows = list(csv.DictReader(f))

    print(f"generating {len(rows)} comparison images using pipeline={DEFAULT_PIPELINE}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for row in rows:
            object_id = row["object_id"]
            glb_src = OUTPUTS_DIR / object_id / DEFAULT_PIPELINE / "model.glb"
            img = INPUTS_DIR / row["input_filename"]
            if not glb_src.exists() or not img.exists():
                print(f"  skip {object_id}: missing inputs")
                continue

            # Copy + scale (in_place=True on the copy)
            glb_unit = tmpdir / f"{object_id}_unit.glb"
            glb_scaled = tmpdir / f"{object_id}_scaled.glb"
            shutil.copy2(glb_src, glb_unit)
            shutil.copy2(glb_src, glb_scaled)
            res = auto_scale(str(glb_scaled), str(img), in_place=True, debug_png=None)
            if not res.get("auto_scaled"):
                print(f"  skip {object_id}: auto_scale failed: {res.get('reason')}")
                continue
            pred = res["dimensions_m"]["longest_m"]
            iou = (res.get("view_alignment") or {}).get("iou") or 0.0
            gt = float(row["longest_m"])
            save_path = COMPARE_DIR / f"{object_id}.png"
            try:
                make_comparison(object_id, row["display_name"], str(glb_unit), str(glb_scaled),
                                gt, pred, iou, save_path)
                err = abs(pred - gt) / gt * 100
                print(f"  OK  {object_id:30s} gt={gt:5.2f}m pred={pred:5.2f}m err={err:5.1f}% iou={iou:.2f}")
            except Exception as e:
                print(f"  err {object_id}: {e}")


if __name__ == "__main__":
    main()
