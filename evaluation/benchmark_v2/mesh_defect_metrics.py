"""Ground-truth-free mesh defect metrics for the 7-pipeline benchmark.

Implements the automated proxies for the rubric's defect flags that need only the
mesh itself (no GT mesh, no multi-view renders):

    floaters        n_components, floater_face_fraction
    holes           boundary_edges, boundary_loops
    flat_collapse   min_axis / longest_axis, and that ratio vs. the object's TRUE
                    proportions from ground_truth.csv  -> aspect_ratio_err

`janus_duplicate` and `front_only_texture` are NOT here: both need orbit renders
(opposite-view CLIP similarity, and rear-vs-front UV statistics).

Why aspect_ratio_err matters: it is orientation-independent (axes are sorted), so
unlike the SSIM/PSNR/LPIPS columns it is unaffected by the fixed-camera pose bug,
and it is grounded in real manufacturer specs rather than a reference render.

Needs trimesh — not present in the host python:
    ~/miniconda3/envs/3D/bin/python mesh_defect_metrics.py
"""

import argparse
import csv
import os
import warnings
from pathlib import Path
from statistics import mean, median

import numpy as np
import trimesh

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS = SCRIPT_DIR / "outputs"
GT_CSV = SCRIPT_DIR.parent / "auto_scale_benchmark" / "ground_truth.csv"
CSV_OUT = SCRIPT_DIR / "mesh_defects.csv"
MD_OUT = SCRIPT_DIR / "mesh_defects_table.md"

PIPELINES = ["trellis2_rembg", "trellis2_sam3", "trellis_rembg", "trellis_sam3",
             "hunyuan_rembg", "hunyuan_sam3", "sam3d_sam3"]
LABEL = {"trellis2_rembg": "TRELLIS.2 · rembg", "trellis2_sam3": "TRELLIS.2 · SAM 3",
         "trellis_rembg": "TRELLIS 1 · rembg", "trellis_sam3": "TRELLIS 1 · SAM 3",
         "hunyuan_rembg": "Hunyuan3D · rembg", "hunyuan_sam3": "Hunyuan3D · SAM 3",
         "sam3d_sam3": "SAM 3D · SAM 3"}


def load_gt() -> dict:
    with open(GT_CSV) as f:
        return {r["object_id"]: r for r in csv.DictReader(f)}


def boundary_stats(mesh) -> tuple[int, int]:
    """Count edges used by exactly one face, and how many loops they form."""
    edges = mesh.edges_sorted
    if len(edges) == 0:
        return 0, 0
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    if len(bnd) == 0:
        return 0, 0
    # loops = connected components of the boundary-edge graph (union-find)
    parent: dict[int, int] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in bnd:
        union(int(a), int(b))
    return len(bnd), len({find(v) for v in parent})


def component_stats(mesh) -> tuple[int, float]:
    """Number of connected components and the face fraction outside the largest."""
    try:
        groups = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    except Exception:
        return 1, 0.0
    if not len(groups):
        return 1, 0.0
    sizes = sorted((len(g) for g in groups), reverse=True)
    return len(sizes), 1.0 - sizes[0] / max(1, len(mesh.faces))


def measure(glb: Path, gt: dict | None) -> dict:
    mesh = trimesh.load(str(glb), force="mesh")

    # CRITICAL: these GLBs ship with unwelded vertices (each triangle carries its
    # own copies, for per-face UVs). Face adjacency is built from shared vertex
    # indices, so without welding every face looks like its own component and
    # every edge looks like a boundary — component counts came out in the
    # thousands and floater fraction at ~0.95 for meshes that are visibly solid.
    # Weld first; topology metrics are meaningless otherwise.
    raw_verts = int(len(mesh.vertices))
    try:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    except TypeError:                     # older trimesh signature
        mesh.merge_vertices()
    welded_verts = int(len(mesh.vertices))

    dims = np.sort(mesh.bounds[1] - mesh.bounds[0])[::-1]   # longest -> shortest
    longest, _, shortest = (float(d) for d in dims)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    aspect = shortest / longest if longest else 0.0

    n_comp, floater_frac = component_stats(mesh)
    bnd_edges, bnd_loops = boundary_stats(mesh)

    row = {
        "faces": int(len(mesh.faces)),
        "vertices": raw_verts,
        "welded_vertices": welded_verts,
        "n_components": n_comp,
        "floater_face_frac": round(floater_frac, 4),
        "boundary_edges": bnd_edges,
        "boundary_loops": bnd_loops,
        "degenerate_faces": int(np.sum(mesh.area_faces < 1e-10)),
        "aspect": round(aspect, 4),
        "min_axis_over_diag": round(shortest / diag if diag else 0.0, 4),
    }
    if gt:
        gt_aspect = float(gt["shortest_m"]) / float(gt["longest_m"])
        row["gt_aspect"] = round(gt_aspect, 4)
        # 1.0 = correct proportions; 0.02 = 50x flatter than reality
        row["aspect_ratio"] = round(aspect / gt_aspect, 4) if gt_aspect else None
        row["aspect_log_err"] = (round(abs(np.log(aspect / gt_aspect)), 4)
                                 if gt_aspect and aspect > 0 else None)
        row["gt_confidence"] = gt["gt_confidence"]
    return row


# Multi-object scenes where the rembg front-end cannot isolate the intended
# target, and whose ground truth is estimated rather than published. Excluded
# from the reported aggregate; still written to the per-row CSV.
EXCLUDE_DEFAULT = "apics"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default=EXCLUDE_DEFAULT,
                    help="substring of object_id to exclude from aggregates "
                         "(default: 'apics'; pass '' to include everything)")
    args = ap.parse_args()
    gt = load_gt()
    rows: list[dict] = []
    for obj in sorted(gt):
        for pipe in PIPELINES:
            glb = OUTPUTS / obj / pipe / "model.glb"
            if not glb.exists():
                print(f"  missing {obj}/{pipe}")
                continue
            r = {"object_id": obj, "pipeline": pipe}
            try:
                r.update(measure(glb, gt.get(obj)))
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {e}"
                print(f"  ERROR {obj}/{pipe}: {e}")
            rows.append(r)
            print(f"  {obj:28s} {pipe:16s} comp={r.get('n_components'):>4} "
                  f"loops={r.get('boundary_loops'):>4} aspect_ratio={r.get('aspect_ratio')}")

    fields = ["object_id", "pipeline", "faces", "vertices", "n_components",
              "floater_face_frac", "boundary_edges", "boundary_loops",
              "degenerate_faces", "welded_vertices", "aspect", "gt_aspect", "aspect_ratio",
              "aspect_log_err", "min_axis_over_diag", "gt_confidence", "error"]
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate ----
    ex = args.exclude
    agg_rows = [r for r in rows if not (ex and ex in r["object_id"])]
    n_obj = len({r["object_id"] for r in agg_rows})
    dropped = sorted({r["object_id"] for r in rows if ex and ex in r["object_id"]})
    L = [f"# Mesh defect metrics — 7 pipelines × {n_obj} objects\n",
         "Ground-truth-free, computed from the GLBs alone. Orientation-independent",
         "(bbox axes are sorted), so unlike the SSIM/PSNR/LPIPS columns these are",
         "unaffected by the fixed-camera pose bug.\n",
         (f"Excluded from these aggregates ({len(dropped)} objects, still present in "
          f"`mesh_defects.csv`): {', '.join(dropped)}. Multi-object scenes where "
          "background removal cannot isolate the intended target, with estimated "
          "rather than published ground truth.\n" if dropped else ""),
         "`aspect_ratio` = predicted (shortest/longest) ÷ true (shortest/longest).",
         "1.00 = correct proportions; 0.02 = 50× flatter than the real object.\n",
         "## Per-pipeline means\n",
         "| Pipeline | components | floater face frac | boundary loops | "
         "degenerate | aspect_ratio (median) | mean \\|log aspect err\\| | n ≤0.5 (too flat) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def num(rs, k):
        return [r[k] for r in rs if isinstance(r.get(k), (int, float))]

    for pipe in PIPELINES:
        rs = [r for r in agg_rows if r["pipeline"] == pipe and "error" not in r]
        if not rs:
            continue
        ar = num(rs, "aspect_ratio")
        L.append(
            f"| {LABEL[pipe]} | {mean(num(rs,'n_components')):.1f} | "
            f"{mean(num(rs,'floater_face_frac')):.3f} | "
            f"{mean(num(rs,'boundary_loops')):.0f} | "
            f"{mean(num(rs,'degenerate_faces')):.0f} | "
            f"{median(ar):.2f} | {mean(num(rs,'aspect_log_err')):.2f} | "
            f"{sum(1 for x in ar if x <= 0.5)}/{len(ar)} |")

    L.append("\n## aspect_ratio per object (bold = ≤0.5, i.e. >2× too flat)\n")
    L.append("| Object | true aspect | " + " | ".join(LABEL[p] for p in PIPELINES) + " |")
    L.append("|---|--:|" + "--:|" * len(PIPELINES))
    for obj in sorted({r["object_id"] for r in agg_rows}):
        rs = {r["pipeline"]: r for r in agg_rows if r["object_id"] == obj}
        first = next((r for r in rs.values() if r.get("gt_aspect")), None)
        cells = []
        for p in PIPELINES:
            v = rs.get(p, {}).get("aspect_ratio")
            cells.append("—" if v is None else (f"**{v:.2f}**" if v <= 0.5 else f"{v:.2f}"))
        L.append(f"| {obj} | {first['gt_aspect'] if first else '—'} | " + " | ".join(cells) + " |")

    MD_OUT.write_text("\n".join(L) + "\n")
    print(f"\n✓ {CSV_OUT}\n✓ {MD_OUT}")


if __name__ == "__main__":
    main()
