"""
Benchmark the auto-scale pipeline against known real-world dimensions.

Iterates 14 HTX objects × 5 generation pipelines (Hunyuan/TRELLIS/SAM3D × rembg/sam3 segs),
runs auto_scale on each GLB without modifying the original, compares to ground truth from
ground_truth.csv, and produces:
    - results.csv          all per-row data
    - results_summary.csv  per-pipeline + overall aggregates
    - report.md            markdown tables for the report
    - scatter.png          predicted vs actual longest dim, all pipelines
    - per_pipeline.png     MAPE per pipeline (bar)
    - per_object.png       per-object error bars across pipelines

Run from inside the htx-3d container:
    docker exec htx-3d python /app/evaluation/auto_scale_benchmark/benchmark_auto_scale.py
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# matplotlib for plots (already in container)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# auto_scale module
sys.path.insert(0, "/app")
from app.services.auto_scale import auto_scale  # type: ignore

ROOT = Path(__file__).resolve().parent
INPUTS_DIR = Path("/app/gallery/_bench_input")
OUTPUTS_DIR = Path("/app/evaluation/benchmark_v2/outputs")
GT_CSV = ROOT / "ground_truth.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PIPELINES = ["trellis_rembg", "trellis_sam3", "hunyuan_rembg", "hunyuan_sam3", "sam3d_sam3"]


@dataclass
class Row:
    object_id: str
    display_name: str
    pipeline: str
    gt_longest_m: float
    gt_middle_m: float
    gt_shortest_m: float
    gt_confidence: str
    pred_longest_m: Optional[float]
    pred_middle_m: Optional[float]
    pred_shortest_m: Optional[float]
    err_longest_pct: Optional[float]
    err_middle_pct: Optional[float]
    err_shortest_pct: Optional[float]
    err_mean_pct: Optional[float]
    iou: Optional[float]
    view_method: Optional[str]
    mask_source: Optional[str]
    pred_confidence: Optional[str]
    object_distance_m: Optional[float]
    auto_scaled: bool
    reason: Optional[str]


def load_gt() -> list[dict]:
    with open(GT_CSV) as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def find_glb(object_id: str, pipeline: str) -> Optional[Path]:
    p = OUTPUTS_DIR / object_id / pipeline / "model.glb"
    return p if p.exists() else None


def safe_pct_err(pred: Optional[float], gt: float) -> Optional[float]:
    if pred is None or gt <= 0:
        return None
    return abs(pred - gt) / gt * 100.0


def run_one(gt_row: dict, pipeline: str, tmpdir: Path) -> Row:
    object_id = gt_row["object_id"]
    img = INPUTS_DIR / gt_row["input_filename"]
    glb_src = find_glb(object_id, pipeline)

    base = Row(
        object_id=object_id,
        display_name=gt_row["display_name"],
        pipeline=pipeline,
        gt_longest_m=float(gt_row["longest_m"]),
        gt_middle_m=float(gt_row["middle_m"]),
        gt_shortest_m=float(gt_row["shortest_m"]),
        gt_confidence=gt_row["gt_confidence"],
        pred_longest_m=None, pred_middle_m=None, pred_shortest_m=None,
        err_longest_pct=None, err_middle_pct=None, err_shortest_pct=None, err_mean_pct=None,
        iou=None, view_method=None, mask_source=None, pred_confidence=None,
        object_distance_m=None, auto_scaled=False, reason=None,
    )
    if not glb_src:
        base.reason = f"GLB missing: {object_id}/{pipeline}"
        return base
    if not img.exists():
        base.reason = f"image missing: {img}"
        return base

    # copy GLB to scratch so auto_scale doesn't touch the original
    glb_tmp = tmpdir / f"{object_id}_{pipeline}.glb"
    shutil.copy2(glb_src, glb_tmp)

    out = auto_scale(str(glb_tmp), str(img), in_place=True, debug_png=None)

    base.auto_scaled = bool(out.get("auto_scaled"))
    base.reason = out.get("reason")
    if not base.auto_scaled:
        return base

    dims = out.get("dimensions_m") or {}
    base.pred_longest_m = dims.get("longest_m")
    base.pred_middle_m  = dims.get("middle_m")
    base.pred_shortest_m = dims.get("shortest_m")
    va = out.get("view_alignment") or {}
    base.iou = va.get("iou")
    base.view_method = va.get("method")
    base.mask_source = out.get("mask_source")
    base.pred_confidence = out.get("confidence")
    base.object_distance_m = out.get("object_distance_m")

    base.err_longest_pct  = safe_pct_err(base.pred_longest_m,  base.gt_longest_m)
    base.err_middle_pct   = safe_pct_err(base.pred_middle_m,   base.gt_middle_m)
    base.err_shortest_pct = safe_pct_err(base.pred_shortest_m, base.gt_shortest_m)
    errs = [e for e in (base.err_longest_pct, base.err_middle_pct, base.err_shortest_pct) if e is not None]
    base.err_mean_pct = float(np.mean(errs)) if errs else None
    return base


def write_results_csv(rows: list[Row]) -> Path:
    p = RESULTS_DIR / "results.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return p


def aggregates(rows: list[Row]) -> dict:
    """Compute aggregate MAPE and other stats, both overall and high-confidence-GT-only."""
    def agg(filtered: list[Row]) -> dict:
        ok = [r for r in filtered if r.auto_scaled and r.err_longest_pct is not None]
        if not ok:
            return {"n": 0}
        longest = [r.err_longest_pct for r in ok]
        middle  = [r.err_middle_pct  for r in ok if r.err_middle_pct  is not None]
        short   = [r.err_shortest_pct for r in ok if r.err_shortest_pct is not None]
        ious    = [r.iou for r in ok if r.iou is not None]
        within20 = sum(1 for e in longest if e <= 20.0) / len(longest)
        within50 = sum(1 for e in longest if e <= 50.0) / len(longest)
        return {
            "n": len(ok),
            "longest_mape": float(np.mean(longest)),
            "longest_median_pct": float(np.median(longest)),
            "middle_mape": float(np.mean(middle)) if middle else None,
            "shortest_mape": float(np.mean(short)) if short else None,
            "mean_iou": float(np.mean(ious)) if ious else None,
            "frac_within_20pct": within20,
            "frac_within_50pct": within50,
        }

    out: dict = {}
    out["overall"] = agg(rows)
    out["high_gt_only"] = agg([r for r in rows if r.gt_confidence == "high"])
    out["per_pipeline"] = {p: agg([r for r in rows if r.pipeline == p]) for p in PIPELINES}
    out["per_pipeline_high_gt"] = {p: agg([r for r in rows if r.pipeline == p and r.gt_confidence == "high"]) for p in PIPELINES}
    return out


def plot_scatter(rows: list[Row], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"trellis_rembg":"#4FC3F7","trellis_sam3":"#0288D1","hunyuan_rembg":"#CE93D8","hunyuan_sam3":"#7B1FA2","sam3d_sam3":"#81C784"}
    for p in PIPELINES:
        xs = [r.gt_longest_m for r in rows if r.pipeline == p and r.pred_longest_m is not None]
        ys = [r.pred_longest_m for r in rows if r.pipeline == p and r.pred_longest_m is not None]
        ax.scatter(xs, ys, label=p, s=55, alpha=0.75, color=colors[p], edgecolor="white")
    lim = max(20.0, max([r.gt_longest_m for r in rows] + [r.pred_longest_m or 0 for r in rows]) * 1.1)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y = x (perfect)")
    ax.plot([0, lim], [0, lim * 1.2], "k:", lw=0.7, alpha=0.4)
    ax.plot([0, lim], [0, lim * 0.8], "k:", lw=0.7, alpha=0.4, label="±20%")
    ax.set_xlabel("Ground-truth longest dim (m)")
    ax.set_ylabel("Auto-scale predicted longest dim (m)")
    ax.set_title("Auto-scale accuracy vs ground truth (14 objects × 5 pipelines)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_per_pipeline(agg: dict, out_png: Path) -> None:
    pipelines = PIPELINES
    mape_all   = [agg["per_pipeline"][p].get("longest_mape") or 0 for p in pipelines]
    mape_high  = [agg["per_pipeline_high_gt"][p].get("longest_mape") or 0 for p in pipelines]
    x = np.arange(len(pipelines))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, mape_all,  w, label="all 14 objects",     color="#90A4AE")
    ax.bar(x + w/2, mape_high, w, label="high-conf GT only",  color="#42A5F5")
    ax.set_xticks(x); ax.set_xticklabels(pipelines, rotation=20, ha="right")
    ax.set_ylabel("MAPE on longest dim (%)")
    ax.set_title("Auto-scale error by generation pipeline")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for i, v in enumerate(mape_high):
        ax.text(i + w/2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_per_object(rows: list[Row], out_png: Path) -> None:
    """For each object, plot predicted longest across pipelines vs GT line."""
    by_obj: dict[str, list[Row]] = {}
    for r in rows:
        by_obj.setdefault(r.object_id, []).append(r)
    objs = sorted(by_obj.keys())
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(objs))
    gts = [by_obj[o][0].gt_longest_m for o in objs]
    ax.bar(x, gts, color="#37474F", alpha=0.35, label="ground truth")
    colors = {"trellis_rembg":"#4FC3F7","trellis_sam3":"#0288D1","hunyuan_rembg":"#CE93D8","hunyuan_sam3":"#7B1FA2","sam3d_sam3":"#81C784"}
    markers = {"trellis_rembg":"o","trellis_sam3":"s","hunyuan_rembg":"^","hunyuan_sam3":"v","sam3d_sam3":"D"}
    for p in PIPELINES:
        ys = [next((r.pred_longest_m for r in by_obj[o] if r.pipeline == p and r.pred_longest_m is not None), None) for o in objs]
        xs = [i for i, y in enumerate(ys) if y is not None]
        yvals = [y for y in ys if y is not None]
        ax.scatter(xs, yvals, label=p, color=colors[p], marker=markers[p], s=55, edgecolor="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([by_obj[o][0].display_name.split("(")[0].strip()[:18] for o in objs], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Longest dim (m)")
    ax.set_title("Per-object predicted vs ground truth (longest dim)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


def write_report_md(rows: list[Row], agg: dict, out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Auto-scale benchmark — 14 HTX objects × 5 pipelines\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## Headline\n")

    h = agg["high_gt_only"]
    o = agg["overall"]
    lines.append(f"- **High-confidence ground truth (n={h.get('n', 0)} rows)**: longest-dim MAPE = **{h.get('longest_mape', 0):.1f}%**, median error {h.get('longest_median_pct', 0):.1f}%, mean view-IoU {h.get('mean_iou') or 0:.2f}")
    lines.append(f"- **All ground truth (n={o.get('n', 0)} rows)**: longest-dim MAPE = {o.get('longest_mape', 0):.1f}%, {(o.get('frac_within_20pct') or 0)*100:.0f}% of predictions within ±20% of GT, {(o.get('frac_within_50pct') or 0)*100:.0f}% within ±50%")
    lines.append("")
    lines.append("(\"high-confidence GT\" = objects with manufacturer-published or standard-spec dimensions — Mercedes Sprinter, Terrex, Volvo bus, Hyundai sedan, HIMARS, standard police bike.)\n")

    lines.append("## Per-pipeline accuracy\n")
    lines.append("| Pipeline | n | MAPE longest (all) | MAPE longest (high-GT) | mean IoU | within ±20% | within ±50% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for p in PIPELINES:
        a = agg["per_pipeline"][p]; ah = agg["per_pipeline_high_gt"][p]
        if a.get("n", 0) == 0:
            lines.append(f"| {p} | 0 | – | – | – | – | – |"); continue
        lines.append(f"| {p} | {a['n']} | {a.get('longest_mape', 0):.1f}% | "
                     f"{(ah.get('longest_mape') if ah.get('n') else None) or 0:.1f}% | "
                     f"{a.get('mean_iou') or 0:.2f} | "
                     f"{(a.get('frac_within_20pct') or 0)*100:.0f}% | "
                     f"{(a.get('frac_within_50pct') or 0)*100:.0f}% |")
    lines.append("")

    lines.append("## Per-object detail (all pipelines)\n")
    lines.append("| Object | GT longest (m) | GT conf | Pipeline | Pred longest (m) | err % | IoU | conf |")
    lines.append("|---|---:|:---:|---|---:|---:|---:|:---:|")
    by_obj: dict[str, list[Row]] = {}
    for r in rows: by_obj.setdefault(r.object_id, []).append(r)
    for oid in sorted(by_obj.keys()):
        first = True
        for r in sorted(by_obj[oid], key=lambda x: x.pipeline):
            obj_cell = r.display_name if first else ""
            gt_cell  = f"{r.gt_longest_m:.2f}" if first else ""
            gtc_cell = r.gt_confidence if first else ""
            pred = f"{r.pred_longest_m:.2f}" if r.pred_longest_m is not None else "—"
            err  = f"{r.err_longest_pct:.1f}%" if r.err_longest_pct is not None else "—"
            iou  = f"{r.iou:.2f}" if r.iou is not None else "—"
            conf = r.pred_confidence or "—"
            lines.append(f"| {obj_cell} | {gt_cell} | {gtc_cell} | {r.pipeline} | {pred} | {err} | {iou} | {conf} |")
            first = False
    lines.append("")
    lines.append("## Figures\n")
    lines.append("![scatter](scatter.png)\n")
    lines.append("![per_pipeline](per_pipeline.png)\n")
    lines.append("![per_object](per_object.png)\n")
    out_md.write_text("\n".join(lines))


def main():
    gts = load_gt()
    print(f"loaded {len(gts)} ground-truth rows")
    rows: list[Row] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for gt_row in gts:
            for pipeline in PIPELINES:
                t_one = time.time()
                r = run_one(gt_row, pipeline, tmpdir)
                rows.append(r)
                status = "OK" if r.auto_scaled else f"SKIP ({r.reason})"
                err = f"{r.err_longest_pct:.1f}%" if r.err_longest_pct is not None else "-"
                iou = f"{r.iou:.2f}" if r.iou is not None else "-"
                print(f"  {gt_row['object_id']:28s} {pipeline:18s} {status:8s} long={r.pred_longest_m if r.pred_longest_m else 'NA'}  gt={r.gt_longest_m}  err={err:6s} iou={iou:5s} ({time.time()-t_one:.1f}s)")

    print(f"\ntotal {time.time()-t0:.1f}s, {len(rows)} rows")

    csv_path = write_results_csv(rows)
    print(f"wrote {csv_path}")

    agg = aggregates(rows)
    (RESULTS_DIR / "results_summary.json").write_text(json.dumps(agg, indent=2))

    plot_scatter(rows, RESULTS_DIR / "scatter.png")
    plot_per_pipeline(agg, RESULTS_DIR / "per_pipeline.png")
    plot_per_object(rows, RESULTS_DIR / "per_object.png")
    write_report_md(rows, agg, RESULTS_DIR / "report.md")
    print(f"wrote {RESULTS_DIR}/report.md + 3 plots")


if __name__ == "__main__":
    main()
