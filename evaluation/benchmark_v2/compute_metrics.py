"""Phase 3 — metric computation for the 5-condition x 14-object benchmark.

Computes per-record metrics and per-condition aggregates from
benchmark_v2/results.json + outputs/<id>/<cond>/model.glb.

Phase 3a (host-side, no rendering): mesh + timing metrics.
Phase 3b (after render): SSIM / PSNR / LPIPS / CLIP image metrics.

Usage:
    python compute_metrics.py --phase 3a            # mesh + timing only
    python compute_metrics.py --phase 3b            # image metrics (requires renders)
    python compute_metrics.py --phase all
"""

import os
import json
import argparse
from pathlib import Path
from statistics import mean, median

import numpy as np
import trimesh
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "results.json"
DATA_DIR = SCRIPT_DIR / "data"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
RENDERS_DIR = Path("/home/cj/HTX-3D/gallery/_bench_renders")

METRICS_PATH = SCRIPT_DIR / "metrics.json"
AGG_PATH = SCRIPT_DIR / "metrics_aggregate.json"
TABLE_MD_PATH = SCRIPT_DIR / "metrics_table.md"


def mesh_metrics(glb_path: Path) -> dict:
    """Per-GLB intrinsic mesh metrics."""
    try:
        mesh = trimesh.load(str(glb_path), force="mesh")
    except Exception as e:
        return {"error": str(e)}
    bbox = mesh.bounds
    dims = bbox[1] - bbox[0]
    out = {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "surface_area": round(float(mesh.area), 4),
        "bbox_x": round(float(dims[0]), 4),
        "bbox_y": round(float(dims[1]), 4),
        "bbox_z": round(float(dims[2]), 4),
        "min_axis_dim": round(float(dims.min()), 4),
        "file_size_mb": round(os.path.getsize(glb_path) / (1024 * 1024), 3),
    }
    try:
        areas = mesh.area_faces
        out["degenerate_faces"] = int(np.sum(areas < 1e-10))
    except Exception:
        pass
    return out


def find_render(obj_id: str, condition: str) -> Path | None:
    p = RENDERS_DIR / f"{obj_id}__{condition}.png"
    return p if p.exists() else None


def find_input_image(obj_id: str) -> Path | None:
    obj_dir = DATA_DIR / obj_id
    for f in obj_dir.iterdir():
        if f.stem == "original":
            return f
    return None


def load_resize(path: Path, size=512) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)


def compute_image_metrics(records: list, lpips_model, clip_model, clip_proc) -> None:
    """Augment each record with image metrics. Mutates the records list."""
    import torch
    from torchvision import transforms
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    lpips_tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    for rec in records:
        if rec.get("status") != "completed":
            continue
        obj_id = rec["obj_id"]
        cond = rec["condition"]
        render = find_render(obj_id, cond)
        original = find_input_image(obj_id)
        if render is None or original is None:
            rec["image_metrics_error"] = f"missing render={render is None} input={original is None}"
            continue

        ref = load_resize(original)
        ren = load_resize(render)

        ssim_v = float(ssim(ref, ren, channel_axis=2, data_range=255))
        psnr_v = float(psnr(ref, ren, data_range=255))

        with torch.no_grad():
            t1 = lpips_tx(ref).unsqueeze(0).cuda()
            t2 = lpips_tx(ren).unsqueeze(0).cuda()
            lpips_v = float(lpips_model(t1, t2).item())

        with torch.no_grad():
            inputs = clip_proc(images=[Image.fromarray(ref), Image.fromarray(ren)], return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            clip_v = float((feats[0] @ feats[1]).item())

        rec["image_metrics"] = {
            "ssim": round(ssim_v, 4),
            "psnr": round(psnr_v, 2),
            "lpips": round(lpips_v, 4),
            "clip": round(clip_v, 4),
        }
        print(f"  {obj_id:32s} {cond:15s} SSIM={ssim_v:.3f} PSNR={psnr_v:5.2f} LPIPS={lpips_v:.3f} CLIP={clip_v:.3f}")


def aggregate(records: list) -> dict:
    """Per-condition aggregate over completed records."""
    by_cond: dict[str, list] = {}
    for r in records:
        if r.get("status") != "completed":
            continue
        by_cond.setdefault(r["condition"], []).append(r)

    agg: dict[str, dict] = {}
    keys_mesh = ["vertices", "faces", "surface_area", "file_size_mb",
                 "bbox_x", "bbox_y", "bbox_z", "min_axis_dim", "degenerate_faces"]
    keys_img = ["ssim", "psnr", "lpips", "clip"]

    for cond, recs in by_cond.items():
        a: dict = {"n": len(recs)}
        gen_times = [r["generation_time_s"] for r in recs if r.get("generation_time_s") is not None]
        if gen_times:
            a["gen_time_s_mean"] = round(mean(gen_times), 2)
            a["gen_time_s_median"] = round(median(gen_times), 2)
        for k in keys_mesh:
            vals = [r.get("mesh_metrics", {}).get(k) for r in recs if r.get("mesh_metrics", {}).get(k) is not None]
            if vals:
                a[f"{k}_mean"] = round(mean(vals), 3)
        watertight = [r.get("mesh_metrics", {}).get("watertight") for r in recs]
        watertight = [w for w in watertight if w is not None]
        if watertight:
            a["watertight_pct"] = round(100 * sum(watertight) / len(watertight), 1)
        # Image metrics if present
        for k in keys_img:
            vals = [r.get("image_metrics", {}).get(k) for r in recs if r.get("image_metrics", {}).get(k) is not None]
            if vals:
                a[f"{k}_mean"] = round(mean(vals), 4)
        agg[cond] = a
    return agg


def emit_markdown_tables(agg: dict, records: list) -> str:
    """Emit human-readable tables ready for the report."""
    lines: list[str] = []
    cond_order = ["trellis_rembg", "trellis_sam3", "hunyuan_rembg", "hunyuan_sam3", "sam3d_sam3"]
    cond_order = [c for c in cond_order if c in agg]

    # Combined table: gen time + face count + image metrics + watertight
    lines.append("## Table 6.5.1 — Per-condition mean metrics (14 objects)\n")
    has_img = any("ssim_mean" in agg[c] for c in cond_order)
    headers = ["Condition", "n", "Gen time (s)", "Faces", "Verts", "File MB", "Watertight %", "Min axis (m)"]
    if has_img:
        headers += ["SSIM ↑", "PSNR ↑", "LPIPS ↓", "CLIP ↑"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for c in cond_order:
        a = agg[c]
        row = [
            c,
            str(a.get("n", "—")),
            f"{a.get('gen_time_s_mean', '—')}",
            f"{int(a['faces_mean']):,}" if "faces_mean" in a else "—",
            f"{int(a['vertices_mean']):,}" if "vertices_mean" in a else "—",
            f"{a.get('file_size_mb_mean', '—')}",
            f"{a.get('watertight_pct', '—')}",
            f"{a.get('min_axis_dim_mean', '—')}",
        ]
        if has_img:
            row += [
                f"{a.get('ssim_mean', '—')}",
                f"{a.get('psnr_mean', '—')}",
                f"{a.get('lpips_mean', '—')}",
                f"{a.get('clip_mean', '—')}",
            ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-object full breakdown
    lines.append("## Per-object detail\n")
    obj_ids = sorted({r["obj_id"] for r in records})
    cols = ["object"]
    for c in cond_order:
        cols += [f"{c} faces", f"{c} MB"]
        if has_img:
            cols += [f"{c} SSIM", f"{c} CLIP"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    by_id_cond = {(r["obj_id"], r["condition"]): r for r in records}
    for oid in obj_ids:
        row = [oid]
        for c in cond_order:
            r = by_id_cond.get((oid, c), {})
            mm = r.get("mesh_metrics", {})
            im = r.get("image_metrics", {})
            row.append(str(mm.get("faces", "—")))
            row.append(str(mm.get("file_size_mb", "—")))
            if has_img:
                row.append(str(im.get("ssim", "—")))
                row.append(str(im.get("clip", "—")))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def emit_table_6_5_2(manifest_path: Path) -> str:
    """SAM 3 prompt-mode usage — derived from manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    counts = {"text-only": 0, "text+points": 0, "manual (UI)": 0, "other": 0}
    per_obj = []
    for o in manifest["objects"]:
        prompt = o.get("prompt", {})
        mode = prompt.get("mode", "")
        text = prompt.get("text", "")
        # Best-effort classification — manifest may have evolved
        if mode == "text+points" or "points" in prompt:
            label = "text+points"
        elif mode == "manual" or text == "" and mode == "":
            label = "manual (UI)"
        elif mode == "text" or text:
            label = "text-only"
        else:
            label = "other"
        counts[label] = counts.get(label, 0) + 1
        per_obj.append((o["id"], label, text or mode or "—"))

    lines = ["## Table 6.5.2 — SAM 3 prompt-mode usage (14 objects)\n"]
    lines.append("| Mode | Count | Share |")
    lines.append("|---|---:|---:|")
    total = sum(counts.values())
    for k, v in counts.items():
        if v:
            lines.append(f"| {k} | {v} | {round(100*v/total, 1)}% |")
    lines.append("")
    lines.append("### Per-object\n")
    lines.append("| Object | Mode | Prompt |")
    lines.append("|---|---|---|")
    for oid, mode, text in per_obj:
        lines.append(f"| {oid} | {mode} | {text} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["3a", "3b", "all"], default="all")
    args = p.parse_args()

    with open(RESULTS_PATH) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    # Phase 3a — mesh metrics for every completed record
    print("\n=== Phase 3a — mesh metrics ===")
    for rec in records:
        if rec.get("status") != "completed":
            continue
        glb = OUTPUTS_DIR / rec["obj_id"] / rec["condition"] / "model.glb"
        if not glb.exists():
            rec["mesh_metrics"] = {"error": "missing GLB"}
            continue
        rec["mesh_metrics"] = mesh_metrics(glb)
    print(f"  ✓ mesh metrics computed")

    # Phase 3b — image metrics
    if args.phase in ("3b", "all"):
        print("\n=== Phase 3b — image metrics ===")
        # Verify renders exist
        missing = []
        for rec in records:
            if rec.get("status") != "completed":
                continue
            if not find_render(rec["obj_id"], rec["condition"]):
                missing.append(f"{rec['obj_id']}/{rec['condition']}")
        if missing:
            print(f"  ⚠ {len(missing)} renders missing — phase 3b will skip those")
            print(f"  hint: render via container first (see render_batch.py)")

        if len(missing) < len([r for r in records if r.get("status") == "completed"]):
            print("  Loading LPIPS (AlexNet)...")
            import lpips
            import torch
            torch.cuda.set_device(0)
            lpips_model = lpips.LPIPS(net="alex").cuda().eval()
            print("  Loading CLIP (ViT-B/32)...")
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
            clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("  Computing per-record image metrics...")
            compute_image_metrics(records, lpips_model, clip_model, clip_proc)

    # Aggregate + write
    agg = aggregate(records)
    with open(METRICS_PATH, "w") as f:
        json.dump(records, f, indent=2)
    with open(AGG_PATH, "w") as f:
        json.dump(agg, f, indent=2)

    table_md = emit_markdown_tables(agg, records)
    table_md += "\n\n" + emit_table_6_5_2(SCRIPT_DIR / "manifest.json")
    with open(TABLE_MD_PATH, "w") as f:
        f.write(table_md)

    print(f"\n✓ {METRICS_PATH}")
    print(f"✓ {AGG_PATH}")
    print(f"✓ {TABLE_MD_PATH}")
    print("\n--- aggregate ---")
    for cond, a in agg.items():
        print(f"  {cond}: {a}")


if __name__ == "__main__":
    main()
