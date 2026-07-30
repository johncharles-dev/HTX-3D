"""Harvest manually-created generations out of the gallery into an evaluation set.

The app already stores everything except which *object* each generation is and
which segmentation front-end fed it. This script recovers both and stages the
result so the existing metric suite can run on it.

Two steps.

    # 1. after you have generated, scan the gallery and write a manifest
    python3 harvest_generations.py scan --since 2026-07-30 -o my_eval/manifest.csv

       Groups entries by input-image hash (same photo => same object), reads the
       engine from the gallery index, and infers the segmentation mode from
       whether the stored input image carries a real alpha channel.
       Then YOU fill in object_id / display_name, and the ground-truth dimensions
       for any object that has a published spec. Leave GT blank if there is none —
       an invented dimension corrupts the proportion-accuracy metric.

    # 2. stage it
    python3 harvest_generations.py ingest my_eval/manifest.csv

       Builds my_eval/{data,outputs}/ with hardlinks in the layout the metric
       scripts expect, plus my_eval/ground_truth.csv for the rows you filled in.

Nothing is copied by value and nothing in the gallery is modified.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

GALLERY = Path("/home/cj/HTX-3D/gallery")
INDEX = GALLERY / "index.json"

ENGINE_FROM_MODEL = {
    "trellis-image-to-3d": "trellis",
    "trellis2-image-to-3d": "trellis2",
    "hunyuan-image-to-3d": "hunyuan",
    "sam3d-image-to-3d": "sam3d",
}

FIELDS = ["object_id", "display_name", "pipeline", "engine", "seg_mode", "task_id",
          "image_md5", "image_file", "gen_time_s", "auto_longest_m", "auto_confidence",
          "longest_m", "middle_m", "shortest_m", "gt_confidence", "gt_source"]


def md5(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def seg_mode(img: Path) -> str:
    """'sam3' if the stored input carries a real (non-opaque) alpha channel."""
    try:
        from PIL import Image
        im = Image.open(img)
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            a = im.convert("RGBA").split()[-1]
            return "sam3" if a.getextrema()[0] < 255 else "rembg"
    except Exception:
        pass
    return "rembg"


def cmd_scan(args) -> None:
    items = json.load(open(INDEX))
    items = items if isinstance(items, list) else items.get("items", [])
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    rows, skipped = [], 0
    for it in items:
        if since:                                  # date filter first, so the
            try:                                   # skip count means something
                if datetime.fromisoformat(it["created_at"]) < since:
                    continue
            except Exception:
                continue
        eng = ENGINE_FROM_MODEL.get(it.get("model", ""))
        if eng is None:
            skipped += 1
            continue
        d = GALLERY / it["task_id"]
        img = next((d / f"input_image{e}" for e in (".png", ".jpg", ".webp")
                    if (d / f"input_image{e}").exists()), None)
        glb = d / "model.glb"
        if img is None or not glb.exists():
            skipped += 1
            continue
        a = it.get("auto_scale") or {}
        dims = a.get("dimensions_m") or {}
        sm = seg_mode(img)
        rows.append({
            "object_id": "", "display_name": "",
            "pipeline": f"{eng}_{sm}", "engine": eng, "seg_mode": sm,
            "task_id": it["task_id"], "image_md5": md5(img), "image_file": img.name,
            "gen_time_s": it.get("generation_time_seconds") or "",
            "auto_longest_m": round(dims["longest_m"], 3) if dims.get("longest_m") else "",
            "auto_confidence": a.get("confidence") or "",
            "longest_m": "", "middle_m": "", "shortest_m": "",
            "gt_confidence": "", "gt_source": "",
        })

    # same photo => same object; pre-fill a placeholder id so grouping is visible
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["image_md5"]].append(r)
    for i, (h, group) in enumerate(sorted(by_hash.items(), key=lambda kv: -len(kv[1])), 1):
        for r in group:
            r["object_id"] = f"OBJ{i:02d}"

    rows.sort(key=lambda r: (r["object_id"], r["pipeline"]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} generations, {len(by_hash)} distinct input images "
          f"({skipped} skipped in range: not an image-to-3D task, or files missing)")
    for i, (h, g) in enumerate(sorted(by_hash.items(), key=lambda kv: -len(kv[1])), 1):
        pipes = ", ".join(sorted({r['pipeline'] for r in g}))
        print(f"  OBJ{i:02d}  {len(g)} runs  [{pipes}]  ({g[0]['image_file']}, {h[:8]})")
    print(f"\n-> {out}")
    print("Now fill in object_id / display_name, and longest/middle/shortest_m + "
          "gt_source for any object with a PUBLISHED spec. Leave GT blank otherwise.")


def cmd_ingest(args) -> None:
    rows = list(csv.DictReader(open(args.manifest)))
    root = Path(args.manifest).parent
    data, outs = root / "data", root / "outputs"
    data.mkdir(parents=True, exist_ok=True); outs.mkdir(parents=True, exist_ok=True)

    def link(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            import shutil; shutil.copy2(src, dst)

    n_glb = 0
    seen_img = set()
    for r in rows:
        oid, pipe, tid = r["object_id"].strip(), r["pipeline"].strip(), r["task_id"].strip()
        if not oid or not pipe:
            print(f"  skip {tid}: object_id or pipeline blank"); continue
        d = GALLERY / tid
        glb = d / "model.glb"
        if not glb.exists():
            print(f"  skip {tid}: no model.glb"); continue
        link(glb, outs / oid / pipe / "model.glb"); n_glb += 1
        if oid not in seen_img:
            img = d / r["image_file"]
            if img.exists():
                link(img, data / oid / f"original{img.suffix}")
                seen_img.add(oid)

    gt = [r for r in rows if r["longest_m"].strip()]
    seen = set(); gt_rows = []
    for r in gt:
        if r["object_id"] in seen:
            continue
        seen.add(r["object_id"])
        img = r["image_file"]
        gt_rows.append({
            "object_id": r["object_id"], "input_filename": f"original{Path(img).suffix}",
            "display_name": r["display_name"] or r["object_id"],
            "longest_m": r["longest_m"], "middle_m": r["middle_m"],
            "shortest_m": r["shortest_m"],
            "gt_confidence": r["gt_confidence"] or "low",
            "source": r["gt_source"] or "unspecified",
        })
    if gt_rows:
        with open(root / "ground_truth.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gt_rows[0].keys()))
            w.writeheader(); w.writerows(gt_rows)

    print(f"staged {n_glb} meshes across {len(seen_img)} objects -> {outs}")
    print(f"ground truth for {len(gt_rows)} objects -> {root/'ground_truth.csv'}"
          if gt_rows else "no ground truth filled in — GT-free metrics only")
    print(f"""
Next:
  # mesh integrity (needs trimesh)
  ~/miniconda3/envs/3D/bin/python benchmark_v2/mesh_defect_metrics.py \\
        --outputs {outs} --gt {root/'ground_truth.csv'} --exclude ''
  # 4-view orbit renders + multi-view metrics: hardlink {outs} into gallery/ first,
  # then the two in-container scripts (see FINDINGS_AND_HANDOVER.md §9)""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="read the gallery, write a manifest to fill in")
    s.add_argument("--since", help="ISO date, e.g. 2026-07-30 (UTC)")
    s.add_argument("-o", "--output", default="harvested/manifest.csv")
    s.set_defaults(func=cmd_scan)
    i = sub.add_parser("ingest", help="stage a filled-in manifest for the metric suite")
    i.add_argument("manifest")
    i.set_defaults(func=cmd_ingest)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
