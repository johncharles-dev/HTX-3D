#!/usr/bin/env python3
"""Rebuild _serve/gallery_manifest.json from the live gallery index.

The gallery page is static, so re-run this after generating new models:

    ~/miniconda3/envs/3D/bin/python evaluation/manual_eval/build_gallery_manifest.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GALLERY = Path("/home/cj/HTX-3D/gallery")
OUT = HERE / "_serve" / "gallery_manifest.json"

raw = json.loads((GALLERY / "index.json").read_text())
items = raw if isinstance(raw, list) else raw.get("items", raw.get("models", []))

out = []
for it in items:
    tid = it.get("task_id")
    if not tid or not (GALLERY / tid / "model.glb").exists():
        continue
    auto = it.get("auto_scale") or {}
    dims = auto.get("dimensions_m") or {}
    align = auto.get("view_alignment") or {}
    gen = it.get("generation_time_seconds")
    out.append({
        "id": tid,
        "engine": (it.get("model") or "?")
                  .replace("-image-to-3d", "").replace("-text-to-3d", "·text"),
        # 'sam3' | 'rembg' | None. Only recorded since the field was added, so
        # older generations legitimately have no value.
        "seg": it.get("segmentation"),
        "seed": it.get("seed"),
        "tex": it.get("texture_size"),
        "created": (it.get("created_at") or "")[:19],
        "thumb": (GALLERY / tid / "thumbnail.png").exists(),
        "gen_s": round(gen, 1) if gen else None,
        "longest_m": round(dims["longest_m"], 2) if dims.get("longest_m") else None,
        "iou": round(align["iou"], 3) if align.get("iou") else None,
        "conf": auto.get("confidence"),
    })

out.sort(key=lambda x: x["created"], reverse=True)
OUT.write_text(json.dumps(out))

seg = {}
for o in out:
    seg[o["seg"] or "(not recorded)"] = seg.get(o["seg"] or "(not recorded)", 0) + 1
print(f"{len(out)} models -> {OUT}")
print("  segmentation:", ", ".join(f"{k} {v}" for k, v in sorted(seg.items())))
