"""Multi-view defect metrics — runs INSIDE the htx-3d container (CLIP + torch).

Computes the two rubric proxies that a single view cannot see, from the 4-view
orbit renders in /app/gallery/_orbit_renders:

  janus_duplicate      CLIP cosine similarity between OPPOSITE views (0 vs 180,
                       90 vs 270). Anomalously high => the model duplicated the
                       photographed front onto the back.
  front_only_texture   colour saturation and chroma variance inside the silhouette,
                       compared across views. A front-only texture leaves the rear
                       views flat and desaturated.

Also emits per-view silhouette coverage, which is needed to avoid a false janus
flag: a collapsed flat sheet is a thin line from two opposite yaws, and two thin
lines are trivially similar. Views below MIN_COVERAGE are excluded from the janus
comparison and reported as degenerate instead.

    docker exec htx-3d python /app/gallery/_orbit_metrics.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

REN = "/app/gallery/_orbit_renders"
OUT = "/app/gallery/_orbit_metrics.csv"
YAWS = [0, 90, 180, 270]
MIN_COVERAGE = 0.01       # below this the view is a degenerate sliver
WHITE = 245               # background is pure white; anything darker is foreground


def masked_stats(path):
    """Silhouette coverage, mean saturation and chroma variance inside the mask."""
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    fg = (img.max(axis=2) < WHITE / 255.0) | (np.ptp(img, axis=2) > 0.04)
    cov = float(fg.mean())
    if cov < 1e-6:
        return cov, 0.0, 0.0
    px = img[fg]
    mx, mn = px.max(axis=1), px.min(axis=1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return cov, float(sat.mean()), float(px.std(axis=0).mean())


def main():
    files = sorted(f for f in os.listdir(REN) if f.endswith(".png"))
    models = defaultdict(dict)
    for f in files:
        stem, yaw = f[:-4].rsplit("__yaw", 1)
        models[stem][int(yaw)] = os.path.join(REN, f)
    print(f"{len(models)} models", flush=True)

    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    rows = []
    for i, (stem, views) in enumerate(sorted(models.items()), 1):
        if len(views) != len(YAWS):
            print(f"  skip {stem}: {len(views)} views"); continue
        obj, pipe = stem.split("__", 1)

        stats = {y: masked_stats(views[y]) for y in YAWS}
        cov = {y: stats[y][0] for y in YAWS}
        sat = {y: stats[y][1] for y in YAWS}
        var = {y: stats[y][2] for y in YAWS}

        with torch.no_grad():
            imgs = [Image.open(views[y]).convert("RGB") for y in YAWS]
            inp = proc(images=imgs, return_tensors="pt")
            inp = {k: v.cuda() for k, v in inp.items()}
            fe = clip.get_image_features(**inp)
            fe = fe / fe.norm(dim=-1, keepdim=True)

        idx = {y: k for k, y in enumerate(YAWS)}
        pairs, degenerate = [], []
        for a, b in ((0, 180), (90, 270)):
            if cov[a] < MIN_COVERAGE or cov[b] < MIN_COVERAGE:
                degenerate.append(f"{a}/{b}")
                continue
            pairs.append(float((fe[idx[a]] @ fe[idx[b]]).item()))

        # texture one-sidedness: worst view vs best view inside the silhouette
        live = [y for y in YAWS if cov[y] >= MIN_COVERAGE]
        sat_ratio = (min(sat[y] for y in live) / max(max(sat[y] for y in live), 1e-6)
                     if live else None)
        var_ratio = (min(var[y] for y in live) / max(max(var[y] for y in live), 1e-6)
                     if live else None)

        rows.append({
            "object_id": obj, "pipeline": pipe,
            "opposite_view_clip_max": round(max(pairs), 4) if pairs else None,
            "opposite_view_clip_mean": round(float(np.mean(pairs)), 4) if pairs else None,
            "degenerate_view_pairs": ";".join(degenerate) or "",
            "min_coverage": round(min(cov.values()), 4),
            "max_coverage": round(max(cov.values()), 4),
            "sat_ratio": round(sat_ratio, 4) if sat_ratio is not None else None,
            "chroma_var_ratio": round(var_ratio, 4) if var_ratio is not None else None,
        })
        if i % 20 == 0 or i == len(models):
            print(f"  {i}/{len(models)}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
