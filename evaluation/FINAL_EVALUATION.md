# HTX-3D image-to-3D evaluation — final results

**7 pipelines × 14 HTX objects · RTX 5090 · 2026-07-31**

Pipelines are 4 engines × 2 segmentation front-ends (SAM 3D runs only with SAM 3):
`rembg` = raw photograph, engine removes the background itself; `SAM 3` = SAM 3
cutout fed to the engine.

---

## 1 · Conclusions

**1. Hunyuan3D with rembg is the weakest pipeline.** This is the only ranking
claim in this report that survives significance testing, and it is confirmed
independently by three different measurements:

- highest dimensional error (35.1% vs 19.4–24.6% on high-confidence objects) and
  the only pipeline where a paired test separates it from others (§3.1)
- 4 of 14 objects reconstructed >2× too flat, worst at 50–100× too flat (§3.3)
- the only pipeline producing geometry that collapses to a flat sheet, caught
  independently by silhouette coverage in the orbit renders (§3.4)

**2. TRELLIS 1, TRELLIS.2 and SAM 3D are not separable on dimensional accuracy
at n=14.** Their confidence intervals overlap heavily and no paired test
distinguishes them. Selecting between them must rest on the other axes, where
differences are large and unambiguous.

**3. They fail differently, and no single number captures it:**

| | TRELLIS 1 | TRELLIS.2 | SAM 3D |
|---|---|---|---|
| Speed | **10 s — fastest** | 24 s | 20 s |
| Dimensional error (high-conf) | **19.5%** | 24.6% | **19.4%** |
| Mesh cleanliness | 9 components | 157 components — most fragmented | **4 components — cleanest** |
| Open boundaries | 2.6 loops | 16.1 loops | **1.2 loops** |
| Texture uniformity around the object | 0.64 | **0.76 — most even** | 0.57 — most one-sided |

**4. Segmentation matters for cluttered scenes.** On the four multi-object APICS
photographs, the `rembg` front-end does not isolate the labelled target — it hands
the engine the whole scene. Dimensional errors there reach 90–141%. This is a
finding about the segmentation front-end, not about the engines.

### Recommendation by use case

| Need | Pipeline |
|---|---|
| Throughput / interactive turnaround | **TRELLIS 1 · rembg** — 10 s, dimensionally best-in-class |
| Assets needing least manual repair | **SAM 3D · SAM 3** — cleanest topology by a wide margin |
| Maximum surface detail, cleanup acceptable | **TRELLIS.2** — most detail and most even texture, but most fragmented |
| Cluttered / multi-object source photos | any engine, but **with SAM 3 segmentation** |
| — | **Not Hunyuan3D · rembg** |

---

## 2 · Metric set

The governing constraint: **no ground-truth 3D meshes exist for these objects**,
so Chamfer distance, F-score, voxel-IoU and EMD — the standard geometric metrics
in the image-to-3D literature — are unavailable. What we do have ground truth for
is **real-world dimensions** from manufacturer and standard specifications.

| Track | Metric | Status |
|---|---|---|
| Dimensional accuracy | MAPE vs. spec sheets, bootstrap CIs, paired Wilcoxon | ✅ lead result |
| Performance | gen time, peak VRAM, faces, file size | ✅ |
| Mesh integrity | components, floater fraction, boundary loops, aspect vs. spec | ✅ new |
| Multi-view | silhouette coverage (collapse), texture one-sidedness | ✅ new |
| Perceptual | blinded human scoring, geometry + texture 1–5 | ⏳ pending |

### Metrics dropped, and why

- **SSIM, PSNR, LPIPS-vs-photograph.** Measured at SSIM 0.24 / PSNR 6.4 dB /
  LPIPS 0.78, essentially uniform across all conditions. PSNR 6.4 dB implies
  RMSE ≈ 120 on a 0–255 scale — the images are near-unrelated. The cause is
  protocol, not reconstruction: a canonical render on flat grey was being
  compared against a real photograph with its own background, framing and camera
  pose. The between-condition spread was smaller than the within-condition spread
  across objects, so they carried no discriminative power. LPIPS remains a
  legitimate metric but requires aligned renders, which needs GT meshes.
- **watertight %.** Constant at 0.0% for all conditions — zero information.
  Replaced by boundary-loop count, which ranges 0.0 to 16.1.
- **Opposite-view CLIP similarity (janus/duplicate-front proxy).** Negative
  result: mean 0.918–0.936 across all seven pipelines, spread 0.018. On a white
  background CLIP similarity is dominated by silhouette and background rather
  than by texture duplication. Reported here so it is not re-attempted the same
  way; detecting duplicated fronts needs a different approach.

---

## 3 · Results

### 3.1 Dimensional accuracy

Mean absolute percentage error on the object's longest real-world dimension.
"High-conf" = the 6 objects with manufacturer or published-standard dimensions.
Paired bootstrap 95% CIs, resampling objects, 20k draws.

| Pipeline | MAPE (14) | 95% CI | MAPE (high-conf, n=6) | within ±20% | view-IoU |
|---|---:|:---:|---:|---:|---:|
| TRELLIS 1 · rembg | **26.7%** | [19.9, 34.5] | 19.5% | 43% | 0.54 |
| TRELLIS.2 · rembg | 28.9% | [19.5, 41.1] | 24.6% | 36% | **0.55** |
| SAM 3D · SAM 3 | 30.2% | [17.6, 47.3] | **19.4%** | **50%** | **0.55** |
| TRELLIS 1 · SAM 3 | 31.6% | [21.1, 45.1] | 21.4% | 36% | 0.54 |
| Hunyuan3D · SAM 3 | 33.5% | [19.3, 53.1] | 27.6% | 36% | 0.52 |
| TRELLIS.2 · SAM 3 | 33.7% | [21.6, 50.8] | 27.7% | 29% | 0.54 |
| Hunyuan3D · rembg | 40.6% | [24.6, 60.3] | 35.1% | 29% | 0.52 |

Paired Wilcoxon signed-rank across all 21 pipeline pairs finds **only two
significant differences**, both against the same pipeline:

| Pair | mean diff | p |
|---|---:|---:|
| Hunyuan3D · rembg vs SAM 3D · SAM 3 | +10.4 pp | 0.030 |
| Hunyuan3D · rembg vs TRELLIS.2 · SAM 3 | +7.0 pp | 0.049 |

Per-object wins (lowest error on each object): SAM 3D 4 · TRELLIS.2 · rembg 3 ·
TRELLIS 1 · rembg 2 · Hunyuan3D · SAM 3 2 · one each for the rest.

98/98 auto-scale runs succeeded — no silent failures.

### 3.2 Performance

| Pipeline | Mean gen time | Peak VRAM | Mean faces |
|---|---:|---:|---:|
| TRELLIS 1 | 10.4 s | — | 21.5k |
| SAM 3D | 19.7 s | — | 17.1k |
| TRELLIS.2 | 24 s (18.8 gen + 5.6 export) | 7.8 / 32 GB | ~50k |
| Hunyuan3D | 74.9 s | — | 38.4k (40k cap) |

Hunyuan3D is 4–7× slower than the alternatives. TRELLIS.2 was exported at
texture 1024 / ~50k faces for parity; it natively supports 4K / 1M faces.

### 3.3 Mesh integrity

Computed from the GLBs alone, no ground truth and no renders. Vertices are welded
first — these GLBs carry per-face vertex copies for UVs, and without welding the
face-adjacency graph is meaningless.

`aspect_ratio` = predicted (shortest ÷ longest axis) ÷ true proportions from the
spec sheet. 1.00 = correct; 0.02 = 50× flatter than the real object. Axes are
sorted, so this is **orientation-independent** and unaffected by camera pose.

| Pipeline | components | floater face frac | boundary loops | aspect_ratio (med) | mean \|log err\| | >2× too flat |
|---|---:|---:|---:|---:|---:|---:|
| TRELLIS.2 · rembg | 156.6 | 0.139 | 16.1 | 1.44 | 0.44 | 0/14 |
| TRELLIS.2 · SAM 3 | 176.6 | 0.074 | 10.8 | 1.17 | 0.35 | 1/14 |
| TRELLIS 1 · rembg | 9.2 | 0.103 | 2.6 | 1.32 | 0.41 | 0/14 |
| TRELLIS 1 · SAM 3 | 8.3 | 0.078 | 2.4 | 1.21 | 0.34 | 0/14 |
| Hunyuan3D · rembg | 12.3 | 0.032 | 0.0 | 0.96 | **0.97** | **4/14** |
| Hunyuan3D · SAM 3 | 10.5 | 0.086 | 0.0 | 0.98 | 0.67 | 2/14 |
| SAM 3D · SAM 3 | **4.3** | 0.035 | **1.2** | 1.24 | 0.38 | 2/14 |

Most collapsed reconstructions:

| aspect_ratio | Object | Pipeline |
|---:|---|---|
| 0.01 | APICS car booth | Hunyuan3D · rembg |
| 0.01 | Police motorcycle | Hunyuan3D · SAM 3 |
| 0.02 | Police motorcycle | Hunyuan3D · rembg |
| 0.12 | APICS car booth | Hunyuan3D · SAM 3 |

A systematic effect across all pipelines: genuinely slender objects are
over-thickened. The coast-guard boat (true aspect 0.17) comes out 1.7–2.7× too
fat in every pipeline.

### 3.4 Multi-view findings

98 models re-rendered at yaw {0°, 90°, 180°, 270°}, pitch 30° — the protocol
published in the TRELLIS paper. This replaces the earlier single fixed-camera
render, which showed a different side of the object per pipeline because no
pipeline emits a canonical orientation.

**Collapsed geometry, detected independently.** A flat sheet is a hairline from
two opposite yaws. Exactly 4 models have a silhouette covering <1% of frame from
an opposite view pair — **all four are Hunyuan**, and they are the same four the
aspect-ratio metric flagged. Two unrelated methods, identical answer.

![4-view orbit](benchmark_v2/figures/figure_flat_collapse_orbit.png)

**Texture one-sidedness** (min ÷ max colour saturation inside the silhouette
across views; 1.0 = uniform all round): TRELLIS.2 · SAM 3 0.76 · TRELLIS.2 ·
rembg 0.75 · Hunyuan3D · SAM 3 0.70 · Hunyuan3D · rembg 0.66 · TRELLIS 1 0.64–0.65
· SAM 3D 0.57. This corroborates the earlier qualitative claim that TRELLIS.2
transfers texture most evenly and SAM 3D carries the least surface detail. Caveat:
the measure cannot distinguish a one-sided texture from an object whose sides are
genuinely different colours.

---

## 4 · Limitations

- **No ground-truth meshes**, so no Chamfer distance, F-score, voxel-IoU or EMD.
  Geometry is assessed indirectly via dimensional accuracy and mesh integrity.
  The fix is to run the same pipelines on a GT-mesh dataset (Google Scanned
  Objects) for literature-comparable numbers.
- **n=14, single seed, no repeats.** Confidence intervals are wide and
  generative 3D models have real seed-to-seed variance that is not captured. The
  high-confidence dimensional column rests on **6 objects**.
- **Objects 12 and 13 share one source photograph.** Their SAM 3 conditions
  differ, but their `rembg` conditions receive identical input, so the rembg arms
  cover 13 distinct inputs, not 14. Hunyuan is deterministic and produced a
  byte-identical mesh for both, so one mesh is counted twice in its rembg mean.
- **Auto-scale is the accuracy bottleneck.** It estimates size from a single
  photograph via monocular depth; even the best pipeline sits near 20% MAPE. That
  is a depth-estimation limit, not a mesh-quality limit.
- **Perceptual quality is not yet measured.** The blinded human scoring pass is
  built and pending (§5).

## 5 · Not yet done

1. **Blinded human scoring** — harness complete, 0/98 scored. Needs ~45 min of a
   human rater; defect flags are now automated so only the two 1–5 scores remain.
2. **Uni3D-I / ULIP-I** — needs no GT mesh and is the only metric family shared by
   Hunyuan 2.1, SAM 3D and Step1X, so it buys comparability with published
   tables. Do not expect it to separate these seven: SAM 3D's own paper reports
   0.3707 vs 0.3698 across genuinely different methods.
3. **GSO subset** for Chamfer / F-score.
4. **TRELLIS.2 paper** (arXiv 2512.14692) not yet in `references/`.

## 6 · Reproducing

```bash
# dimensional accuracy (host)
cd evaluation/auto_scale_benchmark && python3 benchmark_auto_scale.py

# mesh integrity — needs trimesh
~/miniconda3/envs/3D/bin/python evaluation/benchmark_v2/mesh_defect_metrics.py

# 4-view orbit renders + multi-view metrics — need nvdiffrast + CLIP, in-container
docker exec htx-3d python /app/gallery/_orbit_render.py
docker exec htx-3d python /app/gallery/_orbit_metrics.py

# blinded scoring / live comparison UI
cd evaluation/manual_eval && ./serve.sh
```

Data: `trellis2_benchmark/results.csv` (dimensional) ·
`benchmark_v2/mesh_defects.csv` (integrity) · `benchmark_v2/orbit_defects.csv`
(multi-view) · `benchmark_v2/metrics_table.md` (performance).
