# Evaluation findings, known issues, and handover notes

Companion to `FINAL_EVALUATION.md`, which reports *results*. This file records
**how those results were arrived at, what is known to be wrong, and what a
successor needs to know.** Written for handover to HTX and for the end-term
report.

Last updated 2026-07-30.

---

## 1 · Findings about `auto_scale` (our own algorithm)

### 1.1 The azimuth sweep improves alignment but not accuracy — verified negative result

`auto_scale.find_best_view()` (`app/services/auto_scale.py:323`) tries the mesh's
canonical orientation first and, **only if silhouette IoU < 0.5**, sweeps
3 elevations × 7 azimuths (21 extra renders) and keeps whichever rotation
maximises silhouette IoU.

Paired test on the 40 rows where the sweep actually fired (re-run with the sweep
disabled, same meshes and images):

| | Sweep (`multi_view`) | Canonical forced |
|---|---:|---:|
| Silhouette IoU | **0.481** | 0.339 |
| MAPE | 43.8% | 51.7% |
| Median error | 33.5% | 33.2% |
| Sweep better on | 19/40 rows (48%) | |
| Paired Wilcoxon | **p = 0.567** | |

Excluding APICS (n=19): 30.5% vs 31.4%, 9/19, **p = 0.922**.

**Interpretation.** The sweep does what it is written to do — it raises IoU from
0.34 to 0.48 — but that gain does not transfer to metric size accuracy. Win rate
is a coin flip. The apparent mean advantage comes almost entirely from the APICS
car booth, which is excluded from reported results.

**Why it matters.** Maximising 2D silhouette overlap is not a proxy for getting
3D scale right. This is the most likely reason auto-scale plateaus near 20–25%
MAPE: **the optimisation target is misaligned with the goal.**

**Recommended action.** Either remove the sweep (saves up to 21 renders per
object at no measured accuracy cost) or change its objective. Do not simply force
canonical everywhere — that is marginally *worse*, because those rows are the
ones where canonical alignment already failed.

Reproduce: `docker exec htx-3d python /app/gallery/_align_test.py`
→ `gallery/_align_test_results.csv`

### 1.2 Alignment quality is the limiting factor, and it is viewpoint-driven

Across the 70 reported rows, mean silhouette IoU is **0.563** (range 0.40–0.67).
At 0.56 roughly half the silhouette area disagrees — the mesh never lines up well
with the photograph, for any pipeline. Between-pipeline IoU spread is 0.02–0.04,
i.e. negligible.

`corr(IoU, dimensional error) = −0.313` — better alignment gives lower error.
Per object, the two worst-aligned are the two worst-scoring:

| Object | IoU | Error |
|---|---:|---:|
| Security guard house (boxy, square-on photo) | 0.65 | 13.3% |
| SBS double-decker bus | 0.58 | 14.5% |
| EOD robot (irregular, oblique photo) | 0.47 | 44.9% |
| HIMARS launcher | 0.53 | 46.1% |

**This is a limitation of the measurement, not of the generations.** It is the
mechanism behind "the instrument is as noisy as the effect", and it is why the
dimensional track cannot rank the engines.

### 1.3 The instrument is as noisy as the effect being measured

Auto-scale's own high-confidence MAPE is ~24.6%. The best pipeline measures
~19.4%. **The measuring error is the same magnitude as the quantity measured.**
Any conclusion that requires separating pipelines by a few percentage points of
dimensional error is therefore unsupportable with this instrument.

### 1.4 No engine emits metric scale — this is why auto-scale exists

Measured raw output extents, before any scaling, across all 98 meshes:

| Engine | Largest bbox dimension | Convention |
|---|---:|---|
| TRELLIS.2 | 0.998 (0.936–1.002) | unit cube |
| TRELLIS 1 | 1.001 | unit cube |
| SAM 3D | 0.998 | unit cube, best-centred |
| Hunyuan3D | 1.951–1.978 | [-1, 1] — 2× larger |

The TRELLIS.2 paper confirms unit-cube normalisation, but only as an *evaluation*
step ("Prior to any metric calculation, all ground-truth and predicted meshes are
normalized to fit within a unit cube"). **No paper of the four claims metric or
real-world scale.** Auto-scale is not post-processing; it supplies a capability
absent from the entire model literature.

Hunyuan's 2× convention does not bias results — auto-scale rescales from the
bounding box — but it matters to anyone importing raw GLBs.

---

## 2 · Corrections to earlier claims

Recorded so they are not repeated:

| Earlier claim | Correction |
|---|---|
| "`multi_view` alignment is ~80% worse than `canonical` (43.8% vs 24.2%)" | **Confounded by construction.** The sweep only runs when canonical IoU < 0.5, so those rows are the hard cases by definition. The comparison measured difficulty, not method. See §1.1 for the valid paired test. |
| "Forcing canonical could cut MAPE from 32% to 24%" | **Wrong.** Forced canonical is slightly *worse* (51.7% vs 43.8% on the affected rows). |
| "Live gallery entries clustering at 0.918 m indicate an auto-scale fallback" | **Wrong.** Those six generations share a byte-identical input photo. Auto-scale derives absolute scale mainly from monocular depth on the photograph, so identical photo → identical estimate is correct behaviour. |
| "Hunyuan's off-centre meshes may explain its worse accuracy" | **Not supported.** As a fraction of object size the offsets are small (0.002–0.035); corr with view-IoU is −0.10, with error −0.02. |
| "The midterm report's Hunyuan+rembg 136.3% figure looks wrong" | **The report is right.** 136.3% is the prior-OFF baseline, which is the correct comparison for isolating the prior's contribution. My check used the wrong subset. |

---

## 3 · Midterm report verification (2026-07-30)

Every auto-scale figure in `docs/3rd_term_midterm_report_content.md` was checked
against `evaluation/auto_scale_benchmark/*/results_summary.json`:

| Claim | Data | |
|---|---|---|
| Baseline 77.5% / 38.4% / 53% / 37% | identical | ✓ |
| Prior OFF 63.7% / 29.8% / 57% | identical | ✓ |
| Shipped 24.6% / 19.8% / 90% / 50% | identical | ✓ |
| n = 30 high-confidence | 30 | ✓ |
| TRELLIS+rembg 40.5% → 19.6% | identical | ✓ |
| Prior contributes 39.1 of 52.9 pts (~74%) | arithmetic checks | ✓ |
| Hunyuan+rembg 136.3% → 35.1% | identical (prior-OFF baseline) | ✓ |

**The midterm submission stands.** The shipped 24.6% also reproduces under an
independent re-scoring on 2026-07-30 (25.0% over 42 high-confidence rows,
7 pipelines).

Two items post-date the report and belong in the end-term version, as additions
rather than corrections: the sweep result (§1.1) — line 75 lists "multi-view IoU
search" as a component, which is accurate as description but should not be
credited with accuracy gains — and the explanation for the ~20–25% floor (§1.2).

---

## 4 · How the model papers evaluate (verified against the PDFs)

All four PDFs are in `references/`; extract with pymupdf and grep with `-a`.

| Paper | Protocol |
|---|---|
| **TRELLIS** | Toys4k (held out). Chamfer 0.0083, F-score 0.9999, PSNR 32.74, LPIPS 0.025, normal-map PSNR-N 36.11. Renders yaw {0,90,180,270}, pitch 30°. User study: 94.5% win rate on image-to-3D. |
| **TRELLIS.2** | Mesh Distance, Chamfer, F-score computed two ways (point-cloud and continuous-mesh), τ=1e-6, 1M points from 100 depth maps. PSNR/LPIPS on **normal maps**. Unit-cube normalisation first. Views yaw {30,120,210,300}, pitch 30°, radius 10, FoV 6°. |
| **SAM 3D** | Own benchmark **SA-3DAO**: 1,000 image–mesh pairs modelled by professional artists ("expert human upper bound"), plus ISO3D and Aria Digital Twin. Chamfer, voxel-IoU, EMD, F-score after per-pair **ICP**. Headline is **human preference, ≥5:1 win rate** — "preference" occurs 74 times. |
| **Hunyuan3D 2.1** | **Zero** occurrences of Chamfer, F-score, PSNR or SSIM. Entire shape evaluation is one table: ULIP-T/I, Uni3D-T/I (ULIP-I 0.1395 vs TRELLIS 0.1267). Framed as a tutorial/system report. |

**Implications for our protocol.** Three of four rely on ground-truth meshes,
which we do not have — that is why our protocol differs, not an oversight.
SAM 3D is the closest precedent for our situation: facing the same absence of GT
for real photographs, they made human preference primary. That is published
justification for treating blinded human scoring as a primary axis.

**Nobody evaluates real-world dimensional accuracy.** Our auto-scale track
measures something the literature ignores entirely.

**Two cheap adoptions for the end-term report:** ULIP-I/Uni3D-I (no GT needed,
the only metric family shared by Hunyuan, SAM 3D and Step1X, gives comparability
with their tables) and **PSNR/LPIPS on normal maps** — the correctly-formed
version of the image metric we had to drop, since normal maps are
texture-independent and background-free. Both need GT meshes for the latter, so
it is gated behind a GSO run.

---

## 5 · Research-integrity notes to carry forward

**Blinding leak, disclosed.** While building the reveal toggle for the scoring
tool, an assistant session printed `key.json` for **object 01 only**, exposing all
seven slot→pipeline mappings for that object (A=trellis2_rembg, B=hunyuan_sam3,
C=trellis_sam3, D=trellis2_sam3, E=trellis_rembg, F=sam3d_sam3, G=hunyuan_rembg).
Objects 02–14 were never displayed. Separately, on object 05 a likely-Hunyuan slot
was inferred from a flat-sheet visual signature.

Consequence: the **human** scoring pass is unaffected (the rater never saw the
key). If a VLM-as-judge track is ever run, exclude object 01 and disclose this.
`analyze_manual.py --pattern 'vlm_scores_*.csv'` writes a separate
`vlm_judge_report.md` and never pools model scores with human ones.

**APICS exclusion cost us the significance, and that is stated.** Excluding the
four multi-object scenes removed the only two pairs that reached p<0.05 (both
against Hunyuan·rembg, because those scenes are where it fails hardest). Both
readings are reported in `FINAL_EVALUATION.md` rather than the more flattering
one. The exclusion criteria were chosen independently of outcome: the target
cannot be isolated by background removal, and the ground truth is estimated
rather than published.

**Ground truth must never be invented.** `ground_truth.csv` cites a source per
object. Objects without a published dimension belong in the `low` tier or out of
the file entirely — an estimated ratio silently corrupts the proportion-accuracy
metric, which is what the Hunyuan conclusion now rests on.

---

## 6 · Known bugs

**Hunyuan + `target_face_count` fails.** `app/services/hunyuan.py:686`
`_decimate_mesh()` loads the GLB into pymeshlab and calls `save_current_mesh()`
on a `.glb` — pymeshlab has no GLB writer, so it raises
`Unknown format for save: glb`. Any Hunyuan generation with `target_face_count > 0`
dies at export.

**This affects the main tool**, not just evaluation: `frontend/src/api/client.ts`
sends `target_face_count` for Hunyuan whenever `settings.targetFaceCount > 0`.
The benchmark avoided it by leaving Hunyuan at its own 40k cap.

Fix options: decimate through trimesh (which can write GLB), or skip decimation
for `.glb` with a warning instead of raising. Not yet applied — the backend is
unchanged.

**Fixed-camera renders show different sides per pipeline.**
`benchmark_v2/render_batch.py:68` fixes the camera on +Z and `make_mvp()` applies
only centring and uniform scale — no rotation. Since no engine emits a canonical
orientation, each pipeline is rendered from whatever direction it happened to
produce. Superseded for scoring by the interactive 3D viewer, and for metrics by
the 4-view orbit renders, but the single-view renders in
`trellis2_benchmark/renders/` still carry this defect.

---

## 7 · Metrics computed and deliberately not reported

| Metric | Why dropped |
|---|---|
| SSIM / PSNR / LPIPS vs source photograph | PSNR 6.4 dB ⇒ RMSE ≈ 120 on 0–255. Measures background, framing and pose mismatch, not reconstruction. Between-condition spread smaller than within-condition spread across objects. LPIPS is legitimate but needs aligned renders, which needs GT meshes. |
| `watertight %` | Constant 0.0% for all conditions. Replaced by boundary-loop count (0.0–11.0). |
| Opposite-view CLIP similarity (janus/duplicate-front proxy) | Negative result: 0.918–0.936 across all seven pipelines, spread 0.018. On a white background CLIP similarity is dominated by silhouette and background. Detecting duplicated fronts needs a different approach. |

Reporting these would have implied precision we do not have. Stating the drops
with reasons is a stronger position than presenting uninformative numbers.

---

## 8 · Outstanding work

**Needs a human:** blinded perceptual scoring, objects 01–10, ~50 min.
Tool at `evaluation/manual_eval/` → `./serve.sh`.

**Needs GPU time:** seed-variance run (quantifies whether engine differences
exceed re-running one engine at a different seed); same-mesh/many-photo variance
(isolates instrument noise using existing GT — 25 Terrex, 13 bus, 8 HIMARS photos
already on disk).

**Needs sourcing work:** expanding the high-confidence set. 125 distinct object
types exist in `~/HTX-3D_samples/`; the constraint is a published dimension per
object, not images. The headline currently rests on **6 objects**.

**Larger:** GSO subset for real Chamfer / F-score — the actual fix for the
shape-accuracy gap. Uni3D-I / ULIP-I for comparability. Normal-map PSNR/LPIPS.

---

## 9 · Reproduction

```bash
# dimensional accuracy (host)
cd evaluation/auto_scale_benchmark && python3 benchmark_auto_scale.py

# mesh integrity — needs trimesh; --exclude filters the aggregate
~/miniconda3/envs/3D/bin/python evaluation/benchmark_v2/mesh_defect_metrics.py

# 4-view orbit renders + multi-view metrics (in-container: nvdiffrast + CLIP)
docker exec htx-3d python /app/gallery/_orbit_render.py
docker exec htx-3d python /app/gallery/_orbit_metrics.py

# alignment-sweep paired test (in-container)
docker exec htx-3d python /app/gallery/_align_test.py

# scoring / live-compare / results UI
cd evaluation/manual_eval && ./serve.sh
```

Data files: `trellis2_benchmark/results.csv` (dimensional, 98 rows) ·
`benchmark_v2/mesh_defects.csv` (integrity, 98) ·
`benchmark_v2/orbit_defects.csv` (multi-view, 98) ·
`benchmark_v2/metrics_table.md` (performance) ·
`auto_scale_benchmark/results{,_tier1,_noprior}/` (auto-scale ablation) ·
`gallery/_align_test_results.csv` (sweep test).

Image-heavy directories are gitignored and exist only on this machine:
`~/HTX-3D_samples/` (949 MB source pool), `gallery/` (6.8 GB),
`benchmark_v2/{data,outputs}/`, all render and montage directories.
**None of it is backed up.**
