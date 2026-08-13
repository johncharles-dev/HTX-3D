# Image-to-3D Model Conversion Pipeline — 3rd Term Mid-Term Report (content to merge)

**Author:** Lourduraj John Charles — Student ID 1011136
**Programme:** Master of Science in Design and AI for Enterprise (MDAI-E), Singapore University of Technology and Design (SUTD)
**Placement:** HTX Singapore — Internship
**Report period:** May – July 2026 (mid-term of the 3rd term). Internship week ~35 of 48.

> **Note on how to use this file.** This is the *new and updated* content for the 3rd Term Mid-Term Report, written in the same style, tone, and level of rigour as the 2nd Term Final Report. It is intended to be merged into the existing report. Section numbers below are indicative and assume continuation of the 2nd-term structure; renumber as needed when combining. Sections marked **(update)** replace/extend an existing 2nd-term section; sections marked **(new)** are additions. A consolidated list of *new* references to fold into the References section is given at the end.

---

## 0. Report Scope and Framing (new — short opener for the mid-term)

The 2nd Term Final Report (April 2026) delivered the project from a two-engine evaluation harness to a complete three-engine operator workflow: TRELLIS, Hunyuan3D-2.1, and SAM 3D Objects, with SAM 3 interactive segmentation, mesh post-processing, PBR material editing, gallery management, and a polished web UI. That report also enumerated the failure modes and open limitations of single-image 3D reconstruction, and set out the planned work for the 3rd term.

This mid-term report documents progress against that plan during the first half of the 3rd term. The headline addition is a **metric auto-scaling** capability that closes one of the specific limitations called out as unsolved in the 2nd-term report — *scale ambiguity*, the fact that a single photograph carries no absolute size information, so every generated asset previously arrived in the simulation at an arbitrary "unit" scale requiring manual sizing. A second addition, **logo-on-surface post-processing**, responds to a direct HTX request to place company/agency logos onto generated product models so the old logo is hidden and the new one reads as native to the surface. The system was also **deployed and demonstrated live to HTX** on the RTX 5090 during the term. Multi-image fusion remains work-in-progress and is reported as such.

---

## 1. Recap: Status at the 2nd-Term Final Report (new — one-paragraph bridge)

At the close of the 2nd term the pipeline supported single-image (and experimental multi-image) 3D reconstruction across three engines, with SAM 3 segmentation as an optional pre-processing stage routed per engine (crop + rembg for TRELLIS/Hunyuan3D, raw binary mask for SAM 3D Objects). The extended evaluation — 14 HTX-relevant objects × 5 generation conditions = 70 reconstructions — established image-metric and mesh-property baselines (SSIM, PSNR, LPIPS, CLIP; face count, file size, watertightness, degenerate faces). Two limitations were explicitly deferred to term 3: (i) **scale ambiguity** — outputs were metrically meaningless and had to be sized by hand in the sim; and (ii) **mesh-level watertightness / hole-filling**. This report addresses (i) in full and continues to carry (ii) as planned work.

---

## 2. Metric Auto-Scaling (new — the centrepiece section)

### 2.1 Motivation

Image-to-3D engines emit meshes at an arbitrary normalised scale: a 7 m armoured vehicle and a 0.3 m helmet both come out roughly unit-sized. Until now this made every asset unusable in a Unity/Unreal digital-twin scene until an operator manually estimated the real size and typed it in. This is precisely the *scale ambiguity* limitation documented in Section 3.5 of the 2nd-term report ("A photograph contains no absolute scale information… it must be scaled in the digital-twin scene by the operator using a known reference"). The goal of this work was to make every generated model open at the **correct real-world size in metres, automatically, with no manual triage**, while retaining a one-click manual override for cases that need exact dimensions.

### 2.2 Design Rationale (the *why*)

Three design decisions define the approach:

- **View-aligned metric scaling (chosen) over naïve 3D-bbox matching or manual-only.** The naïve approach — match the longest dimension of an image-derived 3D bounding box to the longest dimension of the GLB's bounding box — only works when the input view happens to capture the object's longest axis. A car photographed from the front has an image extent of roughly width × height, whereas the GLB's longest dimension is its length; the axes do not correspond, and the scale is under-estimated by a factor of ~3. The **view-aligned** method instead renders the GLB's silhouette under the *same* camera intrinsics **K** and the *same* object distance **D** as the input photograph, then matches 2D pixel bounding boxes. Because both measurements live in the same image plane, the ratio is correct regardless of viewpoint. A manual-only path was rejected because the assets are required to be sim-ready without per-generation manual triage.

- **UniDepth v2 over Depth Anything V2 for depth + intrinsics.** Both are CVPR 2024 monocular metric-depth models. UniDepth v2 **predicts the camera intrinsics in the same forward pass**, which is essential for operator- or web-sourced images where EXIF focal length is unreliable or absent. Depth Anything V2 (metric) assumes default intrinsics, degrading accuracy on non-phone photographs. *License caveat:* UniDepth is CC BY-NC 4.0 (non-commercial) — acceptable for HTX research and internship reporting, and consistent with the project's existing non-commercial dependencies (SAM 3, Hunyuan3D-2.1). Should commercial deployment become a requirement, Depth Anything V2 metric (Apache 2.0) is the drop-in substitute.

- **nvdiffrast over pytorch3d for silhouette rendering, and rembg over SAM 3 for the auto-scale mask.** nvdiffrast's CUDA rasteriser needs no OpenGL display (relevant inside Docker) and is simpler for pure-silhouette rendering; the container's pytorch3d build lacked GPU support. rembg (U²-Net) is used for the auto-scale mask because it is lightweight (~150 MB, ~1 s, no VRAM contention with the heavy engines), already a project dependency, and does not require prompts. The mask uses a **fallback ladder**: existing alpha channel → rembg → full-frame (flagged low-confidence). This is decoupled from the user's UI segmentation toggle, because "segmentation off for generation" does not mean "off for scaling" — the object must be isolated for sizing regardless of how it entered the engine.

### 2.3 Method — the six-step pipeline

Given the input photograph (always saved as raw RGB at generation time), auto-scaling proceeds as:

1. **Metric depth + intrinsics.** UniDepth v2 (ViT-S/14 backbone, ~340 MB, ~0.4 s/image, ~0.78 GB VRAM) predicts a per-pixel depth map in metres, the camera intrinsics **K** (fx, fy, cx, cy), and pre-unprojected 3D points in the camera frame.
2. **Object mask.** The fallback ladder (alpha → rembg → full-frame) produces a binary object mask.
3. **Mask refinement.** Pixels more than 1.5σ off the object's median depth are dropped, and the lowest-confidence 25% of mask pixels are removed. This suppresses road, shadow, and sky bleed that would otherwise inflate the measured extent.
4. **Image-side measurement.** From the refined mask, take the object's bounding box in pixels and its median depth **D**; convert the *longer* pixel extent to metres as `extent_m = pixels × D / focal_length`. This single number — the object's longest visible side in metres — is the one measurement the pipeline makes.
5. **GLB-side silhouette rendering.** The GLB is rasterised with nvdiffrast at the same distance **D** and same intrinsics **K**. The canonical (unrotated) view is tried first; if the shape-IoU against the image mask is ≥ 0.5 it is accepted, otherwise a search sweeps elevation ∈ {−15°, 0°, +15°} × azimuth ∈ {−135°, −90°, −45°, +45°, +90°, +135°, 180°} and keeps the best-IoU orientation. Shape-IoU is computed by cropping both masks to their bounding boxes, resizing to 128×128, and taking IoU on the normalised shapes — position- and scale-invariant, answering "do the silhouettes match in shape" rather than "do they overlap pixel-for-pixel".
6. **Scale solve and bake.** `scale = max(image_bbox_px) / max(render_bbox_px)`. Because steps 4 and 5 use the same **K** and **D**, the perspective terms cancel and this ratio is exactly the metres-per-unit scale factor. Every GLB vertex is multiplied by `scale` (uniform scale baked into the root transform) and the model is written back in place.

**Confidence tiers** are attached to every result: *high* (IoU ≥ 0.5 and mask from alpha/rembg), *medium* (IoU ≥ 0.3), *low* (otherwise). The tier is surfaced in the UI so the operator knows when to verify.

### 2.4 The uniform-scale principle and what the three reported dimensions mean

The pipeline measures **one** real-world dimension and applies it as a **uniform** scale — the same scalar on x, y, and z. The three dimensions shown in the UI (longest / middle / shortest, in metres) are therefore not three independent estimates: only the **longest** is measured; the middle and shortest are the GLB's own engine-produced proportions multiplied by the same factor.

Uniform scaling preserves geometry exactly — every distance, angle, and volume ratio is unchanged, so circles stay circles and human silhouettes stay human-shaped. This was a deliberate choice over independent per-axis fitting (matching image width and height with two different factors), which would **distort** the mesh: any photograph taken at an angle other than dead side-on would systematically stretch or squish the body, and sim engines expect proportional meshes for physics and collision. The trade-off is that the pipeline trusts the engine to produce proportionally correct meshes and only estimates *size*. Consequently the longest dimension carries only *scale error*, while the middle and shortest additionally carry the engine's *proportion error* — visible in the benchmark as a ~10–12% higher MAPE on the derived axes.

### 2.5 Accuracy booster — CLIP class-size prior

Monocular depth systematically **over-estimates distance for vehicles**, inflating size by 2–4×. A sanity-check layer bounds this:

- A **CLIP zero-shot classifier** (openai/clip-vit-base-patch32 — already a project dependency) labels the object against a 40-entry HTX-domain table (vehicles, weapons, gear, structures).
- Each class carries a plausible size range from manufacturer specifications (e.g. "armoured military vehicle" ≈ 5.5–8.5 m).
- If the geometric estimate falls inside the class range it is accepted unchanged. If it falls outside, the result is blended toward the range, weighted by classifier confidence; if it is wildly off (>2×) and the classifier is confident, it is snapped toward the class median.

The blend is Bayesian in spirit: `final = (1 − α)·geometric + α·class_bound`, with `α` the classifier confidence. It preserves the geometric signal when reasonable and reins it in only when implausible. This layer is the single largest contributor to the benchmark improvement: a controlled ablation (Section 2.7.1) shows that, holding the geometry refinements fixed, enabling the class prior reduces longest-dimension MAPE from 63.7% to 24.6% — roughly three-quarters of the total gain.

### 2.6 System Integration

Auto-scaling is wired into the existing pipeline with no operator action required:

- **Backend.** A new `auto_scale.py` service (~340 lines: mask fallback, UniDepth wrapper, nvdiffrast silhouette renderer, multi-view IoU search, CLIP prior, GLB scale bake, orchestrator, CLI entry) is invoked from `task_manager._save_to_gallery` for `image` and `multi_image` tasks. The result is written as an `auto_scale` block in the gallery index and mirrored onto the task result. New Pydantic schemas (`AutoScaleMetadata`, `AutoScaleDimensions`, `AutoScaleViewAlignment`, `RescaleRequest`) carry the metadata through the API.
- **Manual override.** `POST /api/task/{id}/rescale` accepts a target longest dimension in metres, re-bakes the GLB, and updates the index and in-memory task. Round-trip tested for exactness and reversibility (e.g. 1.99 m → 4.00 m → 1.99 m with no drift; invalid values rejected with 422).
- **Frontend.** A new `AutoScalePanel.tsx` (in the right sidebar, above the export panel) shows dimension cards (longest/middle/shortest in metres), a confidence badge (green/amber/red), the view-match IoU and method, and a "Set longest dim" override input with Apply. The panel hides for older gallery items that have no `auto_scale` metadata.

Each `auto_scale` metadata block records: `auto_scaled`, `confidence`, `mask_source`, `scale_source`, `scale_factor`, `dimensions_m` (xyz + longest/middle/shortest), `view_alignment` (method, IoU, azimuth, elevation), and `object_distance_m`. A side-by-side debug PNG (rendered silhouette overlaid on the input) is written per generation for inspection.

### 2.7 Evaluation

Auto-scaling was benchmarked on the **same 14 HTX-domain objects across all 5 generation pipelines (70 runs)** used for the 2nd-term evaluation, against **manufacturer-specification ground truth** for the longest dimension. The improvement over the initial baseline comes from a bundle of three refinements added on top of the already-present view-aligned silhouette matching: the **CLIP class-size prior (with confidence-weighted blend)**, **depth-gated mask refinement**, and **confidence-weighted object distance**. A controlled ablation isolating the CLIP prior's marginal contribution is reported in Section 2.7.1. Results on the high-confidence ground-truth subset (longest dimension, n = 30):

| Metric (longest dim, high-confidence GT, n = 30) | Before | After |
|---|---:|---:|
| Mean absolute percentage error (MAPE) | 77.5% | **24.6%** |
| Median error | 38.4% | **19.8%** |
| Within ±50% of truth | 53% | **90%** |
| Within ±20% of truth | 37% | **50%** |
| Best pipeline (TRELLIS + rembg) | 40.5% | **19.6%** |

Object-level improvements worth noting:

- **Terrex APC: 184% → 20%** (mean across the 5 pipelines) — CLIP recognises "armoured vehicle" and bounds the depth over-shoot.
- **APICS boom gate: 189% → 27%** (mean across the 5 pipelines; the worst single pipeline fell from a 671% Hunyuan3D failure) — recovered by the class prior and mask refinement.
- **SCDF ambulance: 64% → 21%** — now consistent across all five pipelines.

**Cross-engine consistency.** On a shared input (passport-kiosk pair), a standalone development run — recorded during initial implementation, prior to the class-size prior — produced 2.15 × 1.76 × 1.00 m on TRELLIS (IoU 0.89) and 2.19 × 2.01 × 1.06 m on Hunyuan3D, a ~2% spread on the measured axis, confirming the sizing is engine-independent. (In the full 14×5 benchmark, which includes the class prior, the same kiosk pair measures ≈1.9 m longest; the two figures come from different pipeline configurations.)

### 2.7.1 Ablation: contribution of the CLIP class-size prior

To separate the CLIP class-size prior's effect from the geometric refinements, the 14×5 benchmark was re-run with the prior disabled (`use_class_prior=False`) while keeping the depth-gated mask refinement and confidence-weighted distance in place. On the high-confidence ground-truth subset (longest dimension, n = 30):

| Configuration | MAPE ↓ | Median ↓ | Within ±50% ↑ |
|---|---:|---:|---:|
| Initial baseline (no refinements) | 77.5% | 38.4% | 53% |
| Geometry refinements, **prior OFF** | 63.7% | 29.8% | 57% |
| Geometry refinements, **prior ON** (shipped) | **24.6%** | **19.8%** | **90%** |

The class prior contributes **39.1 of the 52.9-point total MAPE reduction (~74%)**; the geometry refinements (mask + distance) contribute the remaining ~13.8 points. The effect is largest on the pipelines most prone to depth over-estimation on vehicles — for example Hunyuan3D + rembg falls from 136.3% to 35.1% MAPE once the prior clamps the estimate into the class size range. The prior is therefore the dominant driver of accuracy, and its authority is gated by classifier confidence (`α`) so it only intervenes when the geometric estimate is implausible.

**Honest limitations** (consistent with the discussion style of Section 7 in the 2nd-term report):

- **Boats remain hard (~46% error).** Their genuine 8–25 m range is too wide for the class prior to tighten meaningfully.
- **Generic kiosks/booths have uncertain ground truth**, so part of the reported error is unknown-truth rather than model error.
- **Vehicles at mid-range** can still be over-estimated when UniDepth over-predicts distance and the classifier is not confident enough to intervene.
- The middle/shortest dimensions inherit the engine's proportion error on top of the scale error (Section 2.4).

Next steps for accuracy are an optional open-source VLM correction tier when the geometric estimate is low-confidence, and — the only true fix for the hidden third axis — multi-view triangulation, which the multi-image fusion work (Section 4) would enable.

### 2.8 Model-Foundations Addendum (update to Section 3 of the 2nd-term report)

Consistent with the 2nd-term report's treatment of each model in terms of representation, training signal, and inherent limitations, the auto-scale stage adds two supporting models:

- **UniDepth v2** [Piccinelli et al., 2024] is a universal monocular metric-depth network that jointly regresses a dense metric depth map *and* the camera intrinsics from a single RGB image, using a self-promptable camera module that decouples the camera representation from the depth head. Predicting intrinsics in-band is what lets the pipeline recover true metric depth from images with no reliable EXIF. *Inherent limitation:* like all monocular metric-depth methods it exhibits systematic distance bias for some object classes (notably vehicles), which motivates the class-prior correction.
- **CLIP ViT-B/32** [Radford et al., 2021] (already used elsewhere in the pipeline for the CLIP evaluation metric) is used here zero-shot to classify the object into an HTX-domain size table. *Inherent limitation:* CLIP class confidence is coarse and can be miscalibrated on HTX-specific or brand-named assets; the blend weight `α` therefore governs how much authority the prior is given.

---

## 3. Logo-on-Surface Post-Processing (new)

### 3.1 Motivation

HTX requested the ability to place a company/agency logo onto a generated product model. Generation itself cannot reliably render a clean logo on a surface, so this is implemented as a **post-process** with two operator-selectable outputs: a crisp overlay decal, and a bake that projects the logo into the actual albedo texture while **hiding any existing logo underneath**. The latter is what makes the logo look native to the surface rather than a sticker floating on top.

### 3.2 Phase 1 — Interactive decal placement (frontend)

A new `LogoDecal.tsx` component, modelled on the interactive eraser's lifecycle and wired into the 3D viewer, lets the operator:

- Upload a PNG (loaded as a `THREE.Texture`) and click the model surface; a raycast places a `DecalGeometry` oriented to the hit normal.
- Select a placed logo (highlighted), **drag it across the surface** to reposition (it re-projects and re-orients to the new normal, and can hop onto a different mesh part), and adjust **Size / Rotate** live. Undo/Reset are provided; orbit is suspended while dragging.

The logo uses an unlit `MeshBasicMaterial` (exported via glTF `KHR_materials_unlit`) so the PNG shows true colours rather than rendering black on unlit faces. Each decal geometry is baked into the **target mesh's local space** and parented to that mesh, so it stays glued to the surface and exports correctly through `GLTFExporter`. This overlay path ("Keep as decal") produces a crisp result but is GLB-only and cannot hide underlying content.

### 3.3 Phase 2 — Bake into the surface texture, with smudge (backend)

A new backend service (`logo_bake.py`) and endpoint (`POST /api/logo/bake`, multipart: base GLB, logo PNG, placements JSON, smudge flag, target resolution) projects the logo into the albedo texture:

1. Load the GLB (`trimesh`, forced to a single mesh) → vertices, faces, UVs, base-colour texture.
2. Upscale the albedo so its longest side is ≥ the target resolution (default 4096; Hi-res 8192), giving the small decal footprint enough texels.
3. Per placement, build an orthonormal projector frame (u, v, w) from the surface normal plus in-plane rotation, and cull candidate faces to those inside a shallow projector box and **front-facing** (`face_normal · w > 0`).
4. Rasterise each candidate face in **albedo-UV space**; for each covered texel, barycentric-interpolate the surface point, project it into the decal frame, and if inside the box, **bilinear-sample** the logo and alpha-composite it onto the albedo, accumulating a footprint mask.
5. **Smudge:** before compositing, `cv2.inpaint(albedo, footprint, INPAINT_TELEA)` heals the old surface content so the new logo reads cleanly over a repaired background.
6. Write the texture back, export the GLB (all formats), and register a new gallery item via the existing edited-model save path.

The frontend sends placements in the GLB scene-root frame (matching trimesh's baked vertices) and exposes toolbar controls for **Smudge** (default on), **Hi-res** (8192 vs 4096), **Keep as decal**, and **Bake into surface**. Logo aspect ratio is preserved on both sides: `size` is the longer side and the other axis follows `aspect = w/h`, so non-square logos are not squished.

### 3.4 Limitations

- **Projector punch-through on highly layered geometry.** A shallow uniform projector box is used to avoid the logo appearing on multiple layers (worst case observed on a fire-truck grille). Smooth surfaces are unaffected; very layered geometry could still punch through and would need a connectivity-limited footprint (BFS from the seed face) if it arises.
- **Tiny-text sharpness ceiling.** Bilinear sampling plus albedo upscaling to 4096/8192 fixes most blur, but the hard ceiling is atlas texel density — where a surface region has little UV area, even 8192 is limited. Razor-sharp small text would require a dedicated high-resolution logo material rather than merging into the shared texture atlas.

---

## 4. Multi-Image Fusion (update — still work-in-progress)

Multi-image fusion remains the most experimental feature and is carried forward as work-in-progress from the 2nd-term report. Two Hunyuan3D fusion modes are implemented in the diffusion sampling loop — **stochastic** (cycle through image conditionings, one per sampling step) and **multidiffusion** (run all conditionings at every step and average the noise predictions, with classifier-free guidance applied per conditioning before averaging). It is functional but still being validated against the single-image baseline. Beyond quality, multi-image input is the natural route to **true three-axis metric sizing**: multiple photographs at different angles provide independent measurements of the axes that a single view leaves hidden, which would remove the uniform-scale assumption of Section 2.4.

---

## 5. Deployment and HTX Progress Review (new)

The system was **deployed and demonstrated live to HTX** during the term (progress meeting, 16 June 2026; demo prepared and verified the prior day) on the RTX 5090 with all three engines registered. The live run reproduced the benchmark: the Terrex APC classified as "armoured military vehicle" (confidence 0.83), longest dimension 8.21 m against a 7.0 m ground truth (17.2% error), view-IoU 0.58, dimensions 8.21 × 1.84 × 2.93 m.

The demonstrated operator workflow for sizing is: upload → generate (TRELLIS + rembg is fastest at ~10 s and the most accurate pipeline) → the Auto-Scale panel shows dimensions in metres with a confidence badge → if needed, type a target size to rescale instantly via the override endpoint. The model then drops into Unity/Unreal at the correct physical size.

Deployment mechanics were also hardened during the term. The system runs as a single Docker container (backend on port 8000, also serving the built frontend via StaticFiles). Because the backend Python source is baked into the image (only weights, gallery, and caches are volume-mounted), backend changes are applied with `docker cp` + container restart (restart is cheap — engines load lazily), while the static frontend is updated by copying the built bundle with no restart. This is the workflow used to bring the live demo up to date with the auto-scale feature.

---

## 6. Updated Progress, Timeline, and Planned Work (update to Sections 9 and 11)

### 6.1 Completed this term (additions to the 2nd-term "Completed Work" list)

- **Metric auto-scaling** — view-aligned monocular metric scaling (UniDepth v2 + rembg + nvdiffrast) with a CLIP class-size prior; automatic on every image / multi-image generation, with a manual rescale override endpoint. Benchmarked on the 14×5 set: longest-dimension MAPE 77.5% → 24.6%, 90% of results within ±50% of ground truth.
- **Logo-on-surface post-processing** — interactive decal placement (Phase 1) and bake-into-texture with cv2 inpaint smudge to hide underlying logos (Phase 2), with aspect-ratio-preserving projection and front-face culling.
- **Live HTX deployment and progress demonstration** on the RTX 5090, including the sizing workflow and one-click rescale override; deployment workflow (docker cp / static-bundle) documented.

### 6.2 Work in progress

- **Multi-image fusion** — Hunyuan3D stochastic / multidiffusion sampling; functional, still being validated against the single-image baseline. Prerequisite for true three-axis metric sizing.

### 6.3 Planned work (carried forward / refined)

- **Mesh post-processing** — hole-filling for non-watertight outputs; seam-aware UV-preserving mesh simplification. (Still open from the 2nd term.)
- **Auto-scale accuracy tiers** — optional open-source VLM correction when the geometric estimate is low-confidence (Tier 2); multi-view triangulation for the hidden third axis (Tier 3, requires the multi-image workflow).
- **Export optimisation** — Unity / Unreal Engine 5 import-ready presets (LOD generation, collider mesh, PBR channel packing).
- **Remote deployment** — currently local/lab; remote container deployment with operator authentication.
- **Operator UX refinements** — based on HTX user testing and the June review.
- **Optional model integration** — evaluation of any newer 3D reconstruction / metric-depth models that emerge during the term.

### 6.4 Timeline note (update to Section 11)

The internship remains a one-year, two-days-per-week programme (start 10 November 2025, 48 weeks). At this mid-term point the project is at approximately **week 35 of 48**, within **Phase 3 — Refinement and Integration** (Mar – Aug 2026). This half of the term delivered metric auto-scaling, logo-on-surface post-processing, and the live HTX deployment; the remaining Phase 3 work is mesh watertightness, export optimisation, and remote deployment, ahead of **Phase 4 — Documentation and Delivery** (Aug 2026).

---

## 7. Conclusion (update to Section 12)

At this 3rd-term mid-point, the project has moved beyond the complete three-engine operator workflow reported at the end of term 2 by making its outputs **directly usable in a simulation without manual sizing**. Metric auto-scaling closes the scale-ambiguity limitation that the 2nd-term report explicitly deferred: every generated model now opens at an estimated real-world size in metres, with a benchmarked longest-dimension error reduced from 77.5% to 24.6% MAPE (90% within ±50% of ground truth) and a one-click manual override for exact dimensions. Logo-on-surface post-processing answers a direct HTX operational request by baking agency logos into the surface texture and healing any pre-existing logo underneath. The system was deployed and demonstrated live to HTX on the RTX 5090.

The methodological pattern of the 2nd-term report is maintained throughout: each new component is justified against alternatives (view-aligned vs naïve scaling; UniDepth vs Depth Anything; uniform vs per-axis scaling), described in enough depth to be reproduced, and evaluated on the same 14-object HTX-relevant benchmark with an explicit, honest account of where it fails. The remaining term-3 work — mesh watertightness, export optimisation, remote deployment, and the multi-image path to true three-axis sizing — builds directly on these foundations.

---

## 8. New References to add (fold into the References section, alphabetical)

- Piccinelli, L., Yang, Y.-H., Sakaridis, C., Segu, M., Li, S., Van Gool, L., & Yu, F. (2024). *UniDepth: Universal Monocular Metric Depth Estimation.* CVPR 2024. (UniDepth v2; CC BY-NC 4.0)
- Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020). *U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection.* Pattern Recognition, 106, 107404. (rembg backbone)

*(Already cited in the 2nd-term report and reused here: Radford et al., 2021 — CLIP; Laine et al., 2020 — nvdiffrast.)*

---

*Prepared as merge-ready content for the 3rd Term Mid-Term Report, styled to match the 2nd Term Final Report (Image-to-3D Model Conversion Pipeline).*
