# Mesh defect metrics — 7 pipelines × 23 objects

Ground-truth-free, computed from the GLBs alone. Orientation-independent
(bbox axes are sorted), so unlike the SSIM/PSNR/LPIPS columns these are
unaffected by the fixed-camera pose bug.


`aspect_ratio` = predicted (shortest/longest) ÷ true (shortest/longest).
1.00 = correct proportions; 0.02 = 50× flatter than the real object.

## Per-pipeline means

| Pipeline | components | floater face frac | boundary loops | degenerate | aspect_ratio (median) | mean \|log aspect err\| | n ≤0.5 (too flat) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRELLIS 1 · rembg | 12.4 | 0.079 | 4 | 0 | — | — | — |
| TRELLIS 1 · SAM 3 | 9.1 | 0.086 | 2 | 1 | — | — | — |
| Hunyuan3D · rembg | 9.7 | 0.116 | 0 | 0 | — | — | — |
| Hunyuan3D · SAM 3 | 12.0 | 0.086 | 0 | 0 | — | — | — |
| SAM 3D · SAM 3 | 3.6 | 0.117 | 0 | 0 | — | — | — |
| trellis2_rembg_2k | 494.7 | 0.165 | 12 | 6 | — | — | — |
| trellis2_rembg_4k | 492.3 | 0.165 | 12 | 6 | — | — | — |
| trellis2_sam3_2k | 306.8 | 0.119 | 8 | 5 | — | — | — |
| trellis2_sam3_4k | 304.3 | 0.119 | 8 | 4 | — | — | — |

## aspect_ratio per object (bold = ≤0.5, i.e. >2× too flat)

| Object | true aspect | TRELLIS 1 · rembg | TRELLIS 1 · SAM 3 | Hunyuan3D · rembg | Hunyuan3D · SAM 3 | SAM 3D · SAM 3 | trellis2_rembg_2k | trellis2_rembg_4k | trellis2_sam3_2k | trellis2_sam3_4k |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| B2_01 | — | — | — | — | — | — | — | — | — | — |
| B2_02 | — | — | — | — | — | — | — | — | — | — |
| B2_03 | — | — | — | — | — | — | — | — | — | — |
| B2_04 | — | — | — | — | — | — | — | — | — | — |
| B2_05 | — | — | — | — | — | — | — | — | — | — |
| B2_06 | — | — | — | — | — | — | — | — | — | — |
| B2_07 | — | — | — | — | — | — | — | — | — | — |
| B2_08 | — | — | — | — | — | — | — | — | — | — |
| B2_09 | — | — | — | — | — | — | — | — | — | — |
| B2_10 | — | — | — | — | — | — | — | — | — | — |
| B2_11 | — | — | — | — | — | — | — | — | — | — |
| B2_12 | — | — | — | — | — | — | — | — | — | — |
| B2_13 | — | — | — | — | — | — | — | — | — | — |
| B2_14 | — | — | — | — | — | — | — | — | — | — |
| B2_15 | — | — | — | — | — | — | — | — | — | — |
| B2_16 | — | — | — | — | — | — | — | — | — | — |
| B2_17 | — | — | — | — | — | — | — | — | — | — |
| B2_18 | — | — | — | — | — | — | — | — | — | — |
| B2_19 | — | — | — | — | — | — | — | — | — | — |
| B2_20 | — | — | — | — | — | — | — | — | — | — |
| B2_21 | — | — | — | — | — | — | — | — | — | — |
| B2_22 | — | — | — | — | — | — | — | — | — | — |
| B2_23 | — | — | — | — | — | — | — | — | — | — |
