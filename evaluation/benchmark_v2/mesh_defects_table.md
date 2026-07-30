# Mesh defect metrics — 7 pipelines × 10 objects

Ground-truth-free, computed from the GLBs alone. Orientation-independent
(bbox axes are sorted), so unlike the SSIM/PSNR/LPIPS columns these are
unaffected by the fixed-camera pose bug.

Excluded from these aggregates (4 objects, still present in `mesh_defects.csv`): 11_apics_red_car_booth, 12_apics_boom_gate, 13_apics_booth_barrier, 14_apics_kiosks_passport. Multi-object scenes where background removal cannot isolate the intended target, with estimated rather than published ground truth.

`aspect_ratio` = predicted (shortest/longest) ÷ true (shortest/longest).
1.00 = correct proportions; 0.02 = 50× flatter than the real object.

## Per-pipeline means

| Pipeline | components | floater face frac | boundary loops | degenerate | aspect_ratio (median) | mean \|log aspect err\| | n ≤0.5 (too flat) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRELLIS.2 · rembg | 184.9 | 0.118 | 11 | 3 | 1.44 | 0.40 | 0/10 |
| TRELLIS.2 · SAM 3 | 189.6 | 0.073 | 7 | 2 | 1.42 | 0.36 | 0/10 |
| TRELLIS 1 · rembg | 9.4 | 0.073 | 3 | 0 | 1.20 | 0.34 | 0/10 |
| TRELLIS 1 · SAM 3 | 9.7 | 0.058 | 3 | 0 | 1.21 | 0.30 | 0/10 |
| Hunyuan3D · rembg | 16.3 | 0.044 | 0 | 2 | 0.90 | 0.85 | 3/10 |
| Hunyuan3D · SAM 3 | 12.3 | 0.101 | 0 | 0 | 1.03 | 0.60 | 1/10 |
| SAM 3D · SAM 3 | 5.4 | 0.048 | 2 | 0 | 1.33 | 0.31 | 0/10 |

## aspect_ratio per object (bold = ≤0.5, i.e. >2× too flat)

| Object | true aspect | TRELLIS.2 · rembg | TRELLIS.2 · SAM 3 | TRELLIS 1 · rembg | TRELLIS 1 · SAM 3 | Hunyuan3D · rembg | Hunyuan3D · SAM 3 | SAM 3D · SAM 3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 01_scdf_ambulance | 0.2911 | 1.54 | 1.66 | 1.42 | 1.56 | 1.39 | 1.41 | 1.64 |
| 02_scdf_fire_engine | 0.2632 | 1.51 | 1.49 | 1.34 | 1.36 | 1.36 | 1.30 | 1.50 |
| 03_terrex_apc | 0.3714 | 0.57 | 0.88 | 0.60 | 0.74 | 0.51 | 0.67 | 1.22 |
| 04_police_motorcycle | 0.4091 | 1.87 | 2.14 | 0.98 | 0.94 | **0.02** | **0.01** | 0.96 |
| 05_bomb_disposal_robot | 0.5385 | 0.74 | 0.78 | 0.85 | 0.97 | 0.76 | 0.93 | 0.89 |
| 06_sbs_double_decker_bus | 0.2125 | 1.88 | 1.81 | 1.89 | 1.93 | 1.04 | 1.07 | 1.66 |
| 07_security_guard_house | 0.72 | 0.98 | 1.00 | 1.06 | 1.05 | **0.35** | 0.94 | 1.09 |
| 08_police_patrol_car | 0.3052 | 1.05 | 1.04 | 1.09 | 1.11 | **0.41** | 1.02 | 1.27 |
| 09_himars_launcher | 0.3429 | 1.37 | 1.35 | 1.31 | 1.30 | 1.09 | 1.04 | 1.39 |
| 10_coast_guard_boat | 0.1667 | 1.88 | 1.88 | 2.69 | 2.18 | 1.90 | 1.71 | 1.96 |
