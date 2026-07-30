# Mesh defect metrics — 7 pipelines × 14 objects

Ground-truth-free, computed from the GLBs alone. Orientation-independent
(bbox axes are sorted), so unlike the SSIM/PSNR/LPIPS columns these are
unaffected by the fixed-camera pose bug.

`aspect_ratio` = predicted (shortest/longest) ÷ true (shortest/longest).
1.00 = correct proportions; 0.02 = 50× flatter than the real object.

## Per-pipeline means

| Pipeline | components | floater face frac | boundary loops | degenerate | aspect_ratio (median) | mean \|log aspect err\| | n ≤0.5 (too flat) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRELLIS.2 · rembg | 156.6 | 0.139 | 16 | 3 | 1.44 | 0.44 | 0/14 |
| TRELLIS.2 · SAM 3 | 176.6 | 0.074 | 11 | 3 | 1.17 | 0.35 | 1/14 |
| TRELLIS 1 · rembg | 9.2 | 0.103 | 3 | 1 | 1.32 | 0.41 | 0/14 |
| TRELLIS 1 · SAM 3 | 8.3 | 0.078 | 2 | 1 | 1.21 | 0.34 | 0/14 |
| Hunyuan3D · rembg | 12.3 | 0.032 | 0 | 1 | 0.96 | 0.97 | 4/14 |
| Hunyuan3D · SAM 3 | 10.5 | 0.086 | 0 | 0 | 0.98 | 0.67 | 2/14 |
| SAM 3D · SAM 3 | 4.3 | 0.035 | 1 | 0 | 1.24 | 0.38 | 2/14 |

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
| 11_apics_red_car_booth | 0.8 | 0.51 | **0.43** | 0.56 | 0.50 | **0.01** | **0.12** | **0.43** |
| 12_apics_boom_gate | 0.3 | 1.78 | 1.02 | 2.27 | 1.61 | 1.17 | 0.52 | **0.40** |
| 13_apics_booth_barrier | 0.4 | 1.33 | 1.30 | 1.70 | 1.38 | 0.88 | 0.87 | 1.55 |
| 14_apics_kiosks_passport | 0.5 | 1.88 | 0.85 | 1.63 | 0.81 | 1.08 | 1.49 | 0.95 |
