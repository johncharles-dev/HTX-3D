## Table 6.5.1 — Per-condition mean metrics (14 objects)

| Condition | n | Gen time (s) | Faces | Verts | File MB | Watertight % | Min axis (m) | SSIM ↑ | PSNR ↑ | LPIPS ↓ | CLIP ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trellis_rembg | 14 | 10.41 | 21,515 | 16,112 | 1.734 | 0.0 | 0.491 | 0.242 | 6.5536 | 0.7823 | 0.516 |
| trellis_sam3 | 14 | 10.2 | 18,561 | 14,285 | 1.657 | 0.0 | 0.438 | 0.2427 | 6.3779 | 0.7858 | 0.5135 |
| hunyuan_rembg | 14 | 74.91 | 38,386 | 26,462 | 5.279 | 0.0 | 0.543 | 0.2893 | 6.13 | 0.7892 | 0.5117 |
| hunyuan_sam3 | 14 | 75.01 | 37,683 | 25,237 | 5.174 | 0.0 | 0.664 | 0.2868 | 6.2836 | 0.7886 | 0.5241 |
| sam3d_sam3 | 14 | 19.69 | 17,136 | 12,511 | 1.594 | 0.0 | 0.434 | 0.252 | 6.3621 | 0.7974 | 0.5306 |

## Per-object detail

| object | trellis_rembg faces | trellis_rembg MB | trellis_rembg SSIM | trellis_rembg CLIP | trellis_sam3 faces | trellis_sam3 MB | trellis_sam3 SSIM | trellis_sam3 CLIP | hunyuan_rembg faces | hunyuan_rembg MB | hunyuan_rembg SSIM | hunyuan_rembg CLIP | hunyuan_sam3 faces | hunyuan_sam3 MB | hunyuan_sam3 SSIM | hunyuan_sam3 CLIP | sam3d_sam3 faces | sam3d_sam3 MB | sam3d_sam3 SSIM | sam3d_sam3 CLIP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_scdf_ambulance | 10481 | 1.486 | 0.2729 | 0.466 | 14435 | 1.619 | 0.3209 | 0.5162 | 40000 | 6.065 | 0.2906 | 0.3851 | 40000 | 5.666 | 0.2933 | 0.4419 | 14090 | 1.549 | 0.2822 | 0.4674 |
| 02_scdf_fire_engine | 18616 | 1.784 | 0.239 | 0.5298 | 19007 | 1.827 | 0.2274 | 0.4869 | 40000 | 7.471 | 0.2058 | 0.4942 | 40000 | 7.809 | 0.2322 | 0.4811 | 17971 | 1.816 | 0.2368 | 0.5448 |
| 03_terrex_apc | 9618 | 1.483 | 0.2297 | 0.4799 | 10749 | 1.385 | 0.2483 | 0.4777 | 40000 | 6.99 | 0.2493 | 0.4652 | 40000 | 6.093 | 0.2442 | 0.5367 | 17376 | 1.59 | 0.2403 | 0.531 |
| 04_police_motorcycle | 18301 | 1.85 | 0.2046 | 0.5517 | 17900 | 1.832 | 0.2232 | 0.5923 | 40000 | 3.01 | 0.3066 | 0.5084 | 40000 | 4.331 | 0.3071 | 0.5413 | 19960 | 1.767 | 0.2575 | 0.5819 |
| 05_bomb_disposal_robot | 21739 | 1.764 | 0.2806 | 0.5533 | 21754 | 1.807 | 0.1362 | 0.5952 | 40000 | 7.519 | 0.2713 | 0.6046 | 40000 | 3.801 | 0.2187 | 0.5903 | 20036 | 1.677 | 0.1816 | 0.6108 |
| 06_sbs_double_decker_bus | 9474 | 1.408 | 0.1514 | 0.4659 | 33109 | 1.905 | 0.1956 | 0.4631 | 40000 | 4.576 | 0.2245 | 0.5146 | 40000 | 4.919 | 0.2258 | 0.5295 | 8438 | 1.412 | 0.1741 | 0.5053 |
| 07_security_guard_house | 52376 | 2.192 | 0.3162 | 0.5294 | 30284 | 1.792 | 0.2786 | 0.47 | 40000 | 3.096 | 0.3227 | 0.4709 | 40000 | 4.834 | 0.3285 | 0.583 | 60790 | 2.266 | 0.2543 | 0.4583 |
| 08_police_patrol_car | 9649 | 1.347 | 0.2288 | 0.4355 | 9777 | 1.357 | 0.2335 | 0.5636 | 22840 | 5.535 | 0.2649 | 0.5233 | 40000 | 6.059 | 0.2204 | 0.5233 | 11313 | 1.434 | 0.2382 | 0.4499 |
| 09_himars_launcher | 19074 | 1.714 | 0.3515 | 0.4979 | 18325 | 1.656 | 0.3255 | 0.5715 | 40000 | 6.053 | 0.4466 | 0.4325 | 40000 | 5.941 | 0.4247 | 0.494 | 20936 | 1.65 | 0.2633 | 0.5658 |
| 10_coast_guard_boat | 14097 | 1.923 | 0.1871 | 0.5126 | 11505 | 1.828 | 0.1841 | 0.49 | 34572 | 6.293 | 0.2051 | 0.4309 | 40000 | 5.708 | 0.1939 | 0.4967 | 9340 | 1.713 | 0.2016 | 0.4887 |
| 11_apics_red_car_booth | 25254 | 1.744 | 0.2318 | 0.4709 | 20800 | 1.753 | 0.2342 | 0.3876 | 40000 | 3.822 | 0.3758 | 0.4642 | 40000 | 4.056 | 0.2944 | 0.3897 | 11030 | 1.446 | 0.235 | 0.4298 |
| 12_apics_boom_gate | 31847 | 1.914 | 0.1798 | 0.5162 | 6824 | 1.263 | 0.2567 | 0.5094 | 40000 | 4.584 | 0.2614 | 0.5747 | 7564 | 3.033 | 0.2877 | 0.4364 | 2560 | 1.422 | 0.2951 | 0.4808 |
| 13_apics_booth_barrier | 31820 | 1.882 | 0.1714 | 0.5549 | 31898 | 1.796 | 0.1816 | 0.506 | 40000 | 4.584 | 0.2614 | 0.5747 | 40000 | 5.554 | 0.245 | 0.6847 | 15862 | 1.361 | 0.1861 | 0.6391 |
| 14_apics_kiosks_passport | 28877 | 1.788 | 0.3435 | 0.6601 | 13496 | 1.38 | 0.3519 | 0.5598 | 40000 | 4.308 | 0.3649 | 0.7208 | 40000 | 4.637 | 0.4999 | 0.6087 | 10213 | 1.215 | 0.4815 | 0.6743 |

## Table 6.5.2 — SAM 3 prompt-mode usage (14 objects)

| Mode | Count | Share |
|---|---:|---:|
| text-only | 13 | 92.9% |
| manual (UI) | 1 | 7.1% |

### Per-object

| Object | Mode | Prompt |
|---|---|---|
| 01_scdf_ambulance | text-only | ambulance |
| 02_scdf_fire_engine | text-only | fire truck |
| 03_terrex_apc | text-only | armored vehicle |
| 04_police_motorcycle | text-only | motorcycle |
| 05_bomb_disposal_robot | text-only | robot |
| 06_sbs_double_decker_bus | text-only | bus |
| 07_security_guard_house | text-only | guard house |
| 08_police_patrol_car | text-only | police car |
| 09_himars_launcher | text-only | military truck |
| 10_coast_guard_boat | text-only | boat |
| 11_apics_red_car_booth | text-only | car |
| 12_apics_boom_gate | manual (UI) | manual |
| 13_apics_booth_barrier | text-only | booth |
| 14_apics_kiosks_passport | text-only | kiosk |