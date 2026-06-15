# Benchmark v2 — 5-Condition × 14-Object Final-Report Evaluation

## Layout

```
benchmark_v2/
├── manifest.json          ← 14 objects, 5 conditions, SAM 3 prompts
├── prepare_masks.py       ← Phase 1: SAM 3 → segmented PNGs
├── run_benchmark.py       ← Phase 2: 5-condition matrix → GLBs
├── data/<id>/             ← per-object inputs (filled by Phase 1)
│   ├── original.<ext>
│   ├── segmented.png
│   └── meta.json
├── outputs/<id>/<cond>/   ← per-(object, condition) GLB (filled by Phase 2)
│   └── model.glb
└── results.json           ← Phase 2 task records (statuses, times, errors)
```

## Quick start

Backend container must be running on `localhost:8000`.

```bash
cd /home/cj/HTX-3D/evaluation/benchmark_v2

# Phase 1 — segmentation prep (manual review, ~25 min for 14 objects)
python prepare_masks.py                 # all 14
python prepare_masks.py 14              # just object 14 (re-run after fixing prompt)
python prepare_masks.py 11 12 13 14     # just APICS group

# Phase 2 — automated benchmark (~1.5 hr GPU time)
python run_benchmark.py                       # all engines, all objects (engine-by-engine)
python run_benchmark.py --engine trellis      # TRELLIS pass only (~15 min)
python run_benchmark.py --engine hunyuan      # Hunyuan pass only (~85 min)
python run_benchmark.py --engine sam3d        # SAM 3D pass only (~12 min)
python run_benchmark.py --only 11_apics_red_car_booth   # one object across all conditions
python run_benchmark.py --no-skip             # force re-run of already-completed entries
```

Phase 2 saves incrementally — interrupting it and re-running resumes from where it stopped (skips completed records by default).

## Conditions

| Key | Engine | SAM 3 mask used? | Engine input |
|---|---|---|---|
| `trellis_rembg` | TRELLIS | no  | original photograph (engine runs internal `rembg`) |
| `trellis_sam3`  | TRELLIS | yes | segmented RGBA → engine takes crop+rembg bridge path |
| `hunyuan_rembg` | Hunyuan3D | no | original photograph |
| `hunyuan_sam3`  | Hunyuan3D | yes | segmented RGBA → crop+rembg bridge |
| `sam3d_sam3`    | SAM 3D Objects | yes | segmented RGBA (mandatory) |

## Object-list ordering rationale

The 14 objects are pre-ordered easy-to-hard:

- **#1–7** clean reference shots, text-only SAM 3 prompts (warmup)
- **#8–10** mixed-context, may need point refinement
- **#11–14** APICS multi-object scenes — point refinement expected

When running Phase 1, work through them in numerical order so you build SAM 3
prompting fluency before tackling the trickier multi-object APICS images.

## When SAM 3 text prompts fail entirely

Some objects (specialized military vehicles, biometric kiosks, EOD robots)
fall outside SAM 3's text vocabulary. For those, **use the existing
SegmentationWorkspace UI** in the running app to segment interactively, then
hand the result to the script:

1. Open the app at `http://localhost:8000` (or `:5173` in dev mode).
2. Upload the image and click "Segment". Use whatever combination of
   text / box / point prompts works to isolate the target.
3. Click "Use Mask". The segmented RGBA file is written into the
   server's temp directory — find it via the browser network log
   (the segmented path is in the response of `/api/segment/confirm`),
   or from the running container at `/tmp/segmented/<session_id>/segmented.png`.
4. Copy the segmented file into this benchmark folder:
   ```bash
   cp <path-to-segmented.png> /home/cj/HTX-3D/evaluation/benchmark_v2/data/<id>/segmented.png
   ```
5. Also copy the original image so the engine's crop+rembg bridge can find it:
   ```bash
   cp /home/cj/HTX-3D_samples/<source>.png /home/cj/HTX-3D/evaluation/benchmark_v2/data/<id>/original.png
   ```

When you re-run `prepare_masks.py`, it will detect the existing
`segmented.png` and skip API segmentation for that entry (logging
`[manual] segmented.png already present, skipping API segmentation`).
Pass `--force` to re-run segmentation despite an existing file:

```bash
python prepare_masks.py 05_bomb_disposal_robot --force
```

## Adjusting a script-driven prompt

If a SAM 3 text prompt picks the wrong region but you'd rather stay in the
manifest-driven workflow, edit the entry to `text+points` mode:

```json
{
  "id": "14_apics_kiosks_pair",
  "prompt": {
    "mode": "text+points",
    "text": "kiosk",
    "points": [[1024, 600], [1800, 580]],
    "labels": [1, 0]
  },
  "mask_index": 0
}
```

`labels`: 1 = positive (target object), 0 = negative (exclude). Then re-run:

```bash
python prepare_masks.py 14_apics_kiosks_pair --force
```

To find correct point coordinates, open the image in any image viewer that
shows pixel position — your IDE preview, GIMP, or `xdg-open` + cursor. SAM 3
operates in original-image pixel coordinates.

## After the benchmark

`results.json` contains per-(object, condition) records with status,
generation time, output GLB path, and any errors. The next step is metric
computation (SSIM / PSNR / LPIPS / CLIP) by re-rendering each output GLB at
512×512 from a canonical viewpoint and comparing against the input
photograph — that step uses the existing `evaluation/render_glb.py` and
`evaluate_v2.py` pipeline.
