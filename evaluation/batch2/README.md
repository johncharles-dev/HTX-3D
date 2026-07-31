# Batch 2 — 23 objects × 9 pipelines (2026-07-30/31)

207 models generated through `manual_eval/_serve/batch.html`. No published
dimensions yet, so only ground-truth-free metrics apply.

| file | contents |
|---|---|
| `defects.csv` | mesh integrity per (object, pipeline) — components, floater fraction, boundary edges/loops, degenerate faces |
| `defects_table.md` | per-pipeline aggregate |
| `orbit_defects.csv` | multi-view — silhouette coverage (collapse), texture one-sidedness, opposite-view CLIP |

Object ids `B2_01..B2_23` match the blinded scoring set
(`manual_eval/blind_key_b2.csv`, seed 2026, slots A–I).

## Findings

- **Segmentation improves mesh quality on every engine.** TRELLIS.2 492 → 304
  components, floater fraction 0.165 → 0.119, boundary loops 12.4 → 7.6.
  TRELLIS 1 12.4 → 9.1 components.
- **4K texture buys nothing measurable** over 2K: identical on every structural
  measure, 2.3× larger files, 30% slower.
- **SAM 3D remains the cleanest topology** (3.6 components vs TRELLIS.2 ~300–500).
- **No collapsed geometry in 207 models**, versus 2/20 Hunyuan SAM 3 runs on the
  benchmark set — its flat-sheet failure is object-dependent, not systematic.

## Caveat

Generated across the deployment of the SAM 3 composite fix (commit `1a3d27c`).
Runs submitted before ~18:05 UTC used the old bounding-box bridge for
`trellis_sam3` and `hunyuan_sam3`; TRELLIS.2 and SAM 3D were never affected.

## Reproducing

```bash
~/miniconda3/envs/3D/bin/python benchmark_v2/mesh_defect_metrics.py \
  --outputs /home/cj/HTX-3D/gallery/_b2_eval --gt /nonexistent.csv \
  --exclude '' --out-prefix batch2/defects
docker exec htx-3d python /app/gallery/_b2_render.py     # 828 orbit renders
docker exec htx-3d python /app/gallery/_b2_metrics.py
```

Meshes are hardlinked as root into `gallery/_b2_eval/` and `gallery/_b2_models/`
(zero extra disk; survives deletion of the gallery entries).
