# Manual (human) evaluation — blinded montage scoring

Raters: **1** (b2_cj) · objects scored: **19/23** · pipelines: **9**

## 1 · Coverage & validity

- Complete (geometry+texture) judgements: **171 / 171** (100%)

- No malformed rows.

## 2 · Mean scores (1–5, higher is better)

Paired bootstrap 95% CI, resampling objects.

| Pipeline | Geometry | Texture | Combined | Combined 95% CI |
|---|---:|---:|---:|:---:|
| trellis2_rembg_4k | 3.79 | 3.79 | **3.79** | [3.32, 4.18] |
| trellis2_rembg_2k | 3.79 | 3.79 | **3.79** | [3.37, 4.18] |
| trellis2_sam3_2k | 3.68 | 3.74 | **3.71** | [3.18, 4.16] |
| trellis2_sam3_4k | 3.68 | 3.68 | **3.68** | [3.21, 4.13] |
| hunyuan_sam3 | 3.47 | 3.26 | **3.37** | [2.84, 3.84] |
| trellis_sam3 | 3.42 | 3.05 | **3.24** | [2.87, 3.61] |
| trellis_rembg | 3.16 | 2.84 | **3.00** | [2.58, 3.45] |
| sam3d_sam3 | 3.21 | 2.53 | **2.87** | [2.53, 3.24] |
| hunyuan_rembg | 2.68 | 2.84 | **2.76** | [2.29, 3.24] |

## 3 · Paired Wilcoxon signed-rank (combined score, paired by object)

No pipeline pair separates at p<0.05 — differences are within noise at this sample size.

## 4 · Defect frequency (count of objects flagged, per pipeline)

Counted once per (object, pipeline): flagged if any rater flagged it.

| Pipeline | floaters | holes | flat collapse | janus duplicate | front only texture | any |
|---|---:|---:|---:|---:|---:|---:|
| hunyuan_rembg | 1/19 | 5/19 | 0/19 | 0/19 | 0/19 | **6/19** |
| hunyuan_sam3 | 2/19 | 2/19 | 0/19 | 0/19 | 0/19 | **4/19** |
| sam3d_sam3 | 0/19 | 1/19 | 0/19 | 0/19 | 0/19 | **1/19** |
| trellis2_rembg_2k | 1/19 | 9/19 | 0/19 | 0/19 | 0/19 | **10/19** |
| trellis2_rembg_4k | 0/19 | 9/19 | 0/19 | 0/19 | 0/19 | **9/19** |
| trellis2_sam3_2k | 3/19 | 12/19 | 0/19 | 0/19 | 0/19 | **14/19** |
| trellis2_sam3_4k | 2/19 | 11/19 | 0/19 | 0/19 | 0/19 | **12/19** |
| trellis_rembg | 2/19 | 3/19 | 0/19 | 1/19 | 0/19 | **6/19** |
| trellis_sam3 | 1/19 | 4/19 | 0/19 | 0/19 | 0/19 | **5/19** |

## 5 · Inter-rater agreement

_Single rater — this is a single-observer assessment. Describe it as such; no agreement statistic is possible._
