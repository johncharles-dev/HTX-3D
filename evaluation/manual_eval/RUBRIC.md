# Manual evaluation — rater instructions

**Time needed: ~45 minutes.** 14 objects, ~3 minutes each.

You are scoring 3D reconstructions produced by 7 different pipelines from a
single photograph. You do **not** know which pipeline produced which column —
that is deliberate. Do not open `blind_key.csv` until you have finished and
saved your scores.

## Setup

```bash
cd /home/cj/HTX-3D/evaluation/manual_eval
cp scores_TEMPLATE.csv scores_<yourname>.csv
```

Open `scores_<yourname>.csv` in LibreOffice/Excel, and open the montages in
`blind_montages/` in an image viewer. Work through objects 01 → 14 in order.

Each montage shows the **INPUT photograph** (left, red label) followed by seven
renders labelled **A–G**, all rendered from the same camera under the same
lighting. Zoom in — the images are 560 px per panel.

## What to score

For every slot A–G, fill in two scores and any defect flags.

### `geometry_1to5` — shape fidelity

Judge structure and proportion, ignoring colour and texture.

| Score | Meaning |
|---|---|
| **5** | All major structures present, correct proportions, thin parts (mirrors, antennas, arms, railings) intact |
| **4** | Correct overall shape, minor detail loss |
| **3** | Recognisable, but notable structures missing, merged, or blobby |
| **2** | Distorted or partly collapsed; identifiable only from context |
| **1** | Unrecognisable — flat sheet, blob, or wrong object |

### `texture_1to5` — appearance fidelity

Judge colour, markings and surface detail against the input photo.

| Score | Meaning |
|---|---|
| **5** | Colours and markings accurate, sharp, correctly placed |
| **4** | Correct colours, some blur or smearing |
| **3** | Approximate colours; markings smeared, stretched, or misplaced |
| **2** | Mostly wrong colours, or heavy bleeding across surfaces |
| **1** | No meaningful texture (flat grey / noise) |

### Defect flags — put `1` if present, leave blank if not

| Column | Flag it when you see |
|---|---|
| `floaters` | Detached fragments or specks floating off the object |
| `holes` | Gaps or missing surface revealing the interior |
| `flat_collapse` | A structure rendered as a flat plane / card instead of a volume |
| `janus_duplicate` | Duplicated or mirrored features (two fronts, repeated faces) |
| `front_only_texture` | Texture present on the visible face but grey/blank elsewhere |

`notes` is free text — use it for anything you want to quote on a slide.

## Rules that keep this defensible

1. **Score every slot, even the obviously bad ones.** Skipping rows biases the means.
2. **Judge against the input photo, not against the other renders.** A montage
   where all seven are poor should score low across the board.
3. **Don't go back and "even out" earlier objects.** First impression, move on.
4. **Don't look up which pipeline is which** until you have saved the file.
5. If you genuinely cannot tell, score it and flag the uncertainty in `notes` —
   do not leave it blank.

## When finished

```bash
python analyze_manual.py
```

This un-blinds the slots and writes `manual_eval_report.md` with mean scores,
bootstrap confidence intervals, paired significance tests, defect frequencies,
and inter-rater agreement if more than one person scored.

## Multiple raters

Each rater copies the template to their own `scores_<name>.csv` and scores
independently, using the **same** montages. `analyze_manual.py` picks up every
`scores_*.csv` automatically. Three raters is enough to report agreement; one
rater is a single-observer assessment and must be described as such.

## Disclosure note

Blinding seed: **1337** (recorded in `make_blind_montages.py --seed`).

These montages were built with **gamma 1.6**: a uniform brightness lift applied
identically to all seven renders of every object, and never to the input
photograph. It was needed because several dark objects (EOD robot, Terrex,
HIMARS) render with their surface detail compressed into the bottom few percent
of the tonal range, making texture unscoreable. Gamma is monotonic, so it
reveals shadow detail without reordering brightness or clipping highlights, and
being uniform across pipelines it cannot favour any one of them.

State this in the method section, e.g.: _"Renders were displayed with a uniform
gamma 1.6 lift, applied equally to all pipelines, to make dark objects
scoreable; input photographs were unmodified."_
