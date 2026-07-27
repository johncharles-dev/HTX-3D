# manual_eval — blinded scoring + live engine comparison

Two browser tools, served by one script.

| Page | URL | Purpose |
|---|---|---|
| Blinded scoring | `/` | Score the 98 benchmark models (14 objects × 7 pipelines) without knowing which engine produced which |
| Live comparison | `/live.html` | Drop any image, generate across all 7 pipelines on the spot, compare in the same 3D grid |

## Run it

```bash
./serve.sh          # start (or restart); prints LAN / Tailscale / SSH-tunnel URLs
./serve.sh stop
./serve.sh 9000     # alternate port
```

`serve.py` serves `_serve/` and reverse-proxies `/api` to the backend on
`127.0.0.1:8000`. Proxying keeps everything same-origin, so the backend's CORS
config doesn't need changing. `serve.sh` reports whether `:8000` is reachable —
live generation needs it, blinded scoring does not.

## Blinded scoring

1. Score each slot A–G on geometry (1–5) and texture (1–5), plus defect flags.
   See `RUBRIC.md` for the anchors and the rules that keep it defensible.
2. Scores autosave in the browser; **Export CSV** writes `scores_cj.csv`.
3. Copy it back here and run:

```bash
python3 analyze_manual.py
```

Produces `manual_eval_report.md`: mean scores with paired bootstrap CIs,
Wilcoxon signed-rank tests across all pipeline pairs, defect frequencies, and
inter-rater agreement if several people scored.

Multiple raters: each copies `scores_TEMPLATE.csv` to `scores_<name>.csv`;
the analyser picks up every `scores_*.csv` automatically.

**The 🔒 Blinded / 🔓 Revealed toggle ends the blinding.** It's for demoing
after scoring is finished. Score first, reveal second.

## Live comparison

Drop an image → pick pipelines → **Generate**. The 3 `rembg` pipelines run on the
raw photo. The 4 `SAM 3` pipelines need a cutout: use the segmentation panel
(point +/−, box drag, or text prompt), then **Use this mask**. One cutout is
shared by every SAM 3 variant, matching the benchmark's fairness control.

Engines share one GPU worker and run sequentially — all 7 takes roughly 6–9
minutes. Results appear progressively as each finishes.

Generations are saved to the gallery like any other, and auto-scale runs on them.

> **Known backend bug:** `target_face_count` is not sent to Hunyuan. Its
> decimation path (`hunyuan.py` `_decimate_mesh`) saves through pymeshlab, which
> cannot write GLB, so any Hunyuan run with `target_face_count > 0` fails with
> `Unknown format for save: glb`. This affects the main tool too.

## Regenerating the ignored assets

These are excluded from git (large or copied) and rebuilt locally:

```bash
# blinded montages + panels + key + score-sheet template  (seed 1337, gamma 1.6)
python3 make_blind_montages.py --gamma 1.6

# per-slot panels used by both pages
python3 - <<'PY'
import csv; from pathlib import Path; from PIL import Image
OUT=Path('blind_panels'); OUT.mkdir(exist_ok=True)
key=list(csv.DictReader(open('blind_key.csv')))
byobj={}
for r in key: byobj.setdefault(r['object_id'],[]).append(r['slot'])
for obj in sorted(byobj):
    m=Image.open(Path('blind_montages')/f'{obj}.png')
    for i,lab in enumerate(['INPUT']+sorted(byobj[obj])):
        x=10+i*570
        m.crop((x,126,x+560,686)).save(OUT/f'{obj}__{lab}.png')
PY

# blinded GLB symlinks
python3 - <<'PY'
import csv; from pathlib import Path
OUT=Path('_serve/blind_models'); OUT.mkdir(parents=True,exist_ok=True)
SRC=Path('../benchmark_v2/outputs').resolve()
for f in OUT.iterdir(): f.unlink()
for r in csv.DictReader(open('blind_key.csv')):
    (OUT/f"{r['object_id']}__{r['slot']}.glb").symlink_to(
        SRC/r['object_id']/r['pipeline']/'model.glb')
PY

# vendored three.js (r182) — needed for the 3D grid, no CDN at runtime
mkdir -p _serve/vendor/jsm/{loaders,controls,utils}
F=../../frontend/node_modules/three
cp $F/build/three.module.js $F/build/three.core.js _serve/vendor/
cp $F/examples/jsm/loaders/GLTFLoader.js          _serve/vendor/jsm/loaders/
cp $F/examples/jsm/controls/OrbitControls.js      _serve/vendor/jsm/controls/
cp $F/examples/jsm/utils/BufferGeometryUtils.js   _serve/vendor/jsm/utils/
```

`blind_key.csv` is deliberately outside `_serve/`, so the answer key is not
reachable from the browser (`/blind_key.csv` returns 404). `key.json` inside
`_serve/` exists only for the reveal toggle.
