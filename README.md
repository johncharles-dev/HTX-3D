# HTX-3D — single photograph to simulation-ready 3D asset

Turns one photograph into a textured 3D model at **real-world metric scale**, for building
digital-twin environments. Built during an SUTD MDAI-E internship with HTX's Process
Modelling and Simulation team.

Five generation engines behind one API, an interactive segmentation front-end, automatic
metric scaling, and a browser UI. Runs on NVIDIA Blackwell (sm_120) as well as older
architectures.

> **Deploying this?** Start with [`docker/.env.example`](docker/.env.example) and
> [`docs/reference/`](docs/reference/). Several models carry licences with real
> restrictions — territory limits, trade controls, and one non-commercial dependency. See
> [Licensing](#licensing) before operational use.

## What it does

- **Image to 3D** — single or multi-view, with background removal or interactive segmentation
- **Text to 3D** — TRELLIS text pipeline
- **Metric auto-scaling** — estimates real-world dimensions from a single photo
  (UniDepth depth + view alignment + class priors), so assets import at true size
- **Interactive segmentation** — SAM 3 with box, point and text prompts
- **Logo baking** — project a decal onto a surface and bake it into the texture
- **Retexture / quick-adjust / text-guided edit** of existing models
- **Export** — GLB (textured PBR), OBJ + MTL, STL, PLY
- **Gallery** — browse, preview, re-download, rescale past generations
- **Live progress** over WebSocket

## Engines

| Engine | Runs | Notes |
|---|---|---|
| **TRELLIS** | In container | Image and text to 3D. Loaded at startup |
| **TRELLIS.2** | **Host service** | Highest-scoring pipeline. Needs the host's torch 2.10 / FA2 stack — see [`services/trellis2/`](services/trellis2/) |
| **Hunyuan3D 2.1** | In container | PBR texture pipeline. Lazy-loaded |
| **SAM 3D Objects** | In container | Image + mask to 3D. Lazy-loaded |
| **SAM 3** | In container | Segmentation front-end, not a generator |

Only one generation engine holds GPU memory at a time; `task_manager.py` swaps them as jobs
require. TRELLIS.2 is a thin HTTP proxy to a service on the host, because it cannot run
inside the container.

## Quick start

```bash
cp docker/.env.example docker/.env      # set host paths for weights and caches
cd docker && docker compose up -d
# http://localhost:8000
```

Model weights are **not** bundled and do not all auto-download — SAM 3 and DINOv3 are gated
and need a HuggingFace account with accepted terms. See
[`docker/.env.example`](docker/.env.example) for what to set, and
[`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) for a full walkthrough.

TRELLIS.2 is optional and runs separately; the other four engines work without it. To
enable it, follow [`services/trellis2/README.md`](services/trellis2/README.md).

### Local development

```bash
pip install -r backend/requirements.txt
cd backend/engines/trellis && bash setup.sh --all && cd ../../..
cd frontend && npm install && npm run dev &          # Vite dev server on :5173
XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa \
  TRELLIS_ENGINE_DIR=./backend/engines/trellis \
  WEIGHTS_DIR=./weights GALLERY_DIR=./gallery \
  TRELLIS2_SERVICE_URL=http://localhost:8710 \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

`TRELLIS2_SERVICE_URL` defaults to `http://host.docker.internal:8710`, which only resolves
inside the container — set it to `localhost` when running the backend directly.

## Layout

```
backend/app/          FastAPI application (~5,300 lines, the actual product)
  routers/            generate · segment · gallery · logo
  services/           4 engines + task_manager + auto_scale + class_priors + logo_bake
backend/engines/      Vendored upstream: trellis, hunyuan, sam3, sam3d_objects
services/trellis2/    Host-side TRELLIS.2 microservice, patch and systemd unit
frontend/             React 19 + Vite + Tailwind + three.js (@react-three/fiber)
docker/               Dockerfile, compose, .env.example
docs/reference/       Vendoring, dependency pins, licence audit
evaluation/           Benchmarks, manual evaluation, ground truth
weights/  gallery/    Gitignored — see docker/.env.example
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/generate/image` | Single image to 3D |
| POST | `/api/generate/multi-image` | 2–4 view generation |
| POST | `/api/generate/text` | Text to 3D |
| POST | `/api/generate/edit` | Text-guided edit |
| POST | `/api/generate/retexture` | Re-texture an existing mesh |
| POST | `/api/generate/quick-adjust` | Fast parameter re-run |
| POST | `/api/segment/start` · `/box` · `/point` · `/points` · `/text` · `/reset` · `/confirm` | SAM 3 interactive segmentation |
| GET | `/api/segment/overlay/{session_id}` | Current mask overlay |
| POST | `/api/logo/bake` | Bake a decal into the texture |
| GET | `/api/task/{id}` | Status and results |
| POST | `/api/task/{id}/cancel` · `/rescale` | Cancel job · override metric scale |
| WS | `/ws/progress/{id}` | Live progress |
| GET | `/api/gallery` · `/api/download/{id}/{file}` | Browse · download |
| DELETE | `/api/gallery/{id}` | Delete an entry |
| GET | `/api/health` | GPU, engines, queue state |

## Hardware

Developed and benchmarked on an **RTX 5090** (Blackwell sm_120, 31.4 GB, driver 590.48.01,
CUDA 12.8). CUDA extensions build for compute 8.0, 8.6, 8.9, 10.0 and 12.0.

Peak VRAM observed is ~8.7 GB for TRELLIS.2, so 12 GB is workable and 24 GB comfortable —
but only the 5090 configuration has been measured.

> **Timings do not transfer.** HTX's RTX PRO 6000 Blackwell Max-Q is the same architecture
> (so the build is unchanged) but is power-limited to 300 W against the 5090's 575 W. Every
> generation time recorded in this repository was measured on the 575 W card and must be
> re-benchmarked. See [`services/trellis2/README.md`](services/trellis2/README.md).

## Licensing

Not all components are permissively licensed. Full audit in
[`docs/reference/vendoring.md`](docs/reference/vendoring.md).

| Component | Licence | Constraint |
|---|---|---|
| TRELLIS, TRELLIS.2 | MIT | None |
| SAM 3, SAM 3D Objects | **SAM License** (Meta) | Trade controls: no military/warfare, nuclear or espionage use. Agreement must travel with the weights |
| Hunyuan3D 2.1 | **Tencent Community** | **Excludes the EU, UK and South Korea** — model *and its output*. Singapore is inside the Territory |
| UniDepth (auto-scale) | **CC BY-NC 4.0** | **Non-commercial only.** Blocks commercial deployment of metric scaling |

SAM 3, SAM 3.1, SAM 3D Objects and DINOv3 are **gated** on HuggingFace — access is granted
per account and does not transfer with the files.

## Reproducibility

Several dependencies were originally cloned from unpinned git tips and two drifted within
25 days. All five are now pinned in the Dockerfile, and each build logs the commits it
resolved.

**The pins are not equally strong.** `pytorch3d` and `moge` are pinned to the commits
recovered from the image that produced the published evaluation results. The three CUDA
extensions are pinned to commits *observed on 2026-08-14*, because the originals are
unrecoverable — they were installed from a local path, so pip recorded no commit. A
current build does not exactly reproduce the benchmarked geometry.
See [`docs/reference/dependency-pins.md`](docs/reference/dependency-pins.md).

## Documentation

| Document | Contents |
|---|---|
| [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) | Full setup for Linux, WSL2 and Docker |
| [`docs/MACHINE_REQUIREMENTS.md`](docs/MACHINE_REQUIREMENTS.md) | Hardware and network requirements |
| [`services/trellis2/README.md`](services/trellis2/README.md) | TRELLIS.2 host service, upstream pin, acceptance test |
| [`docs/reference/vendoring.md`](docs/reference/vendoring.md) | Engine provenance and licences |
| [`docs/reference/dependency-pins.md`](docs/reference/dependency-pins.md) | Pinned commits and observed drift |
| [`evaluation/FINAL_EVALUATION.md`](evaluation/FINAL_EVALUATION.md) | Benchmark results |
| [`evaluation/FINDINGS_AND_HANDOVER.md`](evaluation/FINDINGS_AND_HANDOVER.md) | Findings and handover notes |

## Adding an engine

1. Subclass `BaseEngine` in `backend/app/services/` — implement `load()`, `unload()`,
   `generate_from_image()`, `export_mesh()`
2. Register it in `main.py`'s lifespan so `task_manager` can swap it
3. Add it to the frontend engine selector

`trellis2.py` is the smallest example (156 lines) and shows the out-of-process pattern.
