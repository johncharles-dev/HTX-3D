# HTX 3D Generation Tool

Image-to-3D and Text-to-3D generation tool powered by [TRELLIS](https://github.com/microsoft/TRELLIS), optimized for NVIDIA GPUs including Blackwell (RTX 50 series).

## Features

- **Image to 3D**: Single or multi-view image input with automatic background removal
- **Text to 3D**: Generate 3D models from text descriptions
- **Multiple export formats**: GLB (textured PBR), OBJ (with textures), STL (3D printing), PLY (Gaussian splat)
- **Interactive 3D preview**: Orbit, zoom, and inspect models in the browser
- **Generation gallery**: Browse, preview, and re-download past generations
- **Docker support**: Run on Linux or Windows (via WSL2) with GPU passthrough

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run (model weights auto-download on first start, ~5 GB)
cd docker
docker compose up --build

# Open http://localhost:8000
```

### Option 2: Local Development

```bash
# 1. Install backend + TRELLIS dependencies (in your CUDA-enabled conda env)
pip install -r backend/requirements.txt

# 2. Build CUDA extensions (uses setup.sh from TRELLIS engine)
cd backend/engines/trellis && bash setup.sh --all && cd ../../..

# 3. Install frontend dependencies
cd frontend && npm install && cd ..

# 4. Start backend (model weights auto-download on first start)
XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa \
  TRELLIS_ENGINE_DIR=./backend/engines/trellis \
  WEIGHTS_DIR=./weights \
  GALLERY_DIR=./gallery \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# 5. Start frontend (separate terminal)
cd frontend && npm run dev
```

See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for complete step-by-step instructions for your OS and GPU.

## VRAM Management

The application automatically detects GPU VRAM and adjusts behavior:

| Tier | VRAM | Precision | Model Swapping | Notes |
|------|------|-----------|---------------|-------|
| **High** | >= 24 GB | float32 | None | Both pipelines on GPU |
| **Medium** | 12-23 GB | float16 | Active model on GPU, other on CPU | Swap on pipeline switch |
| **Low** | 8-11 GB | float16 | Aggressive | int32 FlexiCubes, CPU background removal |

No configuration needed — the system selects the right tier at startup based on detected VRAM.

## GPU Compatibility

CUDA extensions are compiled for these architectures:

| Architecture | Compute | Example GPUs |
|-------------|---------|-------------|
| Ampere | 8.0, 8.6 | A100, RTX 3060-3090 |
| Ada Lovelace | 8.9 | RTX 4060-4090 |
| Hopper | 10.0 | H100, H200 |
| Blackwell | 12.0 | RTX 5070-5090, B100/B200 |

## Architecture

```
HTX-3D/
├── backend/
│   ├── app/              # FastAPI application
│   │   ├── main.py       # App entry point, WebSocket, health check
│   │   ├── routers/      # API endpoints (generate, gallery, download)
│   │   ├── services/     # Engine wrappers, task queue manager
│   │   └── models/       # Pydantic request/response schemas
│   └── engines/
│       └── trellis/      # TRELLIS source (Blackwell-compatible)
├── frontend/             # React + Vite + Tailwind + Three.js
├── docker/               # Dockerfile + docker-compose
├── scripts/              # Model downloader
├── docs/                 # Machine requirements, setup guide
├── weights/              # Model weights (gitignored, ~5 GB)
└── gallery/              # Generated models (gitignored)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate/image` | Generate from single image |
| POST | `/api/generate/multi-image` | Generate from 2-4 images |
| POST | `/api/generate/text` | Generate from text prompt |
| POST | `/api/generate/edit` | Text-guided 3D editing |
| GET | `/api/task/{id}` | Get task status and results |
| WS | `/ws/progress/{id}` | Real-time progress streaming |
| GET | `/api/gallery` | List past generations |
| DELETE | `/api/gallery/{id}` | Delete a gallery item |
| GET | `/api/download/{id}/{file}` | Download exported file |
| GET | `/api/health` | System health + GPU info |

## Documentation

| Guide | Description |
|-------|-------------|
| [Setup Guide](docs/SETUP_GUIDE.md) | Complete setup & deployment guide for Linux, Windows (WSL2), and Docker across all GPU generations |
| [Machine Requirements](docs/MACHINE_REQUIREMENTS.md) | Hardware and software requirements, network needs |

**Minimum hardware**: NVIDIA GPU with 8 GB+ VRAM, CUDA compute 8.0+
**Tested on**: RTX 5090 (32 GB, Blackwell) -- ~12s per generation at standard quality

## Adding New Models

The backend is designed for multi-model support. To add a new engine (e.g., Hunyuan, TripoSG):

1. Create a new engine class in `backend/app/services/` that extends `BaseEngine`
2. Implement `load()`, `generate_from_image()`, `export_mesh()`, etc.
3. Register it in `main.py` during startup
4. Add new model type to the frontend's model selector
