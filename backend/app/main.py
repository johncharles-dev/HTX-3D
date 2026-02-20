"""HTX 3D Generation Tool — FastAPI Backend."""

import logging
import asyncio
import json

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import CORS_ORIGINS, TRELLIS_ENGINE_DIR, WEIGHTS_DIR, GALLERY_DIR, detect_gpu
from .routers import generate, gallery
from .services.trellis import TrellisEngine
from .services.task_manager import TaskManager
from .dependencies import set_task_manager, get_task_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load engine and start worker. Shutdown: cleanup."""
    gpu = detect_gpu()
    logger.info(f"GPU: {gpu}")

    # Initialize task manager
    tm = TaskManager()

    # Load Trellis engine
    engine = TrellisEngine(TRELLIS_ENGINE_DIR)
    try:
        engine.load(WEIGHTS_DIR)
        tm.register_engine(engine)
    except Exception as e:
        logger.error(f"Failed to load Trellis engine: {e}")
        logger.info("Backend running without GPU engine (API-only mode)")

    set_task_manager(tm)
    await tm.start()

    yield

    await tm.stop()
    engine.unload()


app = FastAPI(
    title="HTX 3D Generation Tool",
    description="Image-to-3D and Text-to-3D generation API powered by TRELLIS",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(generate.router)
app.include_router(gallery.router)


# ── WebSocket for Progress ─────────────────────────────

@app.websocket("/ws/progress/{task_id}")
async def ws_progress(websocket: WebSocket, task_id: str):
    """Stream real-time progress updates for a generation task."""
    await websocket.accept()
    tm = get_task_manager()
    sub_queue = tm.subscribe_progress(task_id)

    try:
        while True:
            update = await asyncio.wait_for(sub_queue.get(), timeout=120)
            await websocket.send_json(update.model_dump())
            if update.status in ("completed", "failed"):
                break
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        tm.unsubscribe_progress(task_id, sub_queue)


# ── Health Check ───────────────────────────────────────

@app.get("/api/health")
async def health():
    gpu = detect_gpu()
    tm = get_task_manager()
    return {
        "status": "ok",
        "gpu": gpu,
        "models_loaded": [name for name, eng in tm.engines.items() if eng.loaded],
        "queue_size": tm.queue_size,
    }


# ── Serve Frontend (production) ────────────────────────

# Walk up from this file to find frontend/dist (works in both source tree and Docker)
_here = Path(__file__).resolve().parent
for _ancestor in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
    frontend_dist = _ancestor / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
        break
