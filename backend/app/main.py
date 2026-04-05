"""HTX 3D Generation Tool — FastAPI Backend."""

import logging
import asyncio
import json

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import CORS_ORIGINS, TRELLIS_ENGINE_DIR, HUNYUAN_ENGINE_DIR, SAM3D_OBJECTS_DIR, WEIGHTS_DIR, GALLERY_DIR, detect_gpu
from .routers import generate, gallery, segment
from .services.trellis import TrellisEngine
from .services.hunyuan import HunyuanEngine
from .services.sam3d_objects import Sam3DObjectsEngine
from .services.sam3_segmentation import SAM3Service
from .services.task_manager import TaskManager
from .dependencies import set_task_manager, get_task_manager, set_sam3_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load engines and start worker. Shutdown: cleanup."""
    gpu = detect_gpu()
    logger.info(f"GPU: {gpu}")

    # Initialize task manager
    tm = TaskManager()

    # Load Trellis engine (loaded at startup)
    trellis = TrellisEngine(TRELLIS_ENGINE_DIR)
    try:
        trellis.load(WEIGHTS_DIR)
        tm.register_engine(trellis)
    except Exception as e:
        logger.error(f"Failed to load Trellis engine: {e}")
        logger.info("Backend running without Trellis engine")

    # Register Hunyuan engine (not loaded until first use — lazy loading)
    hunyuan = HunyuanEngine(HUNYUAN_ENGINE_DIR)
    tm.register_engine(hunyuan)
    logger.info("Hunyuan3D engine registered (will load on first use)")

    # Register SAM 3D Objects engine (not loaded until first use — lazy loading)
    sam3d = Sam3DObjectsEngine(SAM3D_OBJECTS_DIR)
    tm.register_engine(sam3d)
    logger.info("SAM 3D Objects engine registered (will load on first use)")

    # Initialize SAM3 segmentation service (loaded on-demand per session)
    sam3_service = SAM3Service()
    set_sam3_service(sam3_service)
    logger.info("SAM3 segmentation service initialized")

    set_task_manager(tm)
    await tm.start()

    yield

    await tm.stop()
    if trellis.loaded:
        trellis.unload()
    if hunyuan.loaded:
        hunyuan.unload()
    if sam3d.loaded:
        sam3d.unload()
    if sam3_service.loaded:
        sam3_service.unload()


app = FastAPI(
    title="HTX 3D Generation Tool",
    description="Image-to-3D and Text-to-3D generation API powered by TRELLIS, Hunyuan3D, and SAM 3D Objects",
    version="0.2.0",
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
app.include_router(segment.router)


# -- WebSocket for Progress --------------------------------

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


# -- Health Check ------------------------------------------

@app.get("/api/health")
async def health():
    gpu = detect_gpu()
    tm = get_task_manager()
    return {
        "status": "ok",
        "gpu": gpu,
        "models_loaded": [name for name, eng in tm.engines.items() if eng.loaded],
        "engines_registered": list(tm.engines.keys()),
        "active_engine": tm.active_engine,
        "queue_size": tm.queue_size,
    }


# -- Serve Frontend (production) ---------------------------

# Walk up from this file to find frontend/dist (works in both source tree and Docker)
_here = Path(__file__).resolve().parent
for _ancestor in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
    frontend_dist = _ancestor / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
        break
