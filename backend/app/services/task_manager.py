"""Sequential task queue for GPU-bound generation jobs.

Only one generation runs at a time (VRAM constraint).
Tasks are queued and processed in order.
"""

import asyncio
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import GALLERY_DIR, TEMP_DIR, MAX_SEED
from ..models.schemas import (
    TaskStatus,
    ExportFile,
    ExportFormat,
    GenerationResult,
    GalleryItem,
    ProgressUpdate,
)
from .base import BaseEngine

logger = logging.getLogger(__name__)


class TaskManager:
    """Manages the generation queue and task state."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: dict[str, dict] = {}  # task_id -> task state
        self.engines: dict[str, BaseEngine] = {}
        self._active_engine: Optional[str] = None  # name of currently loaded engine
        self._worker_task: Optional[asyncio.Task] = None
        self._progress_subscribers: dict[str, list[asyncio.Queue]] = {}  # task_id -> subscriber queues
        self._cancelled: set[str] = set()  # task IDs marked for cancellation
        self._load_gallery_index()

    def register_engine(self, engine: BaseEngine):
        self.engines[engine.name] = engine
        if engine.loaded:
            self._active_engine = engine.name

    @property
    def active_engine(self) -> Optional[str]:
        return self._active_engine

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Task worker started")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()

    # -- Engine Switching ------------------------------------

    def _ensure_engine_loaded(self, engine_name: str) -> BaseEngine:
        """Ensure the requested engine is loaded. Unload current if different."""
        engine = self.engines.get(engine_name)
        if not engine:
            raise RuntimeError(f"Engine '{engine_name}' not registered")

        if self._active_engine == engine_name and engine.loaded:
            return engine

        # Unload current engine if a different one is active
        if self._active_engine and self._active_engine != engine_name:
            current = self.engines.get(self._active_engine)
            if current and current.loaded:
                logger.info(f"Unloading engine '{self._active_engine}' to switch to '{engine_name}'")
                current.unload()

        # Load requested engine
        if not engine.loaded:
            from ..config import WEIGHTS_DIR
            logger.info(f"Loading engine '{engine_name}'...")
            engine.load(WEIGHTS_DIR)

        self._active_engine = engine_name
        return engine

    # -- Task Submission ------------------------------------

    def submit_task(self, task_type: str, params: dict) -> str:
        task_id = uuid.uuid4().hex[:12]
        task = {
            "id": task_id,
            "type": task_type,  # "image", "multi_image", "text", "export"
            "params": params,
            "status": TaskStatus.QUEUED,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }
        self.tasks[task_id] = task
        self.queue.put_nowait(task)
        logger.info(f"Task {task_id} queued ({task_type})")
        return task_id

    def get_task(self, task_id: str) -> Optional[dict]:
        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Mark a task for cancellation. Returns True if the task was found."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        self._cancelled.add(task_id)
        task["status"] = TaskStatus.CANCELLED
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._broadcast_progress(task_id, "Cancelled", 0.0, "Task cancelled by user")
        logger.info(f"Task {task_id} cancelled")
        return True

    def is_cancelled(self, task_id: str) -> bool:
        return task_id in self._cancelled

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    # -- Progress Streaming ---------------------------------

    def subscribe_progress(self, task_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._progress_subscribers.setdefault(task_id, []).append(q)
        return q

    def unsubscribe_progress(self, task_id: str, q: asyncio.Queue):
        subs = self._progress_subscribers.get(task_id, [])
        if q in subs:
            subs.remove(q)

    def _broadcast_progress(self, task_id: str, stage: str, progress: float, message: str = ""):
        update = ProgressUpdate(
            task_id=task_id,
            status=self.tasks[task_id]["status"],
            stage=stage,
            progress=progress,
            message=message,
        )
        for q in self._progress_subscribers.get(task_id, []):
            try:
                q.put_nowait(update)
            except asyncio.QueueFull:
                pass

    # -- Worker Loop ----------------------------------------

    async def _worker_loop(self):
        while True:
            try:
                task = await self.queue.get()
                task_id = task["id"]

                # Skip tasks cancelled while queued
                if self.is_cancelled(task_id):
                    self._cancelled.discard(task_id)
                    self.queue.task_done()
                    logger.info(f"Task {task_id} skipped (cancelled while queued)")
                    continue

                task["status"] = TaskStatus.PROCESSING
                task["started_at"] = datetime.now(timezone.utc).isoformat()
                self._broadcast_progress(task_id, "Starting", 0.0)

                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self._process_task, task
                    )
                    # Check if cancelled during execution
                    if self.is_cancelled(task_id):
                        self._cancelled.discard(task_id)
                        logger.info(f"Task {task_id} finished but was cancelled — discarding result")
                    else:
                        task["status"] = TaskStatus.COMPLETED
                        task["result"] = result
                        task["completed_at"] = datetime.now(timezone.utc).isoformat()
                        self._broadcast_progress(task_id, "Complete", 1.0, "Generation finished")
                        self._save_to_gallery(task)
                except Exception as e:
                    if not self.is_cancelled(task_id):
                        logger.exception(f"Task {task_id} failed: {e}")
                        task["status"] = TaskStatus.FAILED
                        task["error"] = str(e)
                        task["completed_at"] = datetime.now(timezone.utc).isoformat()
                        self._broadcast_progress(task_id, "Error", 0.0, str(e))
                    else:
                        self._cancelled.discard(task_id)

                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker loop error: {e}")

    def _process_task(self, task: dict) -> dict:
        task_id = task["id"]
        task_type = task["type"]
        params = task["params"]

        def progress_cb(stage: str, progress: float):
            self._broadcast_progress(task_id, stage, progress)

        # Determine which engine to use
        engine_name = params.get("engine", "trellis")
        engine = self._ensure_engine_loaded(engine_name)

        # Resolve seed
        seed = params.get("seed", 42)
        if params.get("randomize_seed", True):
            seed = random.randint(0, MAX_SEED)
        params["seed"] = seed

        output_dir = os.path.join(GALLERY_DIR, task_id)
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()

        # Build engine_params from task params (engine extracts what it needs)
        engine_params = {k: v for k, v in params.items()
                         if k not in ("image_path", "image_paths", "prompt", "seed",
                                      "randomize_seed", "model", "engine", "formats",
                                      "mesh_simplify", "texture_size", "fill_holes",
                                      "fill_holes_max_size", "mode", "base_task_id",
                                      "mesh_file_path", "target_face_count",
                                      "remove_floaters")}
        # original_image_path is now passed through to engines (Hunyuan uses it
        # to get full RGB when a SAM3 segmented image is the primary input)

        # -- Generate ----------------------------------------
        if task_type == "image":
            gen_data = engine.generate_from_image(
                image_path=params["image_path"],
                seed=seed,
                progress_callback=progress_cb,
                **engine_params,
            )
        elif task_type == "multi_image":
            gen_data = engine.generate_from_images(
                image_paths=params["image_paths"],
                seed=seed,
                mode=params.get("mode", "stochastic"),
                progress_callback=progress_cb,
                **engine_params,
            )
        elif task_type == "text":
            gen_data = engine.generate_from_text(
                prompt=params["prompt"],
                seed=seed,
                progress_callback=progress_cb,
                **engine_params,
            )
        elif task_type == "edit":
            # Edit is TRELLIS-specific — call directly
            gen_data = engine.edit_with_text(
                prompt=params["prompt"],
                seed=seed,
                slat_steps=params.get("slat_steps", 12),
                slat_guidance=params.get("slat_guidance", 3.0),
                base_task_id=params.get("base_task_id"),
                mesh_file_path=params.get("mesh_file_path"),
                progress_callback=progress_cb,
            )
        elif task_type == "retexture":
            # Re-texture an existing shape with new material settings
            base_task_id = params.get("base_task_id")
            if not base_task_id:
                raise ValueError("base_task_id required for retexture")
            base_dir = os.path.join(GALLERY_DIR, base_task_id)
            mesh_path = os.path.join(base_dir, "model_initial.glb")
            input_image_path = os.path.join(base_dir, "input_image.png")
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Shape mesh not found: {mesh_path}")
            if not os.path.exists(input_image_path):
                raise FileNotFoundError(f"Input image not found: {input_image_path}")
            gen_data = {
                "mesh_path": mesh_path,
                "input_image_path": input_image_path,
                "texture_enabled": True,
                "roughness_offset": params.get("roughness_offset", 0.0),
                "metallic_scale": params.get("metallic_scale", 1.0),
            }
        elif task_type == "quick_adjust":
            # Quick material adjust — no GPU, just modify existing textures
            base_task_id = params.get("base_task_id")
            if not base_task_id:
                raise ValueError("base_task_id required for quick_adjust")
            base_dir = os.path.join(GALLERY_DIR, base_task_id)
            obj_path = os.path.join(base_dir, "textured.obj")
            if not os.path.exists(obj_path):
                raise FileNotFoundError(f"Textured OBJ not found in {base_dir}")
            gen_data = {"base_task_dir": base_dir}
        elif task_type == "export":
            # Re-export an existing generation — load gen_data from stored state
            raise NotImplementedError("Re-export from stored state not yet implemented")
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # -- Export ------------------------------------------
        task["status"] = TaskStatus.EXTRACTING
        progress_cb("Extracting mesh", 0.7)

        formats = params.get("formats", ["glb"])
        export_params = {
            "simplify": params.get("mesh_simplify", 0.95),
            "texture_size": params.get("texture_size", 1024),
            "fill_holes": params.get("fill_holes", True),
            "fill_holes_max_size": params.get("fill_holes_max_size", 0.04),
            "target_face_count": params.get("target_face_count", 0),
            "remove_floaters": params.get("remove_floaters", True),
        }
        if task_type == "quick_adjust" and hasattr(engine, "quick_adjust_materials"):
            export_paths = engine.quick_adjust_materials(
                base_task_dir=gen_data["base_task_dir"],
                output_dir=output_dir,
                roughness_offset=params.get("roughness_offset", 0.0),
                metallic_scale=params.get("metallic_scale", 1.0),
                progress_callback=progress_cb,
            )
        elif task_type == "retexture" and hasattr(engine, "retexture_mesh"):
            export_paths = engine.retexture_mesh(
                generation_data=gen_data,
                output_dir=output_dir,
                formats=formats,
                progress_callback=progress_cb,
            )
        else:
            export_paths = engine.export_mesh(
                generation_data=gen_data,
                output_dir=output_dir,
                formats=formats,
                progress_callback=progress_cb,
                **export_params,
            )

        # -- Render preview video ----------------------------
        progress_cb("Rendering preview", 0.95)
        video_path = os.path.join(output_dir, "preview.mp4")
        try:
            engine.render_preview(gen_data, video_path, resolution=512, num_frames=120)
        except (NotImplementedError, Exception) as e:
            logger.warning(f"Preview render skipped: {e}")
            video_path = None

        # -- Save thumbnail and input image for re-texture ------
        thumb_path = None
        thumb_source = params.get("original_image_path") or params.get("image_path")
        if thumb_source:
            thumb_path = os.path.join(output_dir, "thumbnail.png")
            from PIL import Image
            img = Image.open(thumb_source).convert("RGB")
            img.thumbnail((256, 256))
            img.save(thumb_path)
            # Save full-res input image for potential re-texturing
            input_save_path = os.path.join(output_dir, "input_image.png")
            Image.open(thumb_source).convert("RGB").save(input_save_path)

        elapsed = time.time() - start_time

        # -- Build result ------------------------------------
        exports = []
        for fmt, path in export_paths.items():
            exports.append({
                "format": fmt,
                "filename": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            })

        return {
            "seed": seed,
            "exports": exports,
            "video_path": video_path,
            "thumbnail_path": thumb_path,
            "generation_time_seconds": round(elapsed, 1),
        }

    # -- Gallery --------------------------------------------

    def _gallery_index_path(self) -> str:
        return os.path.join(GALLERY_DIR, "index.json")

    def _load_gallery_index(self):
        path = self._gallery_index_path()
        if os.path.exists(path):
            with open(path) as f:
                self._gallery_index = json.load(f)
        else:
            self._gallery_index = []

    def _save_gallery_index(self):
        path = self._gallery_index_path()
        with open(path, "w") as f:
            json.dump(self._gallery_index, f, indent=2)

    def _save_to_gallery(self, task: dict):
        if task["status"] != TaskStatus.COMPLETED or not task.get("result"):
            return
        result = task["result"]
        entry = {
            "task_id": task["id"],
            "type": task["type"],
            "model": task["params"].get("model", "trellis-image-to-3d"),
            "seed": result.get("seed", 0),
            "exports": result.get("exports", []),
            "has_video": result.get("video_path") is not None,
            "has_thumbnail": result.get("thumbnail_path") is not None,
            "generation_time_seconds": result.get("generation_time_seconds"),
            "created_at": task.get("created_at", ""),
        }
        self._gallery_index.insert(0, entry)  # newest first
        self._save_gallery_index()

    def get_gallery(self, page: int = 1, per_page: int = 20) -> tuple[list[dict], int]:
        total = len(self._gallery_index)
        start = (page - 1) * per_page
        end = start + per_page
        return self._gallery_index[start:end], total

    def save_edited_to_gallery(self, glb_data: bytes, label: str = "Edited",
                               source_model: str | None = None, seed: int = 0) -> dict:
        """Save an edited GLB file to the gallery and return its index entry."""
        import uuid
        from datetime import datetime, timezone
        task_id = uuid.uuid4().hex[:12]
        item_dir = os.path.join(GALLERY_DIR, task_id)
        os.makedirs(item_dir, exist_ok=True)
        glb_path = os.path.join(item_dir, "model.glb")
        with open(glb_path, "wb") as f:
            f.write(glb_data)
        size_bytes = os.path.getsize(glb_path)
        entry = {
            "task_id": task_id,
            "type": "edited",
            "model": "edited",
            "seed": seed,
            "source_model": source_model,
            "exports": [{"format": "glb", "filename": "model.glb", "path": glb_path, "size_bytes": size_bytes}],
            "has_video": False,
            "has_thumbnail": False,
            "generation_time_seconds": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": label,
        }
        self._gallery_index.insert(0, entry)
        self._save_gallery_index()
        return entry

    def delete_gallery_item(self, task_id: str) -> bool:
        import shutil
        item_dir = os.path.join(GALLERY_DIR, task_id)
        self._gallery_index = [i for i in self._gallery_index if i["task_id"] != task_id]
        self._save_gallery_index()
        if os.path.isdir(item_dir):
            shutil.rmtree(item_dir)
        if task_id in self.tasks:
            del self.tasks[task_id]
        return True
