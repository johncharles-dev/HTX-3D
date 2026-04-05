"""Generation API endpoints."""

import os
import shutil
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import ValidationError

from ..config import TEMP_DIR
from ..models.schemas import (
    TaskResponse,
    TaskStatus,
    GenerationResult,
    ExportFile,
    ImageGenerateRequest,
    MultiImageGenerateRequest,
    TextGenerateRequest,
    ExportRequest,
    GenerationSettings,
    ExportSettings,
)
from ..dependencies import get_task_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["generation"])


def _save_upload(upload: UploadFile, task_id: str, index: int = 0) -> str:
    """Save an uploaded file to temp and return the path."""
    upload_dir = os.path.join(TEMP_DIR, "uploads", task_id)
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(upload.filename or "image.png")[1] or ".png"
    path = os.path.join(upload_dir, f"input_{index}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def _model_id_for_engine(engine: str, task_type: str) -> str:
    """Map engine name + task type to a ModelType value."""
    if engine == "hunyuan":
        return "hunyuan-image-to-3d"
    if task_type == "text":
        return "trellis-text-to-3d"
    return "trellis-image-to-3d"


# -- Image-to-3D -------------------------------------------

@router.post("/generate/image", response_model=TaskResponse)
async def generate_from_image(
    image: UploadFile = File(..., description="Input image (PNG/JPG, ideally with transparent background)"),
    engine: str = Form("trellis", description="Engine to use: trellis or hunyuan"),
    seed: int = Form(42),
    randomize_seed: bool = Form(True),
    # TRELLIS params
    ss_steps: int = Form(12),
    ss_guidance: float = Form(7.5),
    slat_steps: int = Form(12),
    slat_guidance: float = Form(3.0),
    # Hunyuan params
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    octree_resolution: Optional[int] = Form(None),
    texture: Optional[bool] = Form(None),
    # Export params
    formats: str = Form("glb", description="Comma-separated export formats: glb,obj,stl,ply"),
    mesh_simplify: float = Form(0.95),
    texture_size: int = Form(1024),
    task_manager=Depends(get_task_manager),
):
    """Generate a 3D model from a single image.

    Upload an image and configure generation parameters.
    Returns a task ID that can be polled for progress via WebSocket or GET.
    """
    # Save upload
    temp_id = os.urandom(6).hex()
    image_path = _save_upload(image, temp_id)

    format_list = [f.strip() for f in formats.split(",") if f.strip()]

    params = {
        "model": _model_id_for_engine(engine, "image"),
        "engine": engine,
        "image_path": image_path,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "ss_steps": ss_steps,
        "ss_guidance": ss_guidance,
        "slat_steps": slat_steps,
        "slat_guidance": slat_guidance,
        "formats": format_list,
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size,
    }
    # Add Hunyuan params only if provided
    if num_inference_steps is not None:
        params["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        params["guidance_scale"] = guidance_scale
    if octree_resolution is not None:
        params["octree_resolution"] = octree_resolution
    if texture is not None:
        params["texture"] = texture

    task_id = task_manager.submit_task("image", params)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message=f"Task queued. Position: {task_manager.queue_size}",
    )


# -- Multi-Image-to-3D ------------------------------------

@router.post("/generate/multi-image", response_model=TaskResponse)
async def generate_from_multi_image(
    images: list[UploadFile] = File(..., description="2-4 images of the same object from different views"),
    engine: str = Form("trellis", description="Engine to use: trellis or hunyuan"),
    seed: int = Form(42),
    randomize_seed: bool = Form(True),
    mode: str = Form("stochastic", description="Multi-image fusion: stochastic or multidiffusion"),
    # TRELLIS params
    ss_steps: int = Form(12),
    ss_guidance: float = Form(7.5),
    slat_steps: int = Form(12),
    slat_guidance: float = Form(3.0),
    # Hunyuan params
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    octree_resolution: Optional[int] = Form(None),
    texture: Optional[bool] = Form(None),
    # Export params
    formats: str = Form("glb"),
    mesh_simplify: float = Form(0.95),
    texture_size: int = Form(1024),
    task_manager=Depends(get_task_manager),
):
    """Generate a 3D model from multiple images of the same object."""
    if len(images) < 2 or len(images) > 4:
        raise HTTPException(400, "Provide 2-4 images")

    temp_id = os.urandom(6).hex()
    image_paths = [_save_upload(img, temp_id, i) for i, img in enumerate(images)]
    format_list = [f.strip() for f in formats.split(",") if f.strip()]

    params = {
        "model": _model_id_for_engine(engine, "image"),
        "engine": engine,
        "image_paths": image_paths,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "mode": mode,
        "ss_steps": ss_steps,
        "ss_guidance": ss_guidance,
        "slat_steps": slat_steps,
        "slat_guidance": slat_guidance,
        "formats": format_list,
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size,
    }
    if num_inference_steps is not None:
        params["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        params["guidance_scale"] = guidance_scale
    if octree_resolution is not None:
        params["octree_resolution"] = octree_resolution
    if texture is not None:
        params["texture"] = texture

    task_id = task_manager.submit_task("multi_image", params)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message=f"Task queued. Position: {task_manager.queue_size}",
    )


# -- Text-to-3D -------------------------------------------

@router.post("/generate/text", response_model=TaskResponse)
async def generate_from_text(
    prompt: str = Form(..., description="Text description of the 3D object to generate"),
    engine: str = Form("trellis", description="Engine to use (only trellis supports text)"),
    seed: int = Form(42),
    randomize_seed: bool = Form(True),
    ss_steps: int = Form(12),
    ss_guidance: float = Form(7.5),
    slat_steps: int = Form(12),
    slat_guidance: float = Form(3.0),
    formats: str = Form("glb"),
    mesh_simplify: float = Form(0.95),
    texture_size: int = Form(1024),
    task_manager=Depends(get_task_manager),
):
    """Generate a 3D model from a text description."""
    if engine == "hunyuan":
        raise HTTPException(400, "Hunyuan3D does not support text-to-3D. Use image-to-3D instead.")

    format_list = [f.strip() for f in formats.split(",") if f.strip()]

    task_id = task_manager.submit_task("text", {
        "model": "trellis-text-to-3d",
        "engine": "trellis",
        "prompt": prompt,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "ss_steps": ss_steps,
        "ss_guidance": ss_guidance,
        "slat_steps": slat_steps,
        "slat_guidance": slat_guidance,
        "formats": format_list,
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size,
    })

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message=f"Task queued. Position: {task_manager.queue_size}",
    )


# -- Text-Guided Edit (Variant) ---------------------------

@router.post("/generate/edit", response_model=TaskResponse)
async def edit_with_text(
    prompt: str = Form(..., description="Text describing the desired changes"),
    seed: int = Form(42),
    randomize_seed: bool = Form(True),
    slat_steps: int = Form(12),
    slat_guidance: float = Form(3.0),
    base_task_id: str = Form("", description="Task ID of a previous generation to edit"),
    mesh_file: UploadFile = File(None, description="Upload a mesh file (GLB/OBJ/PLY) to edit"),
    formats: str = Form("glb"),
    mesh_simplify: float = Form(0.95),
    texture_size: int = Form(1024),
    task_manager=Depends(get_task_manager),
):
    """Edit an existing 3D model with a text prompt.

    Preserves the base shape and re-generates appearance/details
    guided by the text prompt. Provide either a previous task ID
    or upload a mesh file as the base model.
    """
    mesh_file_path = None
    if mesh_file and mesh_file.filename:
        temp_id = os.urandom(6).hex()
        mesh_file_path = _save_upload(mesh_file, temp_id)

    if not base_task_id and not mesh_file_path:
        raise HTTPException(400, "Provide either base_task_id or upload a mesh file")

    format_list = [f.strip() for f in formats.split(",") if f.strip()]

    task_id = task_manager.submit_task("edit", {
        "model": "trellis-text-to-3d",
        "engine": "trellis",
        "prompt": prompt,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "slat_steps": slat_steps,
        "slat_guidance": slat_guidance,
        "base_task_id": base_task_id or None,
        "mesh_file_path": mesh_file_path,
        "formats": format_list,
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size,
    })

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message=f"Edit task queued. Position: {task_manager.queue_size}",
    )


# -- Task Status -------------------------------------------

@router.get("/task/{task_id}", response_model=GenerationResult)
async def get_task_status(task_id: str, task_manager=Depends(get_task_manager)):
    """Get the status and results of a generation task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    result = task.get("result") or {}
    exports = []
    for exp in result.get("exports", []):
        exports.append(ExportFile(
            format=exp["format"],
            filename=exp["filename"],
            url=f"/api/download/{task_id}/{exp['filename']}",
            size_bytes=exp["size_bytes"],
        ))

    return GenerationResult(
        task_id=task_id,
        status=task["status"],
        model=task["params"].get("model", "trellis-image-to-3d"),
        seed=result.get("seed", task["params"].get("seed", 0)),
        preview_video_url=f"/api/download/{task_id}/preview.mp4" if result.get("video_path") else None,
        thumbnail_url=f"/api/download/{task_id}/thumbnail.png" if result.get("thumbnail_path") else None,
        exports=exports,
        generation_time_seconds=result.get("generation_time_seconds"),
        error=task.get("error"),
        created_at=task.get("created_at"),
    )
