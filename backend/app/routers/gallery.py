"""Gallery and file download endpoints."""

import os
import logging

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from ..config import GALLERY_DIR
from ..models.schemas import GalleryResponse, GalleryItem, ExportFile
from ..dependencies import get_task_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["gallery"])


@router.get("/gallery", response_model=GalleryResponse)
async def list_gallery(
    page: int = 1,
    per_page: int = 20,
    task_manager=Depends(get_task_manager),
):
    """List past generations with pagination."""
    items_raw, total = task_manager.get_gallery(page, per_page)
    items = []
    for raw in items_raw:
        task_id = raw["task_id"]
        exports = []
        for exp in raw.get("exports", []):
            exports.append(ExportFile(
                format=exp["format"],
                filename=exp["filename"],
                url=f"/api/download/{task_id}/{exp['filename']}",
                size_bytes=exp.get("size_bytes", 0),
            ))
        items.append(GalleryItem(
            task_id=task_id,
            model=raw.get("model", "trellis-image-to-3d"),
            thumbnail_url=f"/api/download/{task_id}/thumbnail.png" if raw.get("has_thumbnail") else None,
            preview_video_url=f"/api/download/{task_id}/preview.mp4" if raw.get("has_video") else None,
            exports=exports,
            seed=raw.get("seed", 0),
            generation_time_seconds=raw.get("generation_time_seconds"),
            created_at=raw.get("created_at", ""),
        ))

    return GalleryResponse(items=items, total=total, page=page, per_page=per_page)


@router.delete("/gallery/{task_id}")
async def delete_gallery_item(task_id: str, task_manager=Depends(get_task_manager)):
    """Delete a gallery item and its files."""
    task_manager.delete_gallery_item(task_id)
    return {"ok": True, "message": f"Deleted {task_id}"}


@router.get("/download/{task_id}/{filename}")
async def download_file(task_id: str, filename: str):
    """Download an exported file or preview."""
    file_path = os.path.join(GALLERY_DIR, task_id, filename)

    # Also check the obj subdirectory
    if not os.path.isfile(file_path):
        file_path = os.path.join(GALLERY_DIR, task_id, "obj", filename)

    if not os.path.isfile(file_path):
        raise HTTPException(404, "File not found")

    # Determine media type
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        ".glb": "model/gltf-binary",
        ".obj": "text/plain",
        ".stl": "model/stl",
        ".ply": "application/octet-stream",
        ".zip": "application/zip",
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".jpg": "image/jpeg",
    }

    return FileResponse(
        file_path,
        media_type=media_types.get(ext, "application/octet-stream"),
        filename=filename,
    )
