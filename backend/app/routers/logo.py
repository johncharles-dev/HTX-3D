"""Logo bake endpoint — project a PNG logo into a model's albedo texture.

The "bake into surface" path for the logo decal feature: the frontend sends the
base model GLB it currently has loaded, the logo PNG, and the placement(s) in
the GLB scene-root frame. We project the logo into the albedo (optionally
inpainting the old content first) and register the result as a new gallery item.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import Optional

from ..models.schemas import GalleryItem, ExportFile
from ..dependencies import get_task_manager
from ..services.logo_bake import bake_logos

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["logo"])


@router.post("/logo/bake")
async def bake_logo(
    file: UploadFile = File(...),
    logo: UploadFile = File(...),
    placements: str = Form(...),
    smudge: bool = Form(True),
    target_resolution: int = Form(4096),
    label: Optional[str] = Form("Logo"),
    source_model: Optional[str] = Form(None),
    seed: Optional[int] = Form(0),
    task_manager=Depends(get_task_manager),
):
    """Bake a logo onto the model surface and save the result to the gallery."""
    glb_data = await file.read()
    logo_data = await logo.read()
    if len(glb_data) < 12:
        raise HTTPException(400, "Invalid GLB file")
    if len(logo_data) < 8:
        raise HTTPException(400, "Invalid logo image")
    try:
        placement_list = json.loads(placements)
    except json.JSONDecodeError:
        raise HTTPException(400, "placements must be valid JSON")
    if not isinstance(placement_list, list) or not placement_list:
        raise HTTPException(400, "placements must be a non-empty list")

    try:
        baked_glb = bake_logos(
            glb_data, logo_data, placement_list,
            smudge=smudge, target_resolution=target_resolution,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.exception("Logo bake failed")
        raise HTTPException(500, f"Logo bake failed: {e}")

    entry = task_manager.save_edited_to_gallery(
        baked_glb, label=label or "Logo",
        source_model=source_model, seed=seed or 0,
    )
    task_id = entry["task_id"]
    exports = [ExportFile(
        format=exp["format"],
        filename=exp["filename"],
        url=f"/api/download/{task_id}/{exp['filename']}",
        size_bytes=exp.get("size_bytes", 0),
    ) for exp in entry.get("exports", [])]
    return {
        "ok": True,
        "task_id": task_id,
        "item": GalleryItem(
            task_id=task_id,
            model="edited",
            exports=exports,
            seed=entry.get("seed", 0),
            generation_time_seconds=None,
            created_at=entry.get("created_at", ""),
        ),
    }
