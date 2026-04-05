"""SAM3 segmentation endpoints.

Provides interactive segmentation workflow:
1. POST /start — upload image, start session
2. POST /text — segment by text prompt
3. POST /box — segment by bounding box
4. POST /point — segment by point click
5. POST /reset — clear prompts
6. POST /confirm — apply mask, produce RGBA image
"""

import os
import shutil
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse

from ..config import TEMP_DIR
from ..models.segment_schemas import (
    SegmentStartResponse,
    TextSegmentRequest,
    BoxSegmentRequest,
    PointSegmentRequest,
    SegmentResponse,
    MaskResult,
    ConfirmSegmentRequest,
    ConfirmSegmentResponse,
)
from ..dependencies import get_sam3_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/segment", tags=["segmentation"])


def _save_upload(upload: UploadFile, session_id: str) -> str:
    """Save uploaded image for segmentation."""
    upload_dir = os.path.join(TEMP_DIR, "segmented", session_id)
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(upload.filename or "image.png")[1] or ".png"
    path = os.path.join(upload_dir, f"input{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


@router.post("/start", response_model=SegmentStartResponse)
async def start_session(
    image: UploadFile = File(..., description="Image to segment"),
    sam3=Depends(get_sam3_service),
):
    """Upload an image and start a segmentation session."""
    temp_id = os.urandom(6).hex()
    image_path = _save_upload(image, temp_id)

    try:
        result = sam3.start_session(image_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to start segmentation: {e}")

    return SegmentStartResponse(
        session_id=result["session_id"],
        width=result["width"],
        height=result["height"],
        message="Session started. Use /text, /box, or /point to segment.",
    )


@router.post("/text", response_model=SegmentResponse)
async def segment_text(
    req: TextSegmentRequest,
    sam3=Depends(get_sam3_service),
):
    """Segment objects matching a text prompt."""
    try:
        result = sam3.segment_text(req.session_id, req.prompt)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    return SegmentResponse(
        session_id=result["session_id"],
        masks=[MaskResult(**m) for m in result["masks"]],
        overlay_url=f"/api/segment/overlay/{result['session_id']}" if result.get("overlay_path") else "",
        message=f"Found {len(result['masks'])} mask(s)",
    )


@router.post("/box", response_model=SegmentResponse)
async def segment_box(
    req: BoxSegmentRequest,
    sam3=Depends(get_sam3_service),
):
    """Segment objects within a bounding box."""
    try:
        result = sam3.segment_box(req.session_id, req.box, req.label)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    return SegmentResponse(
        session_id=result["session_id"],
        masks=[MaskResult(**m) for m in result["masks"]],
        overlay_url=f"/api/segment/overlay/{result['session_id']}" if result.get("overlay_path") else "",
        message=f"Found {len(result['masks'])} mask(s)",
    )


@router.post("/point", response_model=SegmentResponse)
async def segment_point(
    req: PointSegmentRequest,
    sam3=Depends(get_sam3_service),
):
    """Segment objects at a point location."""
    try:
        result = sam3.segment_point(req.session_id, req.x, req.y, req.label)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    return SegmentResponse(
        session_id=result["session_id"],
        masks=[MaskResult(**m) for m in result["masks"]],
        overlay_url=f"/api/segment/overlay/{result['session_id']}" if result.get("overlay_path") else "",
        message=f"Found {len(result['masks'])} mask(s)",
    )


@router.post("/reset")
async def reset_prompts(
    session_id: str,
    sam3=Depends(get_sam3_service),
):
    """Reset all prompts for a session."""
    try:
        return sam3.reset_prompts(session_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/confirm", response_model=ConfirmSegmentResponse)
async def confirm_mask(
    req: ConfirmSegmentRequest,
    sam3=Depends(get_sam3_service),
):
    """Confirm a mask selection and produce RGBA output."""
    try:
        segmented_path = sam3.confirm_mask(req.session_id, req.mask_index)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Mask confirmation failed: {e}")

    # Clean up session (frees memory for generation)
    sam3.cleanup_session(req.session_id)

    return ConfirmSegmentResponse(
        session_id=req.session_id,
        segmented_image_path=segmented_path,
        message="Segmented image ready. Use this path for generation.",
    )


@router.get("/overlay/{session_id}")
async def get_overlay(
    session_id: str,
    sam3=Depends(get_sam3_service),
):
    """Get the overlay image for a session."""
    overlay_path = os.path.join(TEMP_DIR, "segmented", session_id, "overlay.png")
    if not os.path.exists(overlay_path):
        raise HTTPException(404, "Overlay not found")
    return FileResponse(overlay_path, media_type="image/png")
