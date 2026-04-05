"""Pydantic models for SAM3 segmentation endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, List


class SegmentStartRequest(BaseModel):
    """Start a segmentation session by uploading an image."""
    pass  # Image comes as UploadFile


class SegmentStartResponse(BaseModel):
    """Response after starting a segmentation session."""
    session_id: str
    width: int
    height: int
    message: str = ""


class TextSegmentRequest(BaseModel):
    """Segment objects matching a text prompt."""
    session_id: str
    prompt: str = Field(..., min_length=1, max_length=200)


class BoxSegmentRequest(BaseModel):
    """Segment objects within a bounding box."""
    session_id: str
    box: List[float] = Field(..., min_length=4, max_length=4,
                             description="[center_x, center_y, width, height] normalized 0-1")
    label: bool = Field(default=True, description="True=positive (include), False=negative (exclude)")


class PointSegmentRequest(BaseModel):
    """Segment objects at a point location."""
    session_id: str
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate")
    label: bool = Field(default=True, description="True=positive, False=negative")


class PointsSegmentRequest(BaseModel):
    """Segment using accumulated point prompts."""
    session_id: str
    points: List[List[float]] = Field(..., description="List of [x, y] normalized 0-1")
    labels: List[int] = Field(..., description="List of 1 (add) or 0 (remove)")


class MaskResult(BaseModel):
    """A single segmentation mask result."""
    index: int
    score: float
    bbox: List[float] = Field(description="[x0, y0, x1, y1] in pixel coordinates")
    area_pixels: int


class SegmentResponse(BaseModel):
    """Response from a segmentation operation."""
    session_id: str
    masks: List[MaskResult]
    overlay_url: str = Field(description="URL to composite overlay image")
    message: str = ""


class ConfirmSegmentRequest(BaseModel):
    """Confirm a mask selection and produce RGBA output."""
    session_id: str
    mask_index: int = Field(default=0, ge=0, description="Index of the mask to use")


class ConfirmSegmentResponse(BaseModel):
    """Response after confirming segmentation."""
    session_id: str
    segmented_image_path: str = Field(description="Path to RGBA image with mask as alpha")
    message: str = ""
