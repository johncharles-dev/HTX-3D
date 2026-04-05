from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum


# ── Enums ──────────────────────────────────────────────

class ModelType(str, Enum):
    TRELLIS_IMAGE = "trellis-image-to-3d"
    TRELLIS_TEXT = "trellis-text-to-3d"
    HUNYUAN_IMAGE = "hunyuan-image-to-3d"
    SAM3D_IMAGE = "sam3d-image-to-3d"


class ExportFormat(str, Enum):
    GLB = "glb"
    OBJ = "obj"
    STL = "stl"
    PLY = "ply"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


class MultiImageMode(str, Enum):
    STOCHASTIC = "stochastic"
    MULTIDIFFUSION = "multidiffusion"


# ── Request Schemas ────────────────────────────────────

class GenerationSettings(BaseModel):
    """Common generation parameters for all models."""
    seed: int = Field(default=42, ge=0, le=2**31 - 1, description="Random seed for reproducibility")
    randomize_seed: bool = Field(default=True, description="Generate a random seed before running")

    # Stage 1: Sparse structure
    ss_steps: int = Field(default=12, ge=1, le=50, description="Sparse structure sampling steps")
    ss_guidance: float = Field(default=7.5, ge=0.0, le=10.0, description="Sparse structure guidance strength")

    # Stage 2: Structured latent
    slat_steps: int = Field(default=12, ge=1, le=50, description="Structured latent sampling steps")
    slat_guidance: float = Field(default=3.0, ge=0.0, le=10.0, description="Structured latent guidance strength")

    # Hunyuan-specific (optional, ignored by TRELLIS)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=100, description="Hunyuan inference steps")
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0, description="Hunyuan guidance scale")
    octree_resolution: Optional[int] = Field(default=None, description="Hunyuan octree resolution (128/256/384/512)")
    texture: Optional[bool] = Field(default=None, description="Enable PBR texture generation (Hunyuan)")


class ExportSettings(BaseModel):
    """Mesh extraction / export parameters."""
    formats: List[ExportFormat] = Field(default=[ExportFormat.GLB], description="Export formats to generate")
    mesh_simplify: float = Field(default=0.95, ge=0.8, le=0.99, description="Mesh simplification ratio (higher = more simplified)")
    texture_size: int = Field(default=1024, ge=512, le=4096, description="Texture resolution for GLB/OBJ")
    fill_holes: bool = Field(default=True, description="Fill holes in mesh (removes interior faces)")
    fill_holes_max_size: float = Field(default=0.04, ge=0.0, le=0.1, description="Maximum hole area to fill")


class ImageGenerateRequest(BaseModel):
    """Request body for single-image generation."""
    model: ModelType = Field(default=ModelType.TRELLIS_IMAGE)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


class MultiImageGenerateRequest(BaseModel):
    """Request body for multi-image generation."""
    model: ModelType = Field(default=ModelType.TRELLIS_IMAGE)
    mode: MultiImageMode = Field(default=MultiImageMode.STOCHASTIC, description="Multi-image fusion strategy")
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


class TextGenerateRequest(BaseModel):
    """Request body for text-to-3D generation."""
    prompt: str = Field(..., min_length=1, max_length=500, description="Text description of the 3D object")
    model: ModelType = Field(default=ModelType.TRELLIS_TEXT)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


class ExportRequest(BaseModel):
    """Request to export an existing generation in a new format."""
    task_id: str = Field(..., description="ID of a completed generation task")
    export: ExportSettings = Field(default_factory=ExportSettings)


# ── Response Schemas ───────────────────────────────────

class TaskResponse(BaseModel):
    """Response after submitting a generation request."""
    task_id: str
    status: TaskStatus
    message: str = ""


class ExportFile(BaseModel):
    """A single exported file."""
    format: ExportFormat
    filename: str
    url: str
    size_bytes: int


class GenerationResult(BaseModel):
    """Full result of a completed generation."""
    task_id: str
    status: TaskStatus
    model: ModelType
    seed: int
    preview_video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    exports: List[ExportFile] = []
    generation_time_seconds: Optional[float] = None
    error: Optional[str] = None
    created_at: Optional[str] = None


class ProgressUpdate(BaseModel):
    """WebSocket progress message."""
    task_id: str
    status: TaskStatus
    stage: str = ""
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress 0.0 to 1.0")
    message: str = ""


# ── Gallery Schemas ────────────────────────────────────

class GalleryItem(BaseModel):
    """A single gallery entry."""
    task_id: str
    model: ModelType
    thumbnail_url: Optional[str] = None
    preview_video_url: Optional[str] = None
    exports: List[ExportFile] = []
    seed: int = 0
    generation_time_seconds: Optional[float] = None
    created_at: str = ""


class GalleryResponse(BaseModel):
    """Paginated gallery response."""
    items: List[GalleryItem]
    total: int
    page: int
    per_page: int


# ── System Schemas ─────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    gpu: dict
    models_loaded: List[str]
    engines_registered: List[str] = []
    active_engine: Optional[str] = None
    queue_size: int
