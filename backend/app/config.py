import os
import torch
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", str(PROJECT_ROOT / "weights"))
GALLERY_DIR = os.environ.get("GALLERY_DIR", str(PROJECT_ROOT / "gallery"))
TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp/htx-3d")

# Ensure directories exist
for d in [WEIGHTS_DIR, GALLERY_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# Engine paths
TRELLIS_ENGINE_DIR = os.environ.get(
    "TRELLIS_ENGINE_DIR",
    str(BASE_DIR / "engines" / "trellis"),
)
HUNYUAN_ENGINE_DIR = os.environ.get(
    "HUNYUAN_ENGINE_DIR",
    str(BASE_DIR / "engines" / "hunyuan"),
)

# GPU / Hardware
def detect_gpu():
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "compute_capability": None,
            "vram_gb": 0,
            "is_blackwell": False,
        }
    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    return {
        "available": True,
        "name": props.name,
        "compute_capability": cc,
        "vram_gb": round(props.total_memory / (1024 ** 3), 1),
        "is_blackwell": props.major >= 12,
    }

# Server
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# Generation defaults
DEFAULT_SEED = 42
MAX_SEED = 2**31 - 1

# Model identifiers
TRELLIS_IMAGE_MODEL = os.environ.get("TRELLIS_IMAGE_MODEL", "JeffreyXiang/TRELLIS-image-large")
TRELLIS_TEXT_MODEL = os.environ.get("TRELLIS_TEXT_MODEL", "JeffreyXiang/TRELLIS-text-large")

# Processing
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "10"))
MAX_TEXTURE_SIZE = 4096
DEFAULT_TEXTURE_SIZE = 1024
