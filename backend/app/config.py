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
SAM3D_OBJECTS_DIR = os.environ.get(
    "SAM3D_OBJECTS_DIR",
    str(BASE_DIR / "engines" / "sam3d_objects"),
)
# Extra search location for the SAM 3D Objects pipeline.yaml, used when the backend runs
# outside the container and the weights sit somewhere non-standard. Empty by default;
# inside the container the weights arrive at /app/weights/sam3d-objects-hf via the compose
# mount and this is not needed.
#
# NOT the compose mount source — that is SAM3D_HF_DIR, which is read on the host by
# docker-compose.yml and never passed into the container. The two are not interchangeable.
SAM3D_HF_PATH = os.environ.get("SAM3D_HF_PATH", "")
SAM3_DIR = os.environ.get(
    "SAM3_DIR",
    str(BASE_DIR / "engines" / "sam3"),
)
SAM3_BPE_PATH = os.environ.get(
    "SAM3_BPE_PATH",
    os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
)

# TRELLIS.2 runs as a host-side microservice (it needs the host's sm120 torch/FA2 stack).
# The container reaches it at host.docker.internal, which docker-compose.yml maps to the
# host gateway via `extra_hosts` — required on Linux Docker, where that name does not
# resolve by default. Running the backend outside the container instead needs
# TRELLIS2_SERVICE_URL=http://localhost:8710.
TRELLIS2_SERVICE_URL = os.environ.get("TRELLIS2_SERVICE_URL", "http://host.docker.internal:8710")
# Timeout (seconds) for the non-fatal startup reachability check only. Kept short because
# it only has to tell "not running" from "running", and every backend start pays it while
# TRELLIS.2 is not yet deployed. The load path uses its own, longer timeout.
TRELLIS2_PROBE_TIMEOUT = float(os.environ.get("TRELLIS2_PROBE_TIMEOUT", "2"))

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
