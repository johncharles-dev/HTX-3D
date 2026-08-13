"""TRELLIS.2 engine — thin proxy to the host-side TRELLIS.2 microservice.

TRELLIS.2 cannot run inside this container (it needs the host's torch 2.10+cu128 / from-source
FlashAttention-2 stack for Blackwell sm120). Instead the 4B model runs as an HTTP service on the
host (see services/trellis2/ in this repo for the service and its setup), and this engine forwards
the image to it and writes back the returned GLB. To the rest of the backend it behaves like any
other engine.

The heavy call (run + GLB export in the service) happens in export_mesh(), because that is where
texture_size / target_face_count arrive; generate_from_image() only stashes the request.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from .base import BaseEngine
from ..config import TRELLIS2_SERVICE_URL

logger = logging.getLogger(__name__)

# Health-check timeout on the load path, where the host may be mid-generation and slow to
# answer. The startup reachability check overrides this with TRELLIS2_PROBE_TIMEOUT.
LOAD_PROBE_TIMEOUT_S = 5.0


class Trellis2Engine(BaseEngine):
    """Proxy to the host TRELLIS.2-4B microservice."""

    name = "trellis2"
    loaded = False

    def __init__(self, service_url: str = None):
        self.service_url = (service_url or TRELLIS2_SERVICE_URL).rstrip("/")
        self._weights_dir = None  # for task-manager swap compatibility

    def probe(self, timeout: float = LOAD_PROBE_TIMEOUT_S) -> dict:
        """Health-check the host service and return its /health payload.

        Deliberately does NOT set self.loaded. TaskManager.register_engine() derives the
        active engine from that flag and the swap logic keys off it, so a startup
        reachability check must not flip it. Raises RuntimeError naming the resolved URL,
        because a wrong TRELLIS2_SERVICE_URL otherwise fails as a hang or a connection to
        an unrelated host — neither of which is diagnosable from the logs.

        The default timeout suits the load path, where the host may be busy. The startup
        check passes the shorter TRELLIS2_PROBE_TIMEOUT instead.
        """
        try:
            r = requests.get(f"{self.service_url}/health", timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise RuntimeError(
                f"TRELLIS.2 host service not reachable at {self.service_url}: {e}. "
                "Start it on the host and check TRELLIS2_SERVICE_URL — setup steps are in "
                "services/trellis2/README.md."
            ) from e

    def load(self, weights_dir: str = None, device: str = "cuda") -> None:
        # No local weights — just verify the host service is reachable.
        if self.loaded:
            return
        logger.info(f"TRELLIS.2 service reachable at {self.service_url}: {self.probe()}")
        self.loaded = True

    def unload(self) -> None:
        # The model lives in the host service; nothing to free container-side.
        self.loaded = False

    def generate_from_image(
        self,
        image_path: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        # Defer the actual generation to export_mesh (where texture_size/face-count are known).
        if progress_callback:
            progress_callback("Queued on TRELLIS.2 service", 0.1)
        return {"image_path": image_path, "seed": int(seed)}

    def export_mesh(
        self,
        generation_data: dict,
        output_dir: str,
        formats: list[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **export_params,
    ) -> dict[str, Path]:
        image_path = generation_data["image_path"]
        seed = generation_data.get("seed", 0)
        # The tool's shared default texture is 1024; TRELLIS.2 bakes 4K-level detail and looks soft
        # at 1024, so treat the low default as "auto" and give it 4096. Explicit 2048+ are honored.
        texture_size = int(export_params.get("texture_size", 1024))
        if texture_size <= 1024:
            texture_size = 4096
        # Face budget: honor an explicit target, else a high default so geometry stays crisp.
        target = int(export_params.get("target_face_count", 0) or 0)
        decimation_target = target if target > 0 else 200000

        if progress_callback:
            progress_callback("Generating on TRELLIS.2 service", 0.35)

        t0 = time.time()
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{self.service_url}/generate",
                files={"image": (Path(image_path).name, f, "application/octet-stream")},
                data={"seed": seed, "texture_size": texture_size,
                      "decimation_target": decimation_target},
                timeout=900,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"TRELLIS.2 service error {resp.status_code}: {resp.text[:200]}")
        logger.info(f"TRELLIS.2 service returned {len(resp.content)/1e6:.1f}MB in {time.time()-t0:.1f}s "
                    f"(gen {resp.headers.get('X-Gen-Seconds','?')}s, vram {resp.headers.get('X-Peak-VRAM-GB','?')}GB)")

        if progress_callback:
            progress_callback("Writing mesh", 0.9)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        glb_path = out / "model.glb"
        glb_path.write_bytes(resp.content)
        results: dict[str, Path] = {"glb": glb_path}

        # Optional extra formats via trimesh conversion (glb is authoritative).
        extra = [f for f in (formats or []) if f.lower() in ("obj", "stl", "ply") ]
        if extra:
            try:
                import trimesh
                scene = trimesh.load(str(glb_path))
                for fmt in extra:
                    p = out / f"model.{fmt.lower()}"
                    scene.export(str(p))
                    results[fmt.lower()] = p
            except Exception as e:
                logger.warning(f"TRELLIS.2 extra-format export failed ({extra}): {e}")

        if progress_callback:
            progress_callback("Done", 1.0)
        return results

    # --- unsupported task types ------------------------------------------
    def generate_from_images(self, image_paths, seed, mode, progress_callback=None, **engine_params) -> dict:
        raise NotImplementedError("TRELLIS.2 engine supports single-image generation only")

    def generate_from_text(self, prompt, seed, progress_callback=None, **engine_params) -> dict:
        raise NotImplementedError("TRELLIS.2 engine supports single-image generation only")

    def render_preview(self, generation_data, output_path, resolution=512, num_frames=120) -> str:
        raise NotImplementedError("TRELLIS.2 engine does not render preview server-side")
