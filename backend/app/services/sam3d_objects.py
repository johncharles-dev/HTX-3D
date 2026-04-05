"""SAM 3D Objects engine — Meta's 3D reconstruction from image + mask.

Uses the InferencePipelinePointMap to generate Gaussian splats and textured GLB
meshes from a single RGBA image (with segmentation mask in alpha channel).
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


class Sam3DObjectsEngine(BaseEngine):
    """SAM 3D Objects (Meta) — image+mask to 3D reconstruction."""

    name = "sam3d"
    loaded = False

    def __init__(self, sam3d_dir: str):
        self.sam3d_dir = sam3d_dir
        self._weights_dir = sam3d_dir  # for task manager swap compatibility
        self._inference = None

    def load(self, weights_dir: str = None, device: str = "cuda") -> None:
        if self.loaded:
            return

        logger.info("Loading SAM 3D Objects engine...")
        start = time.time()

        # Add to sys.path for imports
        notebook_dir = os.path.join(self.sam3d_dir, "notebook")
        if self.sam3d_dir not in sys.path:
            sys.path.insert(0, self.sam3d_dir)
        if notebook_dir not in sys.path:
            sys.path.insert(0, notebook_dir)

        # Set required env vars (CUDA_HOME needed by pytorch3d, no conda in Docker)
        os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", os.environ.get("CUDA_HOME", "/usr/local/cuda")))
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")

        from inference import Inference

        # Search for pipeline config in multiple locations
        search_paths = [
            os.path.join(self.sam3d_dir, "checkpoints", "hf", "pipeline.yaml"),
            os.path.join(weights_dir or "", "sam3d-objects-hf", "pipeline.yaml"),
            "/data/models/sam-3d-objects/hf/pipeline.yaml",
            "/app/weights/sam3d-objects-hf/pipeline.yaml",
        ]
        config_path = None
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break
        if not config_path:
            raise FileNotFoundError(
                f"SAM 3D Objects config not found. Searched: {search_paths}"
            )

        self._inference = Inference(config_path, compile=False)
        self.loaded = True
        logger.info(f"SAM 3D Objects loaded in {time.time() - start:.1f}s")

    def unload(self) -> None:
        if not self.loaded:
            return

        logger.info("Unloading SAM 3D Objects...")
        del self._inference
        self._inference = None
        self.loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("SAM 3D Objects unloaded")

    def generate_from_image(
        self,
        image_path: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        """Generate 3D from image. Extracts mask from RGBA alpha channel."""
        if not self.loaded:
            raise RuntimeError("SAM 3D Objects engine not loaded")

        if progress_callback:
            progress_callback("Loading image", 0.05)

        img = Image.open(image_path)
        arr = np.array(img)

        # Extract mask from alpha channel if RGBA
        if arr.shape[2] == 4 and not np.all(arr[:, :, 3] == 255):
            image_rgb = arr[:, :, :3]
            mask = arr[:, :, 3] > 0
            logger.info("Using segmentation mask from RGBA alpha channel")
        else:
            # No segmentation — use full image as mask
            image_rgb = np.array(img.convert("RGB"))
            mask = np.ones(image_rgb.shape[:2], dtype=bool)
            logger.info("No segmentation mask found, using full image")

        if progress_callback:
            progress_callback("Running SAM 3D Objects", 0.1)

        # Extract engine-specific params
        with_texture_baking = engine_params.get("sam3d_texture_baking", True)
        use_vertex_color = engine_params.get("sam3d_vertex_color", False)
        stage1_steps = engine_params.get("sam3d_stage1_steps", 25)
        stage2_steps = engine_params.get("sam3d_stage2_steps", 25)

        # Call the pipeline directly for full control
        rgba = self._inference.merge_mask_to_rgba(image_rgb, mask)
        output = self._inference._pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=with_texture_baking,
            with_layout_postprocess=False,
            use_vertex_color=use_vertex_color,
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
        )

        if progress_callback:
            progress_callback("Generation complete", 0.65)

        return {
            "gs": output.get("gs"),
            "glb": output.get("glb"),
            "output": output,
        }

    def generate_from_images(
        self,
        image_paths: list[str],
        seed: int,
        mode: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        raise NotImplementedError("SAM 3D Objects supports single image+mask only")

    def generate_from_text(
        self,
        prompt: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        raise NotImplementedError("SAM 3D Objects does not support text-to-3D")

    def export_mesh(
        self,
        generation_data: dict,
        output_dir: str,
        formats: list[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict[str, Path]:
        """Export GLB and PLY from generation data."""
        if progress_callback:
            progress_callback("Exporting meshes", 0.75)

        exports = {}
        output_dir = Path(output_dir)

        # GLB export — from trimesh mesh object
        if "glb" in formats and generation_data.get("glb") is not None:
            glb_path = output_dir / "model.glb"
            generation_data["glb"].export(str(glb_path))
            exports["glb"] = glb_path
            logger.info(f"Exported GLB: {glb_path}")

        # PLY export — from Gaussian splat
        if "ply" in formats and generation_data.get("gs") is not None:
            ply_path = output_dir / "model.ply"
            generation_data["gs"].save_ply(str(ply_path))
            exports["ply"] = ply_path
            logger.info(f"Exported PLY: {ply_path}")

        # STL/OBJ from GLB mesh if available
        if generation_data.get("glb") is not None:
            mesh = generation_data["glb"]
            if "stl" in formats:
                stl_path = output_dir / "model.stl"
                mesh.export(str(stl_path), file_type="stl")
                exports["stl"] = stl_path
            if "obj" in formats:
                obj_path = output_dir / "model.obj"
                mesh.export(str(obj_path), file_type="obj")
                exports["obj"] = obj_path

        if progress_callback:
            progress_callback("Export complete", 0.9)

        if not exports:
            raise RuntimeError("No exportable data produced")

        return exports

    def render_preview(
        self,
        generation_data: dict,
        output_path: str,
        resolution: int = 512,
        num_frames: int = 120,
    ) -> str:
        """Render preview video from Gaussian splat."""
        gs = generation_data.get("gs")
        if gs is None:
            raise NotImplementedError("No Gaussian splat data for preview rendering")

        # Import render utilities
        notebook_dir = os.path.join(self.sam3d_dir, "notebook")
        if notebook_dir not in sys.path:
            sys.path.insert(0, notebook_dir)

        from inference import render_video, ready_gaussian_for_video_rendering

        # Normalize gaussian for rendering
        scene_gs = ready_gaussian_for_video_rendering(gs, in_place=False)

        # Render frames
        frames = render_video(
            scene_gs,
            resolution=resolution,
            num_frames=num_frames,
        )

        # Save as video
        import imageio
        writer = imageio.get_writer(output_path, fps=30)
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = (frame.cpu().numpy() * 255).astype(np.uint8)
            elif isinstance(frame, np.ndarray) and frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()

        logger.info(f"Preview video saved: {output_path}")
        return output_path
