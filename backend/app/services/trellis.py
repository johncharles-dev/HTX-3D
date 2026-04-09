"""TRELLIS engine wrapper — bridges TRELLIS pipeline into the BaseEngine interface."""

import os
import sys
import gc
import time
import random
import logging
import zipfile
import torch
import numpy as np
import imageio
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


def _add_trellis_to_path(engine_dir: str):
    """Ensure the Trellis source directory is on sys.path."""
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)


class TrellisEngine(BaseEngine):
    """Wraps TRELLIS image-to-3D and text-to-3D pipelines."""

    name = "trellis"
    loaded = False

    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir
        self.image_pipeline = None
        self.text_pipeline = None
        self._device = "cuda"
        self._active_pipeline = None  # tracks which pipeline is on GPU

    # -- Lifecycle -------------------------------------------

    def load(self, weights_dir: str, device: str = "cuda") -> None:
        self._device = device
        _add_trellis_to_path(self.engine_dir)

        # Set environment for Blackwell compatibility
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        os.environ.setdefault("ATTN_BACKEND", "sdpa")

        # Initialize VRAMManager — auto-detects GPU VRAM and sets tier/dtype
        from trellis.utils.vram_manager import init_vram_manager
        vm = init_vram_manager(precision="auto", vram_tier="auto")
        logger.info(f"VRAMManager initialized: {vm}")

        from trellis.pipelines import TrellisImageTo3DPipeline

        image_model_path = os.path.join(weights_dir, "TRELLIS-image-large")
        if os.path.isdir(image_model_path):
            logger.info(f"Loading TRELLIS image model from local: {image_model_path}")
            self.image_pipeline = TrellisImageTo3DPipeline.from_pretrained(image_model_path)
        else:
            logger.info("Downloading TRELLIS image model from HuggingFace (first run)...")
            self.image_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-image-large"
            )
            # Save to persistent weights volume so it survives container restarts
            self._save_pipeline_weights(
                "JeffreyXiang/TRELLIS-image-large", image_model_path
            )

        if vm.dtype == torch.float16:
            self.image_pipeline.to_dtype(torch.float16)
        self.image_pipeline.cuda()
        self._active_pipeline = "image"
        logger.info("Image pipeline loaded to GPU (float16)" if vm.dtype == torch.float16 else "Image pipeline loaded to GPU")

        # Text pipeline — load to CPU only (swapped to GPU on demand)
        from trellis.pipelines import TrellisTextTo3DPipeline

        text_model_path = os.path.join(weights_dir, "TRELLIS-text-large")
        if os.path.isdir(text_model_path):
            logger.info(f"Loading TRELLIS text model from local: {text_model_path}")
            self.text_pipeline = TrellisTextTo3DPipeline.from_pretrained(text_model_path)
        else:
            logger.info("Downloading TRELLIS text model from HuggingFace (first run)...")
            self.text_pipeline = TrellisTextTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-text-large"
            )
            self._save_pipeline_weights(
                "JeffreyXiang/TRELLIS-text-large", text_model_path
            )

        if vm.dtype == torch.float16:
            self.text_pipeline.to_dtype(torch.float16)
        # Keep on CPU — will be swapped to GPU when needed
        logger.info("Text pipeline loaded to CPU (will swap to GPU on demand)")

        self.loaded = True
        logger.info("TRELLIS engine loaded successfully")

    @staticmethod
    def _save_pipeline_weights(repo_id: str, dest_path: str) -> None:
        """Snapshot HuggingFace model to persistent weights volume."""
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Saving {repo_id} to {dest_path} for future restarts...")
            snapshot_download(repo_id, local_dir=dest_path)
            logger.info(f"Saved {repo_id} to {dest_path}")
        except Exception as e:
            logger.warning(f"Could not save weights to {dest_path}: {e}")

    def _activate(self, pipeline_name: str) -> None:
        """Swap the requested pipeline to GPU, move the other to CPU."""
        if self._active_pipeline == pipeline_name:
            return
        if pipeline_name == "image" and self.image_pipeline:
            if self.text_pipeline:
                self.text_pipeline.cpu()
            torch.cuda.empty_cache()
            self.image_pipeline.cuda()
            self._active_pipeline = "image"
            logger.info("Swapped to image pipeline (GPU)")
        elif pipeline_name == "text" and self.text_pipeline:
            if self.image_pipeline:
                self.image_pipeline.cpu()
            torch.cuda.empty_cache()
            self.text_pipeline.cuda()
            self._active_pipeline = "text"
            logger.info("Swapped to text pipeline (GPU)")

    def unload(self) -> None:
        self.image_pipeline = None
        self.text_pipeline = None
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TRELLIS engine unloaded")

    # -- Generation ------------------------------------------

    def generate_from_image(
        self,
        image_path: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        if not self.image_pipeline:
            raise RuntimeError("Image pipeline not loaded")

        ss_steps = engine_params.get("ss_steps", 12)
        ss_guidance = engine_params.get("ss_guidance", 7.5)
        slat_steps = engine_params.get("slat_steps", 12)
        slat_guidance = engine_params.get("slat_guidance", 3.0)

        self._activate("image")

        if progress_callback:
            progress_callback("Loading image", 0.0)

        image = Image.open(image_path).convert("RGBA")

        if progress_callback:
            progress_callback("Preprocessing image", 0.05)

        # If image has SAM3 segmentation mask, crop the original image to the
        # object bounding box and let TRELLIS's internal rembg handle it.
        # TRELLIS was trained with rembg-style alpha — SAM3 binary masks produce
        # wrong depth estimation (elongated shapes).
        alpha = np.array(image)[:, :, 3]
        has_sam_alpha = not np.all(alpha == 255)
        if has_sam_alpha:
            original_path = engine_params.get("original_image_path")
            if original_path and os.path.exists(original_path):
                original = Image.open(original_path).convert("RGB")
                if original.size != image.size:
                    original = original.resize(image.size, Image.Resampling.LANCZOS)

                mask = alpha > 0
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                h, w = mask.shape
                pad_h = int((y_max - y_min) * 0.1)
                pad_w = int((x_max - x_min) * 0.1)
                y_min = max(0, y_min - pad_h)
                y_max = min(h, y_max + pad_h)
                x_min = max(0, x_min - pad_w)
                x_max = min(w, x_max + pad_w)

                image = original.crop((x_min, y_min, x_max, y_max))
                logger.info(f"Cropped original to SAM3 bbox + padding, letting TRELLIS rembg handle it")

        if progress_callback:
            progress_callback("Sampling sparse structure", 0.10)

        outputs = self.image_pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_steps,
                "cfg_strength": ss_guidance,
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": slat_guidance,
            },
        )

        if progress_callback:
            progress_callback("Generation complete", 0.70)

        return {
            "gaussian": outputs["gaussian"][0],
            "mesh": outputs["mesh"][0],
        }

    def generate_from_images(
        self,
        image_paths: list[str],
        seed: int,
        mode: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        if not self.image_pipeline:
            raise RuntimeError("Image pipeline not loaded")

        ss_steps = engine_params.get("ss_steps", 12)
        ss_guidance = engine_params.get("ss_guidance", 7.5)
        slat_steps = engine_params.get("slat_steps", 12)
        slat_guidance = engine_params.get("slat_guidance", 3.0)

        self._activate("image")

        if progress_callback:
            progress_callback("Loading images", 0.0)

        images = [Image.open(p) for p in image_paths]

        if progress_callback:
            progress_callback("Preprocessing images", 0.05)

        if progress_callback:
            progress_callback("Sampling sparse structure", 0.10)

        outputs = self.image_pipeline.run_multi_image(
            images,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=True,
            mode=mode,
            sparse_structure_sampler_params={
                "steps": ss_steps,
                "cfg_strength": ss_guidance,
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": slat_guidance,
            },
        )

        if progress_callback:
            progress_callback("Generation complete", 0.70)

        return {
            "gaussian": outputs["gaussian"][0],
            "mesh": outputs["mesh"][0],
        }

    def generate_from_text(
        self,
        prompt: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        if not self.text_pipeline:
            raise RuntimeError("Text pipeline not loaded. Download TRELLIS-text-large weights first.")

        ss_steps = engine_params.get("ss_steps", 12)
        ss_guidance = engine_params.get("ss_guidance", 7.5)
        slat_steps = engine_params.get("slat_steps", 12)
        slat_guidance = engine_params.get("slat_guidance", 3.0)

        self._activate("text")

        if progress_callback:
            progress_callback("Encoding text prompt", 0.05)

        if progress_callback:
            progress_callback("Sampling sparse structure", 0.10)

        outputs = self.text_pipeline.run(
            prompt,
            seed=seed,
            formats=["gaussian", "mesh"],
            sparse_structure_sampler_params={
                "steps": ss_steps,
                "cfg_strength": ss_guidance,
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": slat_guidance,
            },
        )

        if progress_callback:
            progress_callback("Generation complete", 0.70)

        return {
            "gaussian": outputs["gaussian"][0],
            "mesh": outputs["mesh"][0],
        }

    # -- Text-Guided Editing (Variant) -----------------------

    def edit_with_text(
        self,
        prompt: str,
        seed: int,
        slat_steps: int,
        slat_guidance: float,
        base_task_id: Optional[str] = None,
        mesh_file_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict:
        """Edit an existing 3D model using a text prompt.

        Preserves the base structure (voxel grid from the mesh) and
        re-samples only Stage 2 (structured latent) with the new text.

        Provide either base_task_id (uses a previous generation's mesh)
        or mesh_file_path (an uploaded GLB/OBJ/PLY file).
        """
        if not self.text_pipeline:
            raise RuntimeError("Text pipeline not loaded. Download TRELLIS-text-large weights first.")

        self._activate("text")

        _add_trellis_to_path(self.engine_dir)
        import open3d as o3d

        if progress_callback:
            progress_callback("Loading base mesh", 0.05)

        # Load the base mesh as an Open3D TriangleMesh
        if mesh_file_path:
            mesh_o3d = o3d.io.read_triangle_mesh(mesh_file_path)
        elif base_task_id:
            from ..config import GALLERY_DIR
            # Try to load the GLB from the previous generation
            glb_path = os.path.join(GALLERY_DIR, base_task_id, "model.glb")
            if not os.path.isfile(glb_path):
                raise FileNotFoundError(f"No mesh found for task {base_task_id}")
            mesh_o3d = o3d.io.read_triangle_mesh(glb_path)
        else:
            raise ValueError("Provide either base_task_id or mesh_file_path")

        if progress_callback:
            progress_callback("Voxelizing mesh", 0.1)

        outputs = self.text_pipeline.run_variant(
            mesh_o3d,
            prompt,
            seed=seed,
            formats=["gaussian", "mesh"],
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": slat_guidance,
            },
        )

        if progress_callback:
            progress_callback("Editing complete", 0.7)

        return {
            "gaussian": outputs["gaussian"][0],
            "mesh": outputs["mesh"][0],
        }

    # -- Export -----------------------------------------------

    def export_mesh(
        self,
        generation_data: dict,
        output_dir: str,
        formats: list[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict[str, Path]:
        simplify = engine_params.get("simplify", 0.95)
        texture_size = engine_params.get("texture_size", 1024)
        fill_holes = engine_params.get("fill_holes", True)
        fill_holes_max_size = engine_params.get("fill_holes_max_size", 0.04)

        _add_trellis_to_path(self.engine_dir)
        from trellis.utils import postprocessing_utils

        gaussian = generation_data["gaussian"]
        mesh = generation_data["mesh"]
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # GLB and OBJ both need the textured mesh, so build it once
        needs_textured = any(f in formats for f in ["glb", "obj"])
        glb_mesh = None

        if needs_textured:
            if progress_callback:
                progress_callback("Baking textures", 0.75)

            glb_mesh = postprocessing_utils.to_glb(
                gaussian,
                mesh,
                simplify=simplify,
                fill_holes=fill_holes,
                fill_holes_max_size=fill_holes_max_size,
                texture_size=texture_size,
            )

        if "glb" in formats:
            if progress_callback:
                progress_callback("Exporting GLB", 0.85)
            glb_path = os.path.join(output_dir, "model.glb")
            glb_mesh.export(glb_path)
            results["glb"] = Path(glb_path)

        if "obj" in formats:
            if progress_callback:
                progress_callback("Exporting OBJ", 0.88)
            obj_dir = os.path.join(output_dir, "obj")
            os.makedirs(obj_dir, exist_ok=True)
            obj_path = os.path.join(obj_dir, "model.obj")
            glb_mesh.export(obj_path)
            # Zip the OBJ + MTL + texture files
            zip_path = os.path.join(output_dir, "model_obj.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in Path(obj_dir).iterdir():
                    zf.write(f, f.name)
            results["obj"] = Path(zip_path)

        if "stl" in formats:
            if progress_callback:
                progress_callback("Exporting STL", 0.90)
            # STL = geometry only, no textures needed
            stl_mesh = postprocessing_utils.to_glb(
                gaussian,
                mesh,
                simplify=simplify,
                fill_holes=fill_holes,
                fill_holes_max_size=fill_holes_max_size,
                texture_size=512,  # minimal, won't be used
            ) if glb_mesh is None else glb_mesh
            stl_path = os.path.join(output_dir, "model.stl")
            stl_mesh.export(stl_path)
            results["stl"] = Path(stl_path)

        if "ply" in formats:
            if progress_callback:
                progress_callback("Exporting PLY (Gaussian)", 0.92)
            ply_path = os.path.join(output_dir, "model.ply")
            gaussian.save_ply(ply_path)
            results["ply"] = Path(ply_path)

        if progress_callback:
            progress_callback("Export complete", 1.0)

        return results

    # -- Preview ---------------------------------------------

    def render_preview(
        self,
        generation_data: dict,
        output_path: str,
        resolution: int = 512,
        num_frames: int = 120,
    ) -> str:
        _add_trellis_to_path(self.engine_dir)
        from trellis.utils import render_utils

        gaussian = generation_data["gaussian"]
        video_dict = render_utils.render_video(gaussian, resolution=resolution, num_frames=num_frames)
        frames = video_dict["color"]
        imageio.mimsave(output_path, frames, fps=15)
        return output_path
