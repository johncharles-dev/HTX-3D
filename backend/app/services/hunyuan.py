"""Hunyuan3D-2.1 engine wrapper — bridges Hunyuan pipelines into the BaseEngine interface."""

import gc
import os
import shutil
import sys
import time
import logging
import torch
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


class HunyuanEngine(BaseEngine):
    """Wraps Hunyuan3D-2.1 shape + texture pipelines.

    Shape (~7GB) and texture (~26GB) cannot coexist in VRAM,
    so we always swap: shape generates first, offloads to CPU,
    then texture pipeline loads for PBR material generation.
    """

    name = "hunyuan"
    loaded = False

    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir
        self.shape_pipeline = None
        self.rembg = None
        self._paint_conf = None
        self.paint_pipeline = None
        self._device = "cuda"
        self._vm = None

    # -- Lifecycle -------------------------------------------

    def _add_to_path(self):
        """Add Hunyuan engine subdirectories to sys.path."""
        shape_dir = os.path.join(self.engine_dir, "hy3dshape")
        paint_dir = os.path.join(self.engine_dir, "hy3dpaint")
        for d in [self.engine_dir, shape_dir, paint_dir]:
            if d not in sys.path:
                sys.path.insert(0, d)

    def load(self, weights_dir: str, device: str = "cuda") -> None:
        self._device = device
        self._add_to_path()

        # Apply torchvision compatibility fix
        try:
            from torchvision_fix import apply_fix
            apply_fix()
        except Exception as e:
            logger.warning(f"torchvision fix not applied: {e}")

        # Initialize Hunyuan VRAMManager
        # Use medium tier in multi-engine context to avoid OOM during texture generation
        # (shape offload leaves ~30GB free, but texture pipeline needs headroom for VAE decode)
        from hunyuan_vram_manager import init_vram_manager
        self._vm = init_vram_manager(precision="auto", vram_tier="medium")
        logger.info(f"Hunyuan VRAMManager: {self._vm}")

        # Load background remover
        from hy3dshape.rembg import BackgroundRemover
        self.rembg = BackgroundRemover()

        # Load shape generation pipeline
        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
        model_path = "tencent/Hunyuan3D-2.1"
        hunyuan_weights = os.path.join(weights_dir, "Hunyuan3D-2.1")
        if os.path.isdir(hunyuan_weights):
            logger.info(f"Loading Hunyuan3D shape model from local: {hunyuan_weights}")
            self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(hunyuan_weights)
        else:
            logger.info("Downloading Hunyuan3D shape model from HuggingFace (first run)...")
            self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

        # Prepare texture config (deferred loading to save VRAM)
        max_num_view, resolution = self._vm.texture_config
        logger.info(f"Texture config: {max_num_view} views @ {resolution}px (tier={self._vm.tier})")

        from textureGenPipeline import Hunyuan3DPaintConfig
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        # Set paths relative to engine directory
        conf.realesrgan_ckpt_path = os.path.join(self.engine_dir, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")
        conf.multiview_cfg_path = os.path.join(self.engine_dir, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
        conf.custom_pipeline = os.path.join(self.engine_dir, "hy3dpaint", "hunyuanpaintpbr")
        self._paint_conf = conf

        self.loaded = True
        logger.info("Hunyuan3D engine loaded successfully (texture pipeline deferred)")

    def unload(self) -> None:
        self.shape_pipeline = None
        self.paint_pipeline = None
        self.rembg = None
        self._paint_conf = None
        self._vm = None
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Hunyuan3D engine unloaded")

    # -- Model Swapping --------------------------------------

    def _offload_shape_pipeline(self):
        """Move shape pipeline to CPU to free VRAM for texture."""
        logger.info("Swapping: offloading shape pipeline to CPU...")
        torch.cuda.synchronize()
        # Move all nn.Module components to CPU
        for attr_name in dir(self.shape_pipeline):
            attr = getattr(self.shape_pipeline, attr_name, None)
            if isinstance(attr, torch.nn.Module):
                attr.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        vram_free = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info(f"Swapping: shape offloaded, {vram_free:.1f}GB VRAM free")

    def _load_texture_pipeline(self):
        """Load texture pipeline to GPU."""
        if self.paint_pipeline is None:
            logger.info("Swapping: loading texture pipeline to GPU...")
            self._add_to_path()
            from textureGenPipeline import Hunyuan3DPaintPipeline
            self.paint_pipeline = Hunyuan3DPaintPipeline(self._paint_conf)

    def _restore_shape_pipeline(self):
        """Free texture pipeline and restore shape to GPU."""
        logger.info("Swapping: restoring shape pipeline to GPU...")
        torch.cuda.synchronize()
        if self.paint_pipeline is not None:
            del self.paint_pipeline
            self.paint_pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        self.shape_pipeline.model.to(self._device)
        self.shape_pipeline.vae.to(self._device)
        self.shape_pipeline.conditioner.to(self._device)

    # -- Generation ------------------------------------------

    def generate_from_image(
        self,
        image_path: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        if not self.shape_pipeline:
            raise RuntimeError("Hunyuan shape pipeline not loaded")

        num_inference_steps = engine_params.get("num_inference_steps", 30)
        guidance_scale = engine_params.get("guidance_scale", 5.5)
        octree_resolution = engine_params.get("octree_resolution", 256)
        texture = engine_params.get("texture", True)

        if progress_callback:
            progress_callback("Loading image", 0.0)

        import numpy as np
        image = Image.open(image_path).convert("RGBA")

        if progress_callback:
            progress_callback("Removing background", 0.05)

        # Check if the image has a SAM3 segmentation mask.
        # Hunyuan was trained with rembg-style alpha (soft edges, gradual falloff).
        # SAM3 binary masks produce wrong depth estimation. Instead, use SAM3 mask
        # to crop the original image to the target object, then let rembg process
        # the crop — giving Hunyuan exactly the input distribution it was trained on.
        alpha = np.array(image)[:, :, 3]
        has_alpha = not np.all(alpha == 255)
        if has_alpha:
            original_path = engine_params.get("original_image_path")
            if original_path and os.path.exists(original_path):
                original = Image.open(original_path).convert("RGB")
                if original.size != image.size:
                    original = original.resize(image.size, Image.Resampling.LANCZOS)

                # Find SAM3 mask bounding box and crop original image with padding
                mask = alpha > 0
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                # Add 10% padding around the object for scene context
                h, w = mask.shape
                pad_h = int((y_max - y_min) * 0.1)
                pad_w = int((x_max - x_min) * 0.1)
                y_min = max(0, y_min - pad_h)
                y_max = min(h, y_max + pad_h)
                x_min = max(0, x_min - pad_w)
                x_max = min(w, x_max + pad_w)

                cropped = original.crop((x_min, y_min, x_max, y_max))
                logger.info(f"Cropped original to SAM3 bbox ({x_min},{y_min},{x_max},{y_max}) + 10% padding, running rembg")
                image = self.rembg(cropped)
            else:
                logger.info("Image has alpha but no original — using as-is, skipping rembg")
        else:
            image = self.rembg(image)

        if progress_callback:
            progress_callback("Generating 3D shape", 0.10)

        start_time = time.time()
        mesh = self.shape_pipeline(
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )[0]
        logger.info(f"Shape generation took {time.time() - start_time:.1f}s")

        if progress_callback:
            progress_callback("Shape generation complete", 0.50)

        return {
            "mesh": mesh,
            "image": image,
            "image_path": image_path,
            "texture_enabled": texture,
            "roughness_offset": engine_params.get("roughness_offset", 0.0),
            "metallic_scale": engine_params.get("metallic_scale", 1.0),
        }

    def generate_from_images(
        self,
        image_paths: list[str],
        seed: int,
        mode: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        # Hunyuan doesn't have native multi-view input — use first image
        return self.generate_from_image(
            image_paths[0],
            seed,
            progress_callback=progress_callback,
            **engine_params,
        )

    def generate_from_text(
        self,
        prompt: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        raise NotImplementedError("Hunyuan3D does not support text-to-3D generation")

    # -- Export -----------------------------------------------

    def export_mesh(
        self,
        generation_data: dict,
        output_dir: str,
        formats: list[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict[str, Path]:
        self._add_to_path()

        mesh = generation_data["mesh"]
        image = generation_data.get("image")
        texture_enabled = generation_data.get("texture_enabled", True)
        target_face_count = engine_params.get("target_face_count", 0)
        remove_floaters = engine_params.get("remove_floaters", True)
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # Export initial untextured mesh
        initial_path = os.path.join(output_dir, "model_initial.glb")
        mesh.export(initial_path)

        if texture_enabled:
            try:
                if progress_callback:
                    progress_callback("Swapping to texture pipeline", 0.55)

                # Model swap: shape -> texture
                self._offload_shape_pipeline()
                self._load_texture_pipeline()

                if progress_callback:
                    progress_callback("Generating PBR textures", 0.60)

                # Run texture generation
                output_obj_path = os.path.join(output_dir, "textured.obj")
                start_time = time.time()
                self.paint_pipeline(
                    mesh_path=initial_path,
                    image_path=image,
                    output_mesh_path=output_obj_path,
                    save_glb=False,
                )
                logger.info(f"Texture generation took {time.time() - start_time:.1f}s")

                # Apply roughness/metallic adjustments if specified
                roughness_offset = generation_data.get("roughness_offset", 0.0) or 0.0
                metallic_scale = generation_data.get("metallic_scale", 1.0) or 1.0
                if roughness_offset != 0.0 or metallic_scale != 1.0:
                    if progress_callback:
                        progress_callback("Adjusting materials", 0.83)
                    self._apply_mr_adjustments(output_obj_path, roughness_offset, metallic_scale)

                if progress_callback:
                    progress_callback("Converting to GLB", 0.85)

                # Convert OBJ with PBR materials to GLB
                from hy3dpaint.convert_utils import create_glb_with_pbr_materials
                glb_path = os.path.join(output_dir, "model.glb")
                textures = {
                    "albedo": output_obj_path.replace(".obj", ".jpg"),
                    "metallic": output_obj_path.replace(".obj", "_metallic.jpg"),
                    "roughness": output_obj_path.replace(".obj", "_roughness.jpg"),
                }
                create_glb_with_pbr_materials(output_obj_path, textures, glb_path)

                if "glb" in formats:
                    results["glb"] = Path(glb_path)

            except Exception as e:
                logger.error(f"Texture generation failed: {e}")
                logger.warning("Falling back to untextured mesh")
                # Fall back to untextured mesh
                if "glb" in formats:
                    fallback_path = os.path.join(output_dir, "model.glb")
                    shutil.copy2(initial_path, fallback_path)
                    results["glb"] = Path(fallback_path)
            finally:
                # Always restore shape pipeline for next request
                try:
                    self._restore_shape_pipeline()
                except Exception as e:
                    logger.error(f"Failed to restore shape pipeline: {e}")
        else:
            # No texture — just use the initial mesh
            if "glb" in formats:
                glb_path = os.path.join(output_dir, "model.glb")
                shutil.copy2(initial_path, glb_path)
                results["glb"] = Path(glb_path)

        # Post-processing: remove disconnected floating components
        if remove_floaters:
            if progress_callback:
                progress_callback("Removing floating parts", 0.87)
            self._remove_floaters(output_dir)

        # Post-processing: decimate to target face count if requested
        if target_face_count and target_face_count > 0:
            if progress_callback:
                progress_callback("Decimating mesh", 0.88)
            self._decimate_mesh(output_dir, target_face_count)

        # STL export (geometry only, from initial mesh)
        if "stl" in formats:
            if progress_callback:
                progress_callback("Exporting STL", 0.90)
            import trimesh
            stl_mesh = trimesh.load(initial_path)
            stl_path = os.path.join(output_dir, "model.stl")
            stl_mesh.export(stl_path)
            results["stl"] = Path(stl_path)

        # OBJ export (zipped with textures if available)
        if "obj" in formats:
            if progress_callback:
                progress_callback("Exporting OBJ", 0.92)
            obj_source = os.path.join(output_dir, "textured.obj")
            if os.path.exists(obj_source):
                import zipfile
                zip_path = os.path.join(output_dir, "model_obj.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in Path(output_dir).glob("textured*"):
                        zf.write(f, f.name)
                results["obj"] = Path(zip_path)

        if progress_callback:
            progress_callback("Export complete", 1.0)

        return results

    def retexture_mesh(
        self,
        generation_data: dict,
        output_dir: str,
        formats: list[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict[str, Path]:
        """Re-run only the paint/texture pipeline on an existing shape mesh."""
        self._add_to_path()

        mesh_path = generation_data["mesh_path"]
        input_image_path = generation_data["input_image_path"]
        roughness_offset = generation_data.get("roughness_offset", 0.0) or 0.0
        metallic_scale = generation_data.get("metallic_scale", 1.0) or 1.0
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # Copy shape mesh to new output dir
        initial_path = os.path.join(output_dir, "model_initial.glb")
        shutil.copy2(mesh_path, initial_path)

        # Copy input image for future re-textures of this result
        input_save = os.path.join(output_dir, "input_image.png")
        shutil.copy2(input_image_path, input_save)

        try:
            if progress_callback:
                progress_callback("Loading texture pipeline", 0.10)

            self._offload_shape_pipeline()
            self._load_texture_pipeline()

            if progress_callback:
                progress_callback("Generating PBR textures", 0.20)

            from PIL import Image
            image = Image.open(input_image_path).convert("RGB")
            output_obj_path = os.path.join(output_dir, "textured.obj")
            start_time = time.time()
            self.paint_pipeline(
                mesh_path=initial_path,
                image_path=image,
                output_mesh_path=output_obj_path,
                save_glb=False,
            )
            logger.info(f"Re-texture took {time.time() - start_time:.1f}s")

            # Apply material adjustments
            if roughness_offset != 0.0 or metallic_scale != 1.0:
                if progress_callback:
                    progress_callback("Adjusting materials", 0.80)
                self._apply_mr_adjustments(output_obj_path, roughness_offset, metallic_scale)

            if progress_callback:
                progress_callback("Converting to GLB", 0.85)

            from hy3dpaint.convert_utils import create_glb_with_pbr_materials
            glb_path = os.path.join(output_dir, "model.glb")
            textures = {
                "albedo": output_obj_path.replace(".obj", ".jpg"),
                "metallic": output_obj_path.replace(".obj", "_metallic.jpg"),
                "roughness": output_obj_path.replace(".obj", "_roughness.jpg"),
            }
            create_glb_with_pbr_materials(output_obj_path, textures, glb_path)

            if "glb" in formats:
                results["glb"] = Path(glb_path)

        except Exception as e:
            logger.error(f"Re-texture failed: {e}")
            if "glb" in formats:
                fallback_path = os.path.join(output_dir, "model.glb")
                shutil.copy2(initial_path, fallback_path)
                results["glb"] = Path(fallback_path)
        finally:
            try:
                self._restore_shape_pipeline()
            except Exception as e:
                logger.error(f"Failed to restore shape pipeline: {e}")

        return results

    def quick_adjust_materials(
        self,
        base_task_dir: str,
        output_dir: str,
        roughness_offset: float,
        metallic_scale: float,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[str, Path]:
        """Adjust material textures without re-running diffusion. Instant, no GPU."""
        self._add_to_path()
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        if progress_callback:
            progress_callback("Copying textures", 0.1)

        # Copy all needed files from the base task
        for name in ("model_initial.glb", "input_image.png", "textured.obj",
                      "textured.mtl", "textured.jpg",
                      "textured_metallic.jpg", "textured_roughness.jpg"):
            src = os.path.join(base_task_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, name))

        # Also copy any extra texture files (normals, etc.)
        import glob
        for src in glob.glob(os.path.join(base_task_dir, "textured*.jpg")):
            dst = os.path.join(output_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        output_obj_path = os.path.join(output_dir, "textured.obj")
        if not os.path.exists(output_obj_path):
            raise FileNotFoundError(f"Textured OBJ not found in {base_task_dir}")

        if progress_callback:
            progress_callback("Adjusting materials", 0.4)

        # Apply adjustments to the copied texture files
        if roughness_offset != 0.0 or metallic_scale != 1.0:
            self._apply_mr_adjustments(output_obj_path, roughness_offset, metallic_scale)

        if progress_callback:
            progress_callback("Converting to GLB", 0.7)

        # Re-convert to GLB with adjusted textures
        from hy3dpaint.convert_utils import create_glb_with_pbr_materials
        glb_path = os.path.join(output_dir, "model.glb")
        textures = {
            "albedo": output_obj_path.replace(".obj", ".jpg"),
            "metallic": output_obj_path.replace(".obj", "_metallic.jpg"),
            "roughness": output_obj_path.replace(".obj", "_roughness.jpg"),
        }
        create_glb_with_pbr_materials(output_obj_path, textures, glb_path)

        if "glb" in ["glb"]:
            results["glb"] = Path(glb_path)

        if progress_callback:
            progress_callback("Done", 1.0)

        logger.info(f"Quick adjust complete: roughness={roughness_offset:+.2f}, metallic={metallic_scale:.2f}x")
        return results

    @staticmethod
    def _apply_mr_adjustments(output_obj_path: str, roughness_offset: float, metallic_scale: float):
        """Adjust metallic/roughness texture JPGs after paint pipeline."""
        import numpy as np
        from PIL import Image

        metallic_path = output_obj_path.replace(".obj", "_metallic.jpg")
        roughness_path = output_obj_path.replace(".obj", "_roughness.jpg")

        if os.path.exists(metallic_path) and metallic_scale != 1.0:
            img = np.array(Image.open(metallic_path)).astype(np.float32)
            img = np.clip(img * metallic_scale, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(metallic_path, quality=95)
            logger.info(f"Metallic texture adjusted (scale={metallic_scale:.2f})")

        if os.path.exists(roughness_path) and roughness_offset != 0.0:
            img = np.array(Image.open(roughness_path)).astype(np.float32)
            img = np.clip(img + roughness_offset * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(roughness_path, quality=95)
            logger.info(f"Roughness texture adjusted (offset={roughness_offset:+.2f})")

    @staticmethod
    def _remove_floaters(output_dir: str, min_ratio: float = 0.05):
        """Remove small disconnected mesh components, keep largest.

        Args:
            output_dir: Directory containing exported meshes.
            min_ratio: Components with fewer faces than this ratio of
                       the largest component are removed.
        """
        import trimesh
        for name in ("model.glb", "model_initial.glb"):
            path = os.path.join(output_dir, name)
            if not os.path.exists(path):
                continue
            scene = trimesh.load(path)
            if isinstance(scene, trimesh.Scene):
                mesh = trimesh.util.concatenate(scene.dump())
            else:
                mesh = scene
            components = mesh.split()
            if len(components) <= 1:
                continue
            # Sort by face count descending
            components.sort(key=lambda c: len(c.faces), reverse=True)
            largest_faces = len(components[0].faces)
            # Keep components that are at least min_ratio of the largest
            keep = [c for c in components if len(c.faces) >= largest_faces * min_ratio]
            removed = len(components) - len(keep)
            if removed > 0:
                cleaned = trimesh.util.concatenate(keep)
                logger.info(
                    f"Floater removal ({name}): removed {removed} component(s), "
                    f"kept {len(keep)} ({len(cleaned.faces)} faces)"
                )
                cleaned.export(path)

    @staticmethod
    def _decimate_mesh(output_dir: str, target_count: int):
        """Decimate exported meshes to a target face count using pymeshlab."""
        import pymeshlab
        for name in ("model.glb", "model_initial.glb"):
            path = os.path.join(output_dir, name)
            if not os.path.exists(path):
                continue
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(path, load_in_a_single_layer=True)
            if ms.current_mesh().face_number() > target_count:
                logger.info(f"Decimating {name}: {ms.current_mesh().face_number()} -> {target_count} faces")
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_count)
                ms.save_current_mesh(path)

    # -- Preview ---------------------------------------------

    def render_preview(
        self,
        generation_data: dict,
        output_path: str,
        resolution: int = 512,
        num_frames: int = 120,
    ) -> str:
        # Hunyuan doesn't produce Gaussian splats — no preview video
        raise NotImplementedError("Hunyuan3D does not support Gaussian preview rendering")
