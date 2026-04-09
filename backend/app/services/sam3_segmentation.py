"""SAM3 segmentation service for interactive object masking.

Manages GPU lifecycle, session state, and segmentation modes (text/box/point).
SAM3 masks are used as input for 3D generation engines (TRELLIS, Hunyuan, SAM 3D Objects).

All inference calls must be wrapped in torch.autocast("cuda", dtype=torch.bfloat16)
because SAM3 model weights are BFloat16 but some layers expect Float32 inputs.

Point prompts use the SAM1-style interactive mask decoder (inst_interactive_predictor)
for proper foreground/background click-to-refine behavior. The backbone features are
shared from the processor's set_image() — no separate backbone needed.
"""

import asyncio
import gc
import logging
import os
import uuid
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..config import SAM3_DIR, SAM3_BPE_PATH, TEMP_DIR

logger = logging.getLogger(__name__)

# GPU lock to prevent concurrent SAM3 + generation engine usage
gpu_lock = asyncio.Lock()

# Enable tf32 for faster computation on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class SAM3Service:
    """Manages SAM3 model lifecycle and segmentation sessions."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._predictor = None  # SAM1-style interactive predictor
        self._loaded = False
        self._sessions: dict[str, dict] = {}

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, device: str = "cuda"):
        """Load SAM3 model onto GPU with interactive predictor enabled."""
        if self._loaded:
            return

        import sys
        if SAM3_DIR not in sys.path:
            sys.path.insert(0, SAM3_DIR)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info("Loading SAM3 model with interactive predictor...")
        start = time.time()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._model = build_sam3_image_model(
                bpe_path=SAM3_BPE_PATH,
                device=device,
                eval_mode=True,
                load_from_HF=True,
                enable_inst_interactivity=True,
            )
        self._processor = Sam3Processor(
            model=self._model,
            resolution=1008,
            device=device,
            confidence_threshold=0.5,
        )
        self._predictor = self._model.inst_interactive_predictor
        self._loaded = True
        logger.info(f"SAM3 loaded in {time.time() - start:.1f}s (interactive predictor: {self._predictor is not None})")

    def unload(self):
        """Free SAM3 from GPU memory."""
        if not self._loaded:
            return

        logger.info("Unloading SAM3...")
        self._sessions.clear()
        del self._processor
        del self._predictor
        del self._model
        self._processor = None
        self._predictor = None
        self._model = None
        self._loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("SAM3 unloaded")

    def _setup_predictor_from_state(self, state: dict, session: dict):
        """Inject shared backbone features into the interactive predictor.

        The processor's set_image() already computed backbone features including
        sam2_backbone_out with conv_s0/conv_s1 applied. We reuse those features
        instead of calling predictor.set_image() which would need its own backbone.
        """
        if self._predictor is None:
            return

        sam2_out = state.get("backbone_out", {}).get("sam2_backbone_out")
        if sam2_out is None:
            logger.warning("No sam2_backbone_out in state — predictor not initialized")
            return

        # Reset predictor state
        self._predictor.reset_predictor()
        self._predictor._orig_hw = [(session["height"], session["width"])]

        # Extract features via _prepare_backbone_features (same as predictor.set_image does)
        tracker = self._predictor.model
        _, vision_feats, _, _ = tracker._prepare_backbone_features(sam2_out)

        # Add no_mem_embed (same as predictor.set_image line 109)
        vision_feats[-1] = vision_feats[-1] + tracker.no_mem_embed

        # Reshape to feature dict (same as predictor.set_image lines 111-115)
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1],
                self._predictor._bb_feat_sizes[::-1],
            )
        ][::-1]

        self._predictor._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self._predictor._is_image_set = True

    def start_session(self, image_path: str) -> dict:
        """Start a segmentation session with an image.

        Returns dict with session_id, width, height.
        """
        if not self._loaded:
            self.load()

        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        session_id = uuid.uuid4().hex[:12]

        # Set image in processor (caches backbone features)
        # Pass PIL Image (not numpy array) so set_image gets correct width/height
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self._processor.set_image(img)

        self._sessions[session_id] = {
            "image_path": image_path,
            "image": img,
            "image_array": img_array,
            "state": state,
            "masks": None,
            "scores": None,
            "boxes": None,
            "width": img.width,
            "height": img.height,
        }

        # Setup interactive predictor from shared backbone features
        self._setup_predictor_from_state(state, self._sessions[session_id])

        logger.info(f"Segmentation session {session_id} started ({img.width}x{img.height})")
        return {
            "session_id": session_id,
            "width": img.width,
            "height": img.height,
        }

    def _get_session(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")
        return session

    def _resize_masks_to_image(self, masks, session: dict):
        """Resize masks to match original image dimensions if needed."""
        if masks is None or masks.shape[0] == 0:
            return masks
        h_img, w_img = session["image_array"].shape[:2]
        _, _, h_mask, w_mask = masks.shape
        if h_mask != h_img or w_mask != w_img:
            logger.info(f"Resizing masks from {h_mask}x{w_mask} to {h_img}x{w_img}")
            masks = torch.nn.functional.interpolate(
                masks.float(), size=(h_img, w_img), mode="nearest"
            ).bool()
        return masks

    def segment_text(self, session_id: str, prompt: str) -> dict:
        """Segment using a text prompt.

        After text segmentation, the best mask's logits are downsampled and stored
        as '_text_mask_logits' in the session. When segment_points() is called next,
        these logits serve as the initial mask_input prior — allowing point clicks
        to refine the text-selected region instead of starting from scratch.
        """
        session = self._get_session(session_id)
        state = session["state"]

        # Reset prompts and run text segmentation
        self._processor.reset_all_prompts(state)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self._processor.set_text_prompt(prompt=prompt, state=state)

        masks = self._resize_masks_to_image(state.get("masks"), session)
        scores = state.get("scores")  # [N] float tensor
        boxes = state.get("boxes")  # [N, 4] pixel coords

        session["masks"] = masks
        session["scores"] = scores
        session["boxes"] = boxes
        session["state"] = state

        # Store downsampled text mask logits for text→point refinement bridge.
        # When the user clicks points after a text search, these logits are passed
        # as mask_input to the interactive predictor so points refine the text result.
        session["_text_mask_logits"] = None
        session["_low_res_logits"] = None  # Clear any previous point logits
        masks_logits = state.get("masks_logits")
        if masks_logits is not None and len(masks_logits) > 0 and scores is not None:
            best_idx = int(scores.argmax())
            mask_input_size = self._predictor.model.sam_prompt_encoder.mask_input_size
            low_res = torch.nn.functional.interpolate(
                masks_logits[best_idx:best_idx+1].float(),
                size=mask_input_size,
                mode="bilinear",
                align_corners=False,
            )
            session["_text_mask_logits"] = low_res[0].cpu().numpy()  # (1, H, W)
            logger.info(f"Session {session_id}: stored text mask logits {session['_text_mask_logits'].shape} for point refinement")

        return self._build_mask_response(session_id, session)

    def segment_box(self, session_id: str, box: list[float], label: bool = True) -> dict:
        """Segment using a bounding box prompt.

        box: [center_x, center_y, width, height] normalized 0-1
        Boxes accumulate in the backend state via add_geometric_prompt.
        """
        session = self._get_session(session_id)
        state = session["state"]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self._processor.add_geometric_prompt(
                box=box,
                label=label,
                state=state,
            )

        masks = self._resize_masks_to_image(state.get("masks"), session)
        scores = state.get("scores")
        boxes = state.get("boxes")

        session["masks"] = masks
        session["scores"] = scores
        session["boxes"] = boxes
        session["state"] = state

        return self._build_mask_response(session_id, session)

    def segment_points(self, session_id: str, points: list[list[float]], labels: list[int]) -> dict:
        """Segment using accumulated point prompts via SAM interactive predictor.

        points: list of [x, y] normalized 0-1
        labels: list of 1 (foreground/add) or 0 (background/remove)

        Aligned with the proven sam3 interactive_ui.py approach:
        - Always uses multimask_output=True (3 mask candidates), picks best by IoU
        - Each call sends ALL accumulated points fresh (no logit chaining between clicks)
        - If text mask logits exist from a prior segment_text() call, they are passed as
          mask_input prior — this enables the text→point refinement workflow where
          users first find an object by text, then refine edges with clicks
        - Uses normalize_coords=True so the model handles coordinate scaling internally
        """
        session = self._get_session(session_id)
        state = session["state"]

        if self._predictor is None:
            raise RuntimeError("Interactive predictor not available")

        # Re-setup predictor features from shared backbone state each call,
        # matching predict_inst() which sets up features fresh per call
        self._setup_predictor_from_state(state, session)

        w, h = session["width"], session["height"]

        # Convert normalized 0-1 coords from frontend to pixel coords,
        # matching the sam3 interactive_ui.py which passes raw pixel coords
        # with normalize_coords=True (model normalizes internally)
        point_coords = np.array([[pt[0] * w, pt[1] * h] for pt in points], dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        # Mask input priority: previous point logits > text logits > none.
        # Logit chaining feeds the previous best mask back as a hint, producing
        # tighter edges on subsequent clicks. Text logits serve as initial prior
        # when refining a text-selected region with points.
        mask_input = None
        prev_logits = session.get("_low_res_logits")
        text_logits = session.get("_text_mask_logits")
        if prev_logits is not None:
            mask_input = prev_logits
        elif text_logits is not None:
            mask_input = text_logits

        # First click without prior: multimask=True (3 candidates).
        # Subsequent clicks with logit prior: multimask=False (single refined mask).
        use_multimask = mask_input is None

        with torch.autocast("cuda", dtype=torch.bfloat16):
            masks_np, scores_np, low_res_logits = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=use_multimask,
                normalize_coords=True,
            )

        # Store best mask logits for next refinement (logit chaining)
        best_idx = int(np.argmax(scores_np))
        session["_low_res_logits"] = low_res_logits[best_idx:best_idx+1]

        # Store only the single best mask
        best_mask = masks_np[best_idx:best_idx+1]  # 1xHxW
        best_score = scores_np[best_idx:best_idx+1]  # 1

        masks = torch.from_numpy(best_mask).unsqueeze(1).bool()  # [1, 1, H, W]
        scores = torch.from_numpy(best_score)

        session["masks"] = masks
        session["scores"] = scores
        session["boxes"] = None

        return self._build_mask_response(session_id, session)

    def segment_point(self, session_id: str, x: float, y: float, label: bool = True) -> dict:
        """Segment using a single point (convenience wrapper)."""
        lbl = 1 if label else 0
        return self.segment_points(session_id, [[x, y]], [lbl])

    def reset_prompts(self, session_id: str) -> dict:
        """Reset all prompts for a session."""
        session = self._get_session(session_id)
        state = session["state"]
        self._processor.reset_all_prompts(state)
        # Re-set the image to get fresh state
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self._processor.set_image(session["image"])
        session["state"] = state
        session["masks"] = None
        session["scores"] = None
        session["boxes"] = None
        session["_low_res_logits"] = None
        session["_text_mask_logits"] = None
        # Re-setup interactive predictor
        self._setup_predictor_from_state(state, session)
        return {"session_id": session_id, "message": "Prompts reset"}

    def confirm_mask(self, session_id: str, mask_index: int = 0) -> str:
        """Apply selected mask to image and save as RGBA PNG.

        Returns path to the RGBA image.
        """
        session = self._get_session(session_id)
        masks = session.get("masks")
        if masks is None or masks.shape[0] == 0:
            raise ValueError("No masks available. Run segmentation first.")
        if mask_index >= masks.shape[0]:
            raise ValueError(f"Mask index {mask_index} out of range (have {masks.shape[0]})")

        # Extract single mask: [1, H, W] -> [H, W] numpy bool
        mask = masks[mask_index, 0].cpu().numpy()

        # Create RGBA image with soft alpha edges.
        # SAM3 produces hard binary masks (0 or 255), but TRELLIS and Hunyuan
        # work better with soft edges like rembg produces. Apply a small gaussian
        # blur to the mask edges to create a gradual alpha transition.
        from scipy.ndimage import gaussian_filter
        img_array = session["image_array"]
        alpha_float = mask.astype(np.float32)
        alpha_soft = gaussian_filter(alpha_float, sigma=2)
        # Keep the interior fully opaque — only soften the boundary
        alpha_soft = np.clip(alpha_soft, 0.0, 1.0)
        alpha_soft = np.where(mask, np.maximum(alpha_soft, 0.9), alpha_soft)
        alpha = (alpha_soft * 255).astype(np.uint8)
        rgba = np.concatenate([img_array, alpha[..., None]], axis=-1)

        # Save to temp directory
        output_dir = os.path.join(TEMP_DIR, "segmented", session_id)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "segmented.png")
        Image.fromarray(rgba).save(output_path)

        logger.info(f"Session {session_id}: saved segmented RGBA to {output_path}")
        return output_path

    def create_overlay(self, session_id: str) -> str:
        """Create a color-coded overlay image of all masks.

        Returns path to the overlay PNG.
        """
        session = self._get_session(session_id)
        masks = session.get("masks")
        if masks is None or masks.shape[0] == 0:
            raise ValueError("No masks available")

        img_array = session["image_array"].copy().astype(np.float32)
        scores = session.get("scores")

        # Color: best mask in solid blue, others in faint outline
        best_idx = 0
        if scores is not None and len(scores) > 0:
            best_idx = int(scores.argmax()) if hasattr(scores, 'argmax') else 0

        # Only show the best mask (matching sam3 interactive_ui.py behavior)
        mask_np = masks[best_idx, 0].cpu().numpy()
        color = np.array([60, 120, 255], dtype=np.float32)
        img_array[mask_np] = img_array[mask_np] * 0.45 + color * 0.55

        overlay = Image.fromarray(img_array.astype(np.uint8))
        output_dir = os.path.join(TEMP_DIR, "segmented", session_id)
        os.makedirs(output_dir, exist_ok=True)
        overlay_path = os.path.join(output_dir, "overlay.png")
        overlay.save(overlay_path)
        return overlay_path

    def _build_mask_response(self, session_id: str, session: dict) -> dict:
        """Build response dict from mask results."""
        masks = session.get("masks")
        scores = session.get("scores")
        boxes = session.get("boxes")

        if masks is None or masks.shape[0] == 0:
            overlay_path = ""
            mask_results = []
        else:
            overlay_path = self.create_overlay(session_id)
            mask_results = []
            for i in range(masks.shape[0]):
                mask_np = masks[i, 0].cpu().numpy()
                box = boxes[i].cpu().tolist() if boxes is not None else [0, 0, 0, 0]
                mask_results.append({
                    "index": i,
                    "score": float(scores[i]) if scores is not None else 0.0,
                    "bbox": box,
                    "area_pixels": int(mask_np.sum()),
                })

        return {
            "session_id": session_id,
            "masks": mask_results,
            "overlay_path": overlay_path,
        }

    def cleanup_session(self, session_id: str):
        """Remove a session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session {session_id} cleaned up")
