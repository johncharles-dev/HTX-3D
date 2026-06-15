"""
View-aligned metric scaling of generated GLBs using UniDepth + pytorch3d silhouette matching.

Pipeline:
    1. Object mask: alpha channel -> rembg -> fullframe (fallback ladder)
    2. UniDepth -> metric depth + camera intrinsics K + pre-unprojected 3D points
    3. Image metric bbox: 2D mask bbox in pixels + object distance D + perspective extent
    4. Render GLB silhouette with the same K and distance D, optionally rotated
    5. Match 2D bbox extents -> uniform scale (image_bbox_px / render_bbox_px)
    6. Bake scale into GLB (trimesh Scene.apply_scale, preserves materials)

Confidence tiers based on mask source and view-alignment IoU.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Mask extraction (3-tier fallback)
# ----------------------------------------------------------------------------

def get_object_mask(image_path: str) -> Tuple[np.ndarray, str]:
    """Return (HxW bool mask, source) where source in {'alpha','rembg','fullframe'}."""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.split()[-1])
        if alpha.std() > 5 and (alpha > 200).sum() > 100:
            return alpha > 128, "alpha"
    try:
        from rembg import remove
        rgba = img.convert("RGBA")
        out = remove(rgba)
        alpha = np.array(out)[..., 3]
        if (alpha > 200).sum() > 100:
            return alpha > 128, "rembg"
    except Exception as e:
        logger.warning("rembg fallback failed: %s", e)
    W, H = img.size
    return np.ones((H, W), dtype=bool), "fullframe"


# ----------------------------------------------------------------------------
# UniDepth singleton (lazy load + GPU residence)
# ----------------------------------------------------------------------------

class _UniDepth:
    _model = None
    _name = "lpiccinelli/unidepth-v2-vits14"

    @classmethod
    def get(cls):
        if cls._model is None:
            logger.info("loading UniDepth %s ...", cls._name)
            from unidepth.models import UniDepthV2
            cls._model = UniDepthV2.from_pretrained(cls._name).to("cuda").eval()
        return cls._model

    @classmethod
    def unload(cls):
        cls._model = None
        torch.cuda.empty_cache()


def estimate_depth_intrinsics(image_path: str) -> dict:
    """Run UniDepth. Returns dict with depth, K, points, confidence, image_size."""
    model = _UniDepth.get()
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).to("cuda")
    with torch.no_grad():
        pred = model.infer(rgb_t)
    return {
        "depth": pred["depth"].squeeze().cpu().numpy().astype(np.float32),         # (H, W) meters
        "K": pred["intrinsics"].squeeze().cpu().numpy().astype(np.float32),         # (3, 3)
        "points": pred["points"].squeeze().permute(1, 2, 0).cpu().numpy().astype(np.float32),  # (H, W, 3)
        "confidence": pred["confidence"].squeeze().cpu().numpy().astype(np.float32),  # (H, W) 0+
        "image_size": img.size,  # (W, H)
    }


# ----------------------------------------------------------------------------
# Mask refinement: depth-gated + confidence-gated
# ----------------------------------------------------------------------------

def refine_mask_by_depth(mask: np.ndarray, depth: np.ndarray,
                         confidence: Optional[np.ndarray] = None,
                         depth_sigma: float = 1.5,
                         min_confidence_pct: float = 25.0) -> tuple[np.ndarray, dict]:
    """Drop mask pixels whose depth is more than depth_sigma σ from the object-median,
    and (optionally) pixels in the bottom min_confidence_pct percentile of UniDepth's
    confidence map. Filters out background bleed-through from rembg (road, shadow, sky).

    Returns (refined_mask, debug_info).
    """
    if int(mask.sum()) < 50:
        return mask, {"refined": False, "reason": "input mask too small"}
    md = depth[mask]
    med = float(np.median(md))
    sd = float(np.std(md))
    # 1) depth gate
    depth_gate = (depth >= med - depth_sigma * sd) & (depth <= med + depth_sigma * sd)
    refined = mask & depth_gate
    # 2) confidence gate (drop the lowest-confidence pixels within the mask)
    conf_dropped = 0
    if confidence is not None and refined.sum() > 100:
        mc = confidence[refined]
        thresh = float(np.percentile(mc, min_confidence_pct))
        conf_gate = confidence >= thresh
        refined_after = refined & conf_gate
        if refined_after.sum() > 100:    # don't drop if it would empty the mask
            conf_dropped = int(refined.sum() - refined_after.sum())
            refined = refined_after
    # If we shrunk the mask too aggressively, fall back to original
    if refined.sum() < max(100, int(mask.sum()) * 0.3):
        return mask, {"refined": False, "reason": "refinement too aggressive — fallback to original",
                      "median_depth_m": med, "sigma_m": sd}
    return refined, {
        "refined": True,
        "median_depth_m": med,
        "sigma_m": sd,
        "kept_fraction": float(refined.sum() / max(mask.sum(), 1)),
        "depth_gate_dropped": int(mask.sum() - (mask & depth_gate).sum()),
        "confidence_gate_dropped": conf_dropped,
    }


def confidence_weighted_distance(mask: np.ndarray, depth: np.ndarray,
                                 confidence: Optional[np.ndarray]) -> float:
    """Robust object distance: median of depth in mask, weighted by UniDepth confidence
    when available. Falls back to plain median if no confidence map."""
    md = depth[mask]
    if confidence is None:
        return float(np.median(md))
    w = confidence[mask].astype(np.float64)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return float(np.median(md))
    # Weighted median
    order = np.argsort(md)
    md_sorted = md[order]
    w_sorted  = w[order]
    cum = np.cumsum(w_sorted)
    target = cum[-1] / 2.0
    idx = int(np.searchsorted(cum, target))
    idx = max(0, min(idx, len(md_sorted) - 1))
    return float(md_sorted[idx])


# ----------------------------------------------------------------------------
# Image-derived metric bbox
# ----------------------------------------------------------------------------

def compute_image_metric_bbox(mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
                              points: np.ndarray,
                              confidence: Optional[np.ndarray] = None) -> dict:
    """Compute object's pixel/metric extent from masked depth+K (uses pre-unprojected points for 3D).
    If confidence is provided, object distance D uses confidence-weighted median."""
    if int(mask.sum()) < 100:
        raise ValueError(f"object mask too small ({int(mask.sum())} px)")
    ys, xs = np.where(mask)
    u0, u1 = int(xs.min()), int(xs.max())
    v0, v1 = int(ys.min()), int(ys.max())
    w_px, h_px = u1 - u0, v1 - v0
    D = confidence_weighted_distance(mask, depth, confidence)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    extent_w_m = w_px * D / fx
    extent_h_m = h_px * D / fy
    obj_pts = points[mask]
    lo = np.percentile(obj_pts, 2, axis=0)
    hi = np.percentile(obj_pts, 98, axis=0)
    return {
        "bbox_2d_px": (u0, v0, u1, v1),
        "bbox_2d_wh_px": (w_px, h_px),
        "D_m": D,
        "extent_w_m": extent_w_m,
        "extent_h_m": extent_h_m,
        "extent_3d_m_xyz": [float(v) for v in (hi - lo)],
        "n_object_pixels": int(mask.sum()),
    }


# ----------------------------------------------------------------------------
# GLB silhouette rendering (pytorch3d, GPU)
# ----------------------------------------------------------------------------

def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _load_glb_as_combined_mesh(glb_path: str) -> trimesh.Trimesh:
    """Load GLB and concatenate all geometry into a single Trimesh (transforms baked in)."""
    obj = trimesh.load(glb_path, force="scene", process=False)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    geoms = []
    for name, geom in obj.geometry.items():
        try:
            tf = obj.graph.get(name)[0]
        except Exception:
            tf = np.eye(4)
        g = geom.copy()
        g.apply_transform(tf)
        geoms.append(g)
    if not geoms:
        raise ValueError(f"no geometry in GLB: {glb_path}")
    return trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]


class _NVDR:
    _ctx = None

    @classmethod
    def get(cls):
        if cls._ctx is None:
            import nvdiffrast.torch as dr
            cls._ctx = dr.RasterizeCudaContext()
        return cls._ctx


def _build_projection(K: np.ndarray, image_size: Tuple[int, int],
                      near: float = 0.05, far: float = 1000.0) -> np.ndarray:
    """OpenGL-style perspective projection matrix from intrinsics K and image size (W, H)."""
    W, H = image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    P = np.array([
        [2 * fx / W, 0,          1 - 2 * cx / W,                0],
        [0,          2 * fy / H, 2 * cy / H - 1,                0],
        [0,          0,          -(far + near) / (far - near),  -2 * far * near / (far - near)],
        [0,          0,          -1,                            0],
    ], dtype=np.float32)
    return P


def render_glb_silhouette(glb_path: str, K: np.ndarray, image_size: Tuple[int, int],
                          obj_distance_m: float,
                          view_R: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """Render binary silhouette via nvdiffrast (CUDA). GLB is centered at origin; camera at
    world (0, 0, D) looking down -Z (OpenGL convention).

    view_R: 3x3 rotation applied to GLB vertices before camera transform (multi-view sweep).
    Returns (HxW bool silhouette, debug_info).
    """
    import nvdiffrast.torch as dr

    W, H = image_size
    tri = _load_glb_as_combined_mesh(glb_path)
    verts_w = tri.vertices.astype(np.float32)  # (V, 3)
    faces = tri.faces.astype(np.int32)         # (F, 3)
    center = verts_w.mean(axis=0)
    verts_w = verts_w - center
    if view_R is not None:
        verts_w = verts_w @ view_R.astype(np.float32).T

    # world -> camera: camera at (0, 0, D), looking -Z (OpenGL). View just translates: z_cam = z - D.
    verts_cam = verts_w.copy()
    verts_cam[:, 2] -= float(obj_distance_m)
    # clip space = P @ (x, y, z, 1)
    verts_cam_h = np.column_stack([verts_cam, np.ones(len(verts_cam), dtype=np.float32)])  # (V, 4)
    P = _build_projection(K, (W, H))
    verts_clip = verts_cam_h @ P.T  # (V, 4) in clip space

    pos = torch.from_numpy(verts_clip)[None].contiguous().to("cuda")  # (1, V, 4)
    tri_t = torch.from_numpy(faces).contiguous().to("cuda")            # (F, 3) int32

    glctx = _NVDR.get()
    rast, _ = dr.rasterize(glctx, pos, tri_t, resolution=[H, W])
    silh_gpu = (rast[0, ..., 3] > 0)  # (H, W) bool — nvdiffrast outputs origin at bottom-left
    silh = silh_gpu.cpu().numpy()[::-1, :].copy()  # flip vertically to match image coords (top-left origin)

    if silh.any():
        ys, xs = np.where(silh)
        r_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        r_w = r_bbox[2] - r_bbox[0]
        r_h = r_bbox[3] - r_bbox[1]
    else:
        r_bbox = (0, 0, 0, 0); r_w = r_h = 0

    return silh, {
        "rendered_bbox_2d_px": r_bbox,
        "rendered_bbox_2d_wh_px": (r_w, r_h),
        "n_pixels": int(silh.sum()),
        "glb_extent_units_xyz": [float(v) for v in (tri.bounds[1] - tri.bounds[0])],
    }


# ----------------------------------------------------------------------------
# View alignment — shape IoU (bbox-cropped + resized)
# ----------------------------------------------------------------------------

def _shape_iou(a: np.ndarray, b: np.ndarray, target: int = 128) -> float:
    """IoU between two binary masks ignoring position: crop each to its bbox, resize to target^2, compare."""
    if not a.any() or not b.any():
        return 0.0
    def _crop_resize(m):
        ys, xs = np.where(m)
        sub = m[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.uint8) * 255
        im = Image.fromarray(sub).resize((target, target), Image.NEAREST)
        return np.array(im) > 128
    aa, bb = _crop_resize(a), _crop_resize(b)
    inter = (aa & bb).sum()
    union = (aa | bb).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def find_best_view(glb_path: str, image_mask: np.ndarray, K: np.ndarray,
                   image_size: Tuple[int, int], obj_distance_m: float) -> dict:
    """Canonical first; if shape-IoU < 0.5, sweep elev x azim. Return best."""
    silh, info = render_glb_silhouette(glb_path, K, image_size, obj_distance_m, view_R=None)
    iou = _shape_iou(silh, image_mask)
    best = {"R": np.eye(3, dtype=np.float32), "iou": iou, "method": "canonical",
            "info": info, "silhouette": silh, "azim_deg": 0.0, "elev_deg": 0.0}
    if iou >= 0.5:
        return best

    for elev in (0.0, 15.0, -15.0):
        for azim in (45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0):
            Rm = _rot_y(np.radians(azim)) @ _rot_x(np.radians(elev))
            silh, info = render_glb_silhouette(glb_path, K, image_size, obj_distance_m, view_R=Rm)
            iou = _shape_iou(silh, image_mask)
            if iou > best["iou"]:
                best = {"R": Rm, "iou": iou, "method": "multi_view",
                        "info": info, "silhouette": silh, "azim_deg": azim, "elev_deg": elev}
    return best


# ----------------------------------------------------------------------------
# Scale solver + GLB bake
# ----------------------------------------------------------------------------

def solve_scale(image_bbox_wh_px: Tuple[int, int],
                render_bbox_wh_px: Tuple[int, int]) -> float:
    """Match longest 2D dim. Both bboxes measured/rendered at same K and same object distance,
    so ratio of longest pixel dim == required uniform scale factor."""
    img_long = max(image_bbox_wh_px)
    ren_long = max(render_bbox_wh_px)
    if ren_long <= 0:
        raise ValueError("rendered bbox has zero extent")
    return img_long / ren_long


def bake_scale_into_glb(glb_path: str, scale: float, out_path: Optional[str] = None) -> dict:
    """Uniformly scale a GLB and save back. trimesh.Scene.apply_scale preserves materials."""
    out_path = out_path or glb_path
    scene = trimesh.load(glb_path, force="scene", process=False)
    orig = (scene.bounds[1] - scene.bounds[0]).tolist()
    scene.apply_scale(float(scale))
    new = (scene.bounds[1] - scene.bounds[0]).tolist()
    scene.export(out_path)
    return {"scale": float(scale),
            "original_extent_xyz": [float(v) for v in orig],
            "scaled_extent_xyz_m": [float(v) for v in new]}


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------

def _save_debug_overlay(image_path: str, image_mask: np.ndarray, rendered_silh: np.ndarray,
                        out_png: str) -> None:
    """Save a side-by-side debug PNG: input | rendered overlay | matched silhouettes."""
    try:
        import matplotlib.pyplot as plt
        rgb = np.array(Image.open(image_path).convert("RGB"))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(rgb); axes[0].set_title("input"); axes[0].axis("off")
        axes[1].imshow(rgb)
        axes[1].imshow(np.ma.masked_where(~rendered_silh, rendered_silh),
                       cmap="autumn", alpha=0.5)
        axes[1].set_title("rendered GLB silhouette over input"); axes[1].axis("off")
        ov = np.zeros((*image_mask.shape, 3), dtype=np.uint8)
        ov[image_mask] = (0, 200, 0)        # green = image mask
        ov[rendered_silh] = ov[rendered_silh] // 2 + np.array([100, 0, 0], dtype=np.uint8)  # +red = render
        axes[2].imshow(ov); axes[2].set_title("image-mask (green) vs render (red)"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=70, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning("debug overlay save failed: %s", e)


def auto_scale(glb_path: str, image_path: str, *, in_place: bool = True,
               debug_png: Optional[str] = None,
               use_class_prior: bool = True) -> dict:
    """Run the full pipeline. Returns metadata dict suitable for the gallery index entry.

    Tier 1 improvements (2026-06-06):
        - CLIP zero-shot classifier identifies object class + plausible dim range
        - rembg mask refined by depth gating + UniDepth confidence percentile
        - object distance D uses confidence-weighted median
        - class prior blends geometric prediction toward plausible range when off

    Safe-fails: any exception sets auto_scaled=False with a reason, never re-raises.
    """
    out: dict = {"auto_scaled": False, "scale_source": None, "mask_source": None}
    try:
        if not os.path.exists(glb_path) or not os.path.exists(image_path):
            raise FileNotFoundError(f"missing input: glb={glb_path} img={image_path}")

        # 1. mask (rembg / alpha / fullframe ladder)
        mask, mask_src = get_object_mask(image_path)
        out["mask_source"] = mask_src

        # 2. UniDepth: depth + intrinsics + confidence + pre-unprojected 3D points
        dk = estimate_depth_intrinsics(image_path)
        depth, K, points, image_size = dk["depth"], dk["K"], dk["points"], dk["image_size"]
        confidence_map = dk.get("confidence")

        # 3. NEW: refine mask using depth gating + confidence percentile
        mask_refined, mask_refine_info = refine_mask_by_depth(mask, depth, confidence_map)

        # 4. compute metric bbox using refined mask + confidence-weighted distance
        img_m = compute_image_metric_bbox(mask_refined, depth, K, points, confidence=confidence_map)

        # 5. view alignment (silhouette IoU search)
        best = find_best_view(glb_path, mask_refined, K, image_size, obj_distance_m=img_m["D_m"])

        # 6. solve geometric scale
        geom_scale = solve_scale(img_m["bbox_2d_wh_px"], best["info"]["rendered_bbox_2d_wh_px"])

        # 7. NEW: class prior blend — preview-only on the geometric prediction
        class_meta: Optional[dict] = None
        prior_meta: Optional[dict] = None
        scale = geom_scale
        if use_class_prior:
            try:
                from .class_priors import classify_object, sanity_check
                class_pred = classify_object(image_path)
                # The "geometric predicted longest dim" before baking = geom_scale × max(GLB unit extent)
                # We get that from the renderer's debug info or by loading the GLB. Simplest: load and check.
                tri = _load_glb_as_combined_mesh(glb_path)
                glb_extent = (tri.bounds[1] - tri.bounds[0])
                glb_longest_unit = float(max(glb_extent))
                geom_longest_m = geom_scale * glb_longest_unit
                prior = sanity_check(geom_longest_m, class_pred)
                # apply blended correction
                scale = geom_scale * prior.snap_scale_factor
                class_meta = {
                    "label": class_pred.label,
                    "classifier_confidence": class_pred.confidence,
                    "top3": class_pred.top3,
                    "plausible_range_m": [class_pred.min_m, class_pred.max_m],
                    "median_m": class_pred.median_m,
                }
                prior_meta = {
                    "in_range": prior.in_range,
                    "snap_applied": prior.snap_applied,
                    "pre_snap_longest_m": prior.pre_snap_longest_m,
                    "post_snap_longest_m": prior.post_snap_longest_m,
                    "snap_scale_factor": prior.snap_scale_factor,
                    "confidence_modifier": prior.confidence_modifier,
                    "blend_alpha": prior.blend_alpha,
                }
            except Exception as e:
                logger.warning("class prior step failed (continuing without): %s", e)

        # 8. bake final scale into GLB
        target_path = glb_path if in_place else glb_path.replace(".glb", "_scaled.glb")
        bake = bake_scale_into_glb(glb_path, scale, target_path)

        scaled = bake["scaled_extent_xyz_m"]
        sorted_dims = sorted(scaled, reverse=True)

        # 9. confidence tier — IoU + mask source + class prior agreement
        if best["iou"] >= 0.5 and mask_src in ("alpha", "rembg"):
            confidence = "high"
        elif best["iou"] >= 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        # downgrade if prior triggered
        if prior_meta:
            mod = prior_meta["confidence_modifier"]
            if mod == "downgrade" and confidence == "high":
                confidence = "medium"
            elif mod == "downgrade2":
                confidence = "low"

        if debug_png:
            _save_debug_overlay(image_path, mask_refined, best["silhouette"], debug_png)

        out.update({
            "auto_scaled": True,
            "scale_source": "unidepth_view_align_classprior" if class_meta else "unidepth_view_align",
            "scale_factor": float(scale),
            "geometric_scale_factor": float(geom_scale),
            "dimensions_m": {
                "xyz": scaled,
                "longest_m": float(sorted_dims[0]),
                "middle_m": float(sorted_dims[1]),
                "shortest_m": float(sorted_dims[2]),
            },
            "view_alignment": {
                "method": best["method"],
                "iou": float(best["iou"]),
                "azim_deg": float(best["azim_deg"]),
                "elev_deg": float(best["elev_deg"]),
            },
            "object_distance_m": img_m["D_m"],
            "image_metrics": {
                "bbox_2d_px": list(img_m["bbox_2d_px"]),
                "extent_w_m": img_m["extent_w_m"],
                "extent_h_m": img_m["extent_h_m"],
            },
            "mask_refinement": mask_refine_info,
            "class_prediction": class_meta,
            "class_prior_check": prior_meta,
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            },
            "confidence": confidence,
        })
    except Exception as e:
        logger.exception("auto_scale failed: %s", e)
        out["reason"] = str(e)
    return out


# ----------------------------------------------------------------------------
# CLI entry (standalone test)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Auto-scale a GLB to real-world meters using its source image.")
    p.add_argument("glb_path")
    p.add_argument("image_path")
    p.add_argument("--no-in-place", action="store_true", help="write <glb>_scaled.glb instead of overwriting")
    p.add_argument("--debug-png", default=None, help="save side-by-side debug PNG to this path")
    args = p.parse_args()
    res = auto_scale(args.glb_path, args.image_path,
                     in_place=not args.no_in_place,
                     debug_png=args.debug_png)
    print(json.dumps(res, indent=2))
