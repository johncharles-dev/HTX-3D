"""Bake a PNG logo onto a textured mesh's albedo, projecting it onto the surface.

Used by the logo decal feature as the "bake into surface" path: instead of
keeping the logo as an overlay mesh, this projects it into the model's actual
albedo texture (UV space), optionally inpainting the old surface content under
the footprint first ("smudge") so the new logo reads cleanly. Works on any
single-mesh, UV-mapped, textured GLB.

Coordinate space: placements arrive in the GLB scene-root frame (the frontend
sends positions via loadedScene.worldToLocal and normals via the inverse normal
matrix). trimesh.load(force='mesh') bakes node transforms into the same root
frame, so the two agree without a display <Center> offset.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import trimesh
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


def _decal_basis(normal: np.ndarray, rotation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal projector frame (u, v, w) with w along the surface normal,
    plus an in-plane rotation around w."""
    w = normal / (np.linalg.norm(normal) + 1e-12)
    up = np.array([0.0, 1.0, 0.0]) if abs(w[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    u = np.cross(up, w)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(w, u)
    c, s = np.cos(rotation), np.sin(rotation)
    return c * u + s * v, -s * u + c * v, w


def _get_base_image(material: Any) -> Image.Image | None:
    """Extract the base color texture as a PIL image across trimesh material types."""
    for attr in ("baseColorTexture", "image"):
        img = getattr(material, attr, None)
        if img is not None:
            return img.convert("RGB")
    return None


def _set_base_image(mesh: trimesh.Trimesh, img: Image.Image) -> None:
    material = mesh.visual.material
    if hasattr(material, "baseColorTexture"):
        material.baseColorTexture = img
    elif hasattr(material, "image"):
        material.image = img
    else:  # promote to a PBR material
        mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorTexture=img)


def bake_logos(
    glb_bytes: bytes,
    logo_bytes: bytes,
    placements: list[dict],
    smudge: bool = True,
    target_resolution: int = 4096,
) -> bytes:
    """Project each placement's logo onto the mesh albedo and return a new GLB.

    placements: list of dicts with keys px,py,pz (position), nx,ny,nz (normal),
    size (footprint in model units), rotation (radians, in-plane).
    target_resolution: the albedo is upscaled so its longest side is at least
    this many pixels before baking, giving the logo footprint enough texels to
    keep fine sub-text crisp (a small decal only covers a fraction of the atlas).
    """
    mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("GLB did not load as a single mesh")
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        raise ValueError("Mesh has no UV coordinates; cannot bake into texture")
    uv = np.asarray(uv, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

    base_img = _get_base_image(mesh.visual.material)
    if base_img is None:
        base_img = Image.new("RGB", (1024, 1024), (200, 200, 200))
    # Upscale the albedo so the small decal footprint has enough texels for
    # fine detail. Lanczos keeps the existing texture acceptably sharp.
    longest = max(base_img.size)
    if target_resolution and longest < target_resolution:
        s = target_resolution / longest
        base_img = base_img.resize((round(base_img.width * s), round(base_img.height * s)), Image.LANCZOS)
    W, H = base_img.size
    albedo = np.asarray(base_img, dtype=np.uint8).copy()

    logo_arr = np.asarray(Image.open(io.BytesIO(logo_bytes)).convert("RGBA"), dtype=np.float64)
    lh, lw = logo_arr.shape[:2]

    logo_buf = np.zeros((H, W, 4), dtype=np.float64)  # accumulated logo RGBA per texel
    footprint = np.zeros((H, W), dtype=np.uint8)       # decal coverage (for inpaint mask)

    for pl in placements:
        pos = np.array([pl["px"], pl["py"], pl["pz"]], dtype=np.float64)
        nrm = np.array([pl["nx"], pl["ny"], pl["nz"]], dtype=np.float64)
        size = float(pl["size"])
        rot = float(pl.get("rotation", 0.0))
        u, v, w = _decal_basis(nrm, rot)
        # Uniform projector box (three.js decal convention). A shallow depth
        # keeps the logo on the local surface patch instead of punching through
        # to far panels on layered geometry.
        hx = hy = hz = size / 2.0

        rel = verts - pos
        du = rel @ u
        dv = rel @ v
        dw = rel @ w

        # Candidate faces: any vertex roughly inside the projector box, front-facing.
        within = (np.abs(du) <= hx * 1.5) & (np.abs(dv) <= hy * 1.5) & (np.abs(dw) <= hz * 1.5)
        facing = (face_normals @ w) > 0.0
        cand = np.where(within[faces].any(axis=1) & facing)[0]

        for fi in cand:
            a, b, c = faces[fi]
            tri = uv[[a, b, c]]
            xpix = tri[:, 0] * (W - 1)
            ypix = (1.0 - tri[:, 1]) * (H - 1)  # glTF v origin is bottom; image row 0 is top
            minx, maxx = max(int(np.floor(xpix.min())), 0), min(int(np.ceil(xpix.max())), W - 1)
            miny, maxy = max(int(np.floor(ypix.min())), 0), min(int(np.ceil(ypix.max())), H - 1)
            if maxx < minx or maxy < miny:
                continue

            x0, y0, x1, y1, x2, y2 = xpix[0], ypix[0], xpix[1], ypix[1], xpix[2], ypix[2]
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-9:
                continue
            ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
            l1 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
            l2 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
            l3 = 1.0 - l1 - l2
            inside = (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4)
            if not inside.any():
                continue

            du_f = l1 * du[a] + l2 * du[b] + l3 * du[c]
            dv_f = l1 * dv[a] + l2 * dv[b] + l3 * dv[c]
            dw_f = l1 * dw[a] + l2 * dw[b] + l3 * dw[c]
            lu = du_f / size + 0.5
            lv = dv_f / size + 0.5
            valid = inside & (lu >= 0) & (lu <= 1) & (lv >= 0) & (lv <= 1) & (np.abs(dw_f) <= hz)
            if not valid.any():
                continue

            # Bilinear sample the logo (smooth edges, no nearest-neighbour aliasing).
            fx = lu[valid] * (lw - 1)
            fy = (1.0 - lv[valid]) * (lh - 1)
            x0 = np.floor(fx).astype(np.int64); y0 = np.floor(fy).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, lw - 1); y1 = np.clip(y0 + 1, 0, lh - 1)
            x0 = np.clip(x0, 0, lw - 1); y0 = np.clip(y0, 0, lh - 1)
            wx = (fx - x0)[:, None]; wy = (fy - y0)[:, None]
            sampled = (
                logo_arr[y0, x0] * (1 - wx) * (1 - wy)
                + logo_arr[y0, x1] * wx * (1 - wy)
                + logo_arr[y1, x0] * (1 - wx) * wy
                + logo_arr[y1, x1] * wx * wy
            )
            yy, xx = ys[valid], xs[valid]
            footprint[yy, xx] = 255
            logo_buf[yy, xx] = sampled

    if not footprint.any():
        raise ValueError("Logo projection produced no coverage — check placement and size")

    if smudge:
        # Heal the old surface content under the whole footprint before compositing.
        albedo = cv2.inpaint(albedo, footprint, 3, cv2.INPAINT_TELEA)

    mask = footprint > 0
    alpha = logo_buf[..., 3:4] / 255.0
    comp = albedo.astype(np.float64)
    comp[mask] = (1.0 - alpha[mask]) * comp[mask] + alpha[mask] * logo_buf[..., :3][mask]
    albedo = np.clip(comp, 0, 255).astype(np.uint8)

    _set_base_image(mesh, Image.fromarray(albedo, "RGB"))
    result = mesh.export(file_type="glb")
    coverage = int(mask.sum())
    logger.info("Logo bake: %d placements, %d texels covered, smudge=%s", len(placements), coverage, smudge)
    return result
