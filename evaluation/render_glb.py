"""
Render GLB models using nvdiffrast with proper UV texture mapping.
"""

import os
import json
import numpy as np
import torch
import trimesh
from PIL import Image
import nvdiffrast.torch as dr


def load_mesh(glb_path):
    """Load GLB with vertices, faces, UVs, and texture map."""
    scene = trimesh.load(glb_path)

    # Get the first mesh from scene
    if isinstance(scene, trimesh.Scene):
        geom = list(scene.geometry.values())[0]
    else:
        geom = scene

    vertices = torch.tensor(geom.vertices, dtype=torch.float32, device="cuda")
    faces = torch.tensor(geom.faces, dtype=torch.int32, device="cuda")

    # Get UV coordinates
    uv = None
    texture = None
    if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
        uv = torch.tensor(geom.visual.uv, dtype=torch.float32, device="cuda")
        # Get texture image
        mat = geom.visual.material
        tex = getattr(mat, 'baseColorTexture', None)
        if tex is not None:
            if isinstance(tex, np.ndarray):
                texture = torch.tensor(tex, dtype=torch.float32, device="cuda") / 255.0
            else:
                # PIL Image
                texture = torch.tensor(np.array(tex.convert("RGB")), dtype=torch.float32, device="cuda") / 255.0

    # Fallback: bake to vertex colors
    if uv is None or texture is None:
        try:
            color_vis = geom.visual.to_color()
            colors = torch.tensor(
                color_vis.vertex_colors[:, :3].astype(np.float32) / 255.0,
                device="cuda",
            )
        except Exception:
            colors = torch.ones(vertices.shape[0], 3, dtype=torch.float32, device="cuda") * 0.7
        return vertices, faces, colors, None, None

    return vertices, faces, None, uv, texture


def make_mvp_matrix(vertices):
    """Create MVP matrix that frames the object from front view."""
    vmin = vertices.min(dim=0).values
    vmax = vertices.max(dim=0).values
    center = (vmin + vmax) / 2.0
    scale = (vmax - vmin).max().item()

    eye_dist = 2.0
    eye = torch.tensor([0.0, 0.0, eye_dist], device="cuda")
    at = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    up = torch.tensor([0.0, 1.0, 0.0], device="cuda")

    z_axis = eye - at
    z_axis = z_axis / z_axis.norm()
    x_axis = torch.linalg.cross(up, z_axis)
    x_axis = x_axis / x_axis.norm()
    y_axis = torch.linalg.cross(z_axis, x_axis)

    view = torch.eye(4, device="cuda")
    view[:3, 0] = x_axis
    view[:3, 1] = y_axis
    view[:3, 2] = z_axis
    view[0, 3] = -torch.dot(x_axis, eye)
    view[1, 3] = -torch.dot(y_axis, eye)
    view[2, 3] = -torch.dot(z_axis, eye)

    model = torch.eye(4, device="cuda")
    model[0, 0] = model[1, 1] = model[2, 2] = 2.0 / scale
    model[0, 3] = -center[0] * 2.0 / scale
    model[1, 3] = -center[1] * 2.0 / scale
    model[2, 3] = -center[2] * 2.0 / scale

    fov = 40.0
    near, far = 0.01, 10.0
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    proj = torch.zeros(4, 4, device="cuda")
    proj[0, 0] = f
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0

    return proj @ view @ model


def compute_face_normals(vertices, faces):
    """Compute per-face normals for basic shading."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = torch.linalg.cross(v1 - v0, v2 - v0)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    return normals


def render_mesh(glb_path, output_path, resolution=512):
    """Render a GLB file to PNG with texture mapping and basic shading."""
    vertices, faces, vert_colors, uv, texture = load_mesh(glb_path)

    mvp = make_mvp_matrix(vertices)

    # Transform to clip space
    v_hom = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device="cuda")], dim=1)
    v_clip = (mvp @ v_hom.T).T.unsqueeze(0).contiguous()
    faces_b = faces.contiguous()

    # Rasterize
    glctx = dr.RasterizeCudaContext()
    rast, rast_db = dr.rasterize(glctx, v_clip, faces_b, resolution=[resolution, resolution])

    mask = (rast[..., 3:4] > 0).float()

    if uv is not None and texture is not None:
        # Texture-mapped rendering
        uv_b = uv.unsqueeze(0).contiguous()
        texc, texc_db = dr.interpolate(uv_b, rast, faces_b, rast_db=rast_db, diff_attrs='all')

        # Sample texture (need HWC texture as [1, H, W, C])
        tex_b = texture.unsqueeze(0).contiguous()
        color = dr.texture(tex_b, texc, texc_db, filter_mode='linear')
    else:
        # Vertex color rendering
        colors_b = vert_colors.unsqueeze(0).contiguous()
        color, _ = dr.interpolate(colors_b, rast, faces_b)

    # Basic directional lighting using face normals
    # Compute per-vertex normals from face normals
    face_normals = compute_face_normals(vertices, faces)
    vert_normals = torch.zeros_like(vertices)
    vert_normals.index_add_(0, faces[:, 0], face_normals)
    vert_normals.index_add_(0, faces[:, 1], face_normals)
    vert_normals.index_add_(0, faces[:, 2], face_normals)
    vert_normals = vert_normals / (vert_normals.norm(dim=1, keepdim=True) + 1e-8)

    # Interpolate normals
    normals_b = vert_normals.unsqueeze(0).contiguous()
    normals_interp, _ = dr.interpolate(normals_b, rast, faces_b)

    # Light from camera direction (0, 0, 1)
    light_dir = torch.tensor([0.0, 0.3, 1.0], device="cuda")
    light_dir = light_dir / light_dir.norm()
    diffuse = torch.clamp(torch.sum(normals_interp[0] * light_dir, dim=-1, keepdim=True), 0.0, 1.0)

    # Ambient + diffuse lighting
    ambient = 0.4
    lit_color = color[0] * (ambient + (1.0 - ambient) * diffuse)

    # White background
    img = lit_color * mask[0] + (1.0 - mask[0])

    img_np = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save(output_path)
    return img_np


def render_all_generations(results_file, gallery_dir, output_dir):
    """Render all generated models."""
    os.makedirs(output_dir, exist_ok=True)

    with open(results_file) as f:
        generations = json.load(f)

    for gen in generations:
        if gen["status"] != "completed":
            continue

        task_id = gen["task_id"]
        engine = gen["engine"]
        img_name = gen["image"]

        glb_path = os.path.join(gallery_dir, task_id, "model.glb")
        if not os.path.exists(glb_path):
            print(f"  SKIP {task_id}: no model.glb")
            continue

        out_name = f"{os.path.splitext(img_name)[0]}_{engine}.png"
        out_path = os.path.join(output_dir, out_name)

        print(f"Rendering [{engine}] {img_name} -> {out_name}")
        try:
            render_mesh(glb_path, out_path)
            print(f"  OK")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    render_all_generations(
        "/app/evaluation/generation_results.json",
        "/app/gallery",
        "/app/evaluation/renders",
    )
