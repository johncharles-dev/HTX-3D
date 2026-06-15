"""Batch renderer — runs INSIDE the htx-3d container with nvdiffrast.

Reads a job manifest at /app/gallery/_bench_render_jobs.json:
    [{"task_id": "...", "obj_id": "...", "condition": "..."}]

For each entry, renders /app/gallery/<task_id>/model.glb to
/app/gallery/_bench_renders/<obj_id>__<condition>.png at 512x512
from a canonical front view (matches evaluation/render_glb.py).
"""

import os
import json
import sys
import traceback

sys.path.insert(0, "/app/evaluation")

import numpy as np
import torch
import trimesh
from PIL import Image
import nvdiffrast.torch as dr


JOB_FILE = "/app/gallery/_bench_render_jobs.json"
OUT_DIR = "/app/gallery/_bench_renders"
RES = 512


def load_mesh(glb_path: str):
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        geom = list(scene.geometry.values())[0]
    else:
        geom = scene

    vertices = torch.tensor(geom.vertices, dtype=torch.float32, device="cuda")
    faces = torch.tensor(geom.faces, dtype=torch.int32, device="cuda")

    uv = None
    texture = None
    if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
        uv = torch.tensor(geom.visual.uv, dtype=torch.float32, device="cuda")
        mat = geom.visual.material
        tex = getattr(mat, "baseColorTexture", None)
        if tex is not None:
            if isinstance(tex, np.ndarray):
                texture = torch.tensor(tex, dtype=torch.float32, device="cuda") / 255.0
            else:
                texture = torch.tensor(np.array(tex.convert("RGB")), dtype=torch.float32, device="cuda") / 255.0
    if uv is None or texture is None:
        try:
            color_vis = geom.visual.to_color()
            colors = torch.tensor(
                color_vis.vertex_colors[:, :3].astype(np.float32) / 255.0, device="cuda"
            )
        except Exception:
            colors = torch.ones(vertices.shape[0], 3, dtype=torch.float32, device="cuda") * 0.7
        return vertices, faces, colors, None, None
    return vertices, faces, None, uv, texture


def make_mvp(vertices):
    vmin = vertices.min(dim=0).values
    vmax = vertices.max(dim=0).values
    center = (vmin + vmax) / 2.0
    scale = (vmax - vmin).max().item()

    eye_dist = 2.0
    eye = torch.tensor([0.0, 0.0, eye_dist], device="cuda")
    at = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    up = torch.tensor([0.0, 1.0, 0.0], device="cuda")

    z_axis = (eye - at) / (eye - at).norm()
    x_axis = torch.linalg.cross(up, z_axis); x_axis = x_axis / x_axis.norm()
    y_axis = torch.linalg.cross(z_axis, x_axis)

    view = torch.eye(4, device="cuda")
    view[:3, 0] = x_axis; view[:3, 1] = y_axis; view[:3, 2] = z_axis
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
    proj[0, 0] = f; proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj @ view @ model


def face_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]; v1 = vertices[faces[:, 1]]; v2 = vertices[faces[:, 2]]
    n = torch.linalg.cross(v1 - v0, v2 - v0)
    return n / (n.norm(dim=1, keepdim=True) + 1e-8)


def render_one(glb_path: str, out_path: str, glctx):
    vertices, faces, vcolors, uv, texture = load_mesh(glb_path)
    mvp = make_mvp(vertices)
    v_hom = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device="cuda")], dim=1)
    v_clip = (mvp @ v_hom.T).T.unsqueeze(0).contiguous()
    fb = faces.contiguous()

    rast, rast_db = dr.rasterize(glctx, v_clip, fb, resolution=[RES, RES])
    mask = (rast[..., 3:4] > 0).float()

    if uv is not None and texture is not None:
        uv_b = uv.unsqueeze(0).contiguous()
        texc, texc_db = dr.interpolate(uv_b, rast, fb, rast_db=rast_db, diff_attrs="all")
        tex_b = texture.unsqueeze(0).contiguous()
        color = dr.texture(tex_b, texc, texc_db, filter_mode="linear")
    else:
        cb = vcolors.unsqueeze(0).contiguous()
        color, _ = dr.interpolate(cb, rast, fb)

    fn = face_normals(vertices, faces)
    vn = torch.zeros_like(vertices)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    vn = vn / (vn.norm(dim=1, keepdim=True) + 1e-8)

    nb = vn.unsqueeze(0).contiguous()
    ni, _ = dr.interpolate(nb, rast, fb)
    light = torch.tensor([0.0, 0.3, 1.0], device="cuda"); light = light / light.norm()
    diff = torch.clamp(torch.sum(ni[0] * light, dim=-1, keepdim=True), 0.0, 1.0)
    ambient = 0.4
    lit = color[0] * (ambient + (1.0 - ambient) * diff)
    img = lit * mask[0] + (1.0 - mask[0])
    arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(JOB_FILE) as f:
        jobs = json.load(f)
    print(f"Rendering {len(jobs)} GLBs...")

    glctx = dr.RasterizeCudaContext()
    ok = 0; fail = 0
    for j in jobs:
        glb = f"/app/gallery/{j['task_id']}/model.glb"
        out = f"{OUT_DIR}/{j['obj_id']}__{j['condition']}.png"
        if not os.path.exists(glb):
            print(f"  SKIP (no GLB) {j['obj_id']}/{j['condition']}")
            fail += 1
            continue
        try:
            render_one(glb, out, glctx)
            ok += 1
            print(f"  ok  {j['obj_id']:32s} {j['condition']:15s} -> {os.path.basename(out)}")
        except Exception as e:
            fail += 1
            print(f"  ERR {j['obj_id']}/{j['condition']}: {e}")
            traceback.print_exc()
    print(f"\nDone. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
