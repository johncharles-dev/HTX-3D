"""Orbit renderer — runs INSIDE the htx-3d container (nvdiffrast + CUDA).

Renders every GLB in /app/gallery/_orbit_src at yaw {0,90,180,270} and pitch 30
degrees, the protocol published in the TRELLIS paper (trellis.txt:2327-2328), to
/app/gallery/_orbit_renders/<name>__yaw<NNN>.png at 512x512.

Four views rather than one fixes two things the fixed +Z camera could not:
  * pipelines emit meshes in their own orientation, so a single view showed
    different sides of the object per pipeline
  * janus_duplicate and front_only_texture are undetectable from one view

    docker exec htx-3d python /app/gallery/_orbit_render.py
"""

import os
import traceback

import numpy as np
import torch
import trimesh
from PIL import Image
import nvdiffrast.torch as dr

SRC = "/app/gallery/_orbit_src"
OUT = "/app/gallery/_orbit_renders"
RES = 512
YAWS = [0, 90, 180, 270]
PITCH = 30.0
FOV = 40.0


def load_mesh(path):
    scene = trimesh.load(path)
    geom = (list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene)
    v = torch.tensor(geom.vertices, dtype=torch.float32, device="cuda")
    f = torch.tensor(geom.faces, dtype=torch.int32, device="cuda")

    uv = tex = None
    if getattr(geom.visual, "uv", None) is not None:
        uv = torch.tensor(np.asarray(geom.visual.uv), dtype=torch.float32, device="cuda")
        t = getattr(geom.visual.material, "baseColorTexture", None)
        if t is not None:
            arr = t if isinstance(t, np.ndarray) else np.array(t.convert("RGB"))
            tex = torch.tensor(arr, dtype=torch.float32, device="cuda") / 255.0
    if uv is None or tex is None:
        try:
            cols = geom.visual.to_color().vertex_colors[:, :3].astype(np.float32) / 255.0
            return v, f, torch.tensor(cols, device="cuda"), None, None
        except Exception:
            return v, f, torch.ones(v.shape[0], 3, device="cuda") * 0.7, None, None
    return v, f, None, uv, tex


def mvp(vertices, yaw_deg, pitch_deg):
    """Normalise the mesh to a unit box at the origin, then orbit the camera."""
    vmin, vmax = vertices.min(0).values, vertices.max(0).values
    center = (vmin + vmax) / 2.0
    scale = (vmax - vmin).max().item()

    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    r = 2.6
    eye = torch.tensor([r*np.sin(y)*np.cos(p), r*np.sin(p), r*np.cos(y)*np.cos(p)],
                       dtype=torch.float32, device="cuda")
    at = torch.zeros(3, device="cuda")
    up = torch.tensor([0.0, 1.0, 0.0], device="cuda")

    z = (eye - at); z = z / z.norm()
    x = torch.linalg.cross(up, z); x = x / x.norm()
    yv = torch.linalg.cross(z, x)

    view = torch.eye(4, device="cuda")
    view[:3, 0], view[:3, 1], view[:3, 2] = x, yv, z
    view[0, 3] = -torch.dot(x, eye)
    view[1, 3] = -torch.dot(yv, eye)
    view[2, 3] = -torch.dot(z, eye)

    model = torch.eye(4, device="cuda")
    s = 2.0 / scale
    model[0, 0] = model[1, 1] = model[2, 2] = s
    model[:3, 3] = -center * s

    near, far = 0.01, 20.0
    fl = 1.0 / np.tan(np.radians(FOV) / 2.0)
    proj = torch.zeros(4, 4, device="cuda")
    proj[0, 0] = proj[1, 1] = fl
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj @ view @ model


def render(v, f, vcol, uv, tex, glctx, yaw):
    m = mvp(v, yaw, PITCH)
    vh = torch.cat([v, torch.ones(v.shape[0], 1, device="cuda")], dim=1)
    clip = (m @ vh.T).T.unsqueeze(0).contiguous()
    fb = f.contiguous()

    rast, db = dr.rasterize(glctx, clip, fb, resolution=[RES, RES])
    mask = (rast[..., 3:4] > 0).float()

    if uv is not None and tex is not None:
        tc, tcdb = dr.interpolate(uv.unsqueeze(0).contiguous(), rast, fb,
                                  rast_db=db, diff_attrs="all")
        color = dr.texture(tex.unsqueeze(0).contiguous(), tc, tcdb, filter_mode="linear")
    else:
        color, _ = dr.interpolate(vcol.unsqueeze(0).contiguous(), rast, fb)

    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = torch.linalg.cross(v1 - v0, v2 - v0)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-8)
    vn = torch.zeros_like(v)
    for i in range(3):
        vn.index_add_(0, f[:, i].long(), fn)
    vn = vn / (vn.norm(dim=1, keepdim=True) + 1e-8)
    ni, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, fb)

    light = torch.tensor([0.3, 0.45, 1.0], device="cuda")
    light = light / light.norm()
    diff = torch.clamp((ni[0] * light).sum(-1, keepdim=True), 0.0, 1.0)
    lit = color[0] * (0.45 + 0.55 * diff)
    img = lit * mask[0] + (1.0 - mask[0])          # white background
    return (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".glb"))
    print(f"rendering {len(files)} models x {len(YAWS)} views", flush=True)
    glctx = dr.RasterizeCudaContext()
    ok = fail = 0
    for i, fn in enumerate(files, 1):
        stem = fn[:-4]
        try:
            v, f, vcol, uv, tex = load_mesh(os.path.join(SRC, fn))
            for yaw in YAWS:
                arr = render(v, f, vcol, uv, tex, glctx, yaw)
                Image.fromarray(arr).save(f"{OUT}/{stem}__yaw{yaw:03d}.png")
            ok += 1
            if i % 10 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  ok={ok} fail={fail}", flush=True)
        except Exception as e:
            fail += 1
            print(f"  ERR {stem}: {e}", flush=True)
            traceback.print_exc()
    print(f"done. ok={ok} fail={fail}  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
