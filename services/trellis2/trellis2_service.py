"""TRELLIS.2 image-to-3D HTTP microservice (host side).

Runs in the `trellis2` conda env on the host, holds the 4B model resident, and returns a GLB
per request. The HTX-3D backend's `trellis2` proxy engine (inside the container) calls this over
the docker bridge gateway. Model is lazy-loaded on first /generate so the service starts instantly.

Endpoints:
  GET  /health              -> {"status","model_loaded"}
  POST /generate            -> multipart: image=<file>, plus form fields:
        seed (int, 0), texture_size (int, 1024), decimation_target (int, 200000)
     -> returns model/gltf-binary (the GLB bytes)

Launch via service/run_service.sh (sets the sm120 / FA2 env).
"""
import os, io, time, threading, tempfile, logging
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("trellis2")

app = FastAPI(title="TRELLIS.2 service")
_pipe = None
_lock = threading.Lock()          # serialize GPU work (one generation at a time)
_load_lock = threading.Lock()


def _get_pipe():
    global _pipe
    if _pipe is None:
        with _load_lock:
            if _pipe is None:
                import torch
                from trellis2.pipelines import Trellis2ImageTo3DPipeline
                t0 = time.time()
                log.info("loading TRELLIS.2-4B ...")
                p = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
                p.cuda()
                _pipe = p
                log.info(f"model loaded in {time.time()-t0:.1f}s")
    return _pipe


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = Form(0),
    texture_size: int = Form(1024),
    decimation_target: int = Form(200000),
):
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:
        raise HTTPException(400, f"bad image: {e}")

    import torch, o_voxel
    pipe = _get_pipe()
    t0 = time.time()
    with _lock:                    # only one GPU job at a time
        try:
            torch.cuda.reset_peak_memory_stats()
            mesh = pipe.run(img, seed=int(seed))[0]
            mesh.simplify(16777216)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(decimation_target), texture_size=int(texture_size),
                remesh=True, remesh_band=1, remesh_project=0, verbose=False,
            )
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tf:
                tmp = tf.name
            glb.export(tmp)          # PNG textures (widely compatible)
            data = open(tmp, "rb").read()
            os.unlink(tmp)
            vram = torch.cuda.max_memory_allocated() / 1e9
        except Exception as e:
            log.exception("generation failed")
            raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}")
        finally:
            torch.cuda.empty_cache()
    dt = time.time() - t0
    log.info(f"generated {len(data)/1e6:.1f}MB glb in {dt:.1f}s (peak {vram:.1f}GB)")
    return Response(content=data, media_type="model/gltf-binary",
                    headers={"X-Gen-Seconds": f"{dt:.1f}", "X-Peak-VRAM-GB": f"{vram:.1f}"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("TRELLIS2_SERVICE_PORT", "8710"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
