"""Phase 2 — automated 5-condition x 14-object benchmark.

Reads benchmark_v2/data/<id>/ (produced by prepare_masks.py) and runs all 5
conditions per object via the existing /api/generate/image endpoint:

    1. trellis_rembg   = TRELLIS + built-in rembg                      (no segmented_image_path)
    2. trellis_sam3    = TRELLIS + SAM3 mask + crop+rembg bridge       (segmented_image_path = data/<id>/segmented.png)
    3. hunyuan_rembg   = Hunyuan + built-in rembg                      (no segmented_image_path)
    4. hunyuan_sam3    = Hunyuan + SAM3 mask + crop+rembg bridge       (segmented_image_path = data/<id>/segmented.png)
    5. sam3d_sam3      = SAM 3D Objects + SAM3 mask                    (segmented_image_path = data/<id>/segmented.png)

Engine ordering: ENGINE-BY-ENGINE to minimize swap overhead. The script makes
three passes (TRELLIS, then Hunyuan, then SAM 3D), each running across all
14 objects, before swapping to the next engine.

Outputs are copied from the gallery into:
    benchmark_v2/outputs/<id>/<condition_key>/model.glb

A per-run record is appended to benchmark_v2/results.json.

Usage:
    python run_benchmark.py                       # run all engines, all objects
    python run_benchmark.py --engine trellis      # only TRELLIS pass
    python run_benchmark.py --engine hunyuan
    python run_benchmark.py --engine sam3d
    python run_benchmark.py --only 01_scdf_ambulance 02_scdf_fire_engine
                                                  # only these object ids

Requires: backend container running at API_BASE.
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api"
GALLERY_DIR = "/home/cj/HTX-3D/gallery"
# Segmented images are staged inside the gallery (which is mounted into the
# backend container at /app/gallery) so the API can resolve the path.
BENCH_SEG_HOST_DIR = Path(GALLERY_DIR) / "_bench_seg"
BENCH_SEG_CONTAINER_DIR = "/app/gallery/_bench_seg"
SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "manifest.json"
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RESULTS_PATH = SCRIPT_DIR / "results.json"

POLL_INTERVAL = 5  # seconds
TIMEOUT_PER_TASK = {
    "trellis": 180,
    "hunyuan": 600,
    "sam3d": 600,
}

DEFAULT_PARAMS = {
    "trellis": {
        "seed": 42,
        "randomize_seed": "false",
        "ss_steps": 12,
        "ss_guidance": 7.5,
        "slat_steps": 12,
        "slat_guidance": 3.0,
        "formats": "glb",
        "texture_size": 1024,
    },
    "hunyuan": {
        "seed": 42,
        "randomize_seed": "false",
        "num_inference_steps": 30,
        "guidance_scale": 5.5,
        "octree_resolution": 256,
        "texture": "true",
        "formats": "glb",
        "texture_size": 1024,
    },
    "sam3d": {
        "seed": 42,
        "randomize_seed": "false",
        "sam3d_texture_baking": "true",
        "sam3d_vertex_color": "true",
        "formats": "glb",
        "texture_size": 1024,
    },
}


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def find_original(obj_id: str) -> Path:
    """Locate the original.<ext> file produced by prepare_masks.py."""
    obj_dir = DATA_DIR / obj_id
    for f in obj_dir.iterdir():
        if f.stem == "original":
            return f
    raise FileNotFoundError(f"No original image in {obj_dir}")


def find_segmented(obj_id: str) -> Path:
    """Stage segmented.png inside the gallery mount and return the host path.

    The backend container resolves paths in its own filesystem, so we stage
    each segmented image into /home/cj/HTX-3D/gallery/_bench_seg/<id>.png
    (visible inside the container as /app/gallery/_bench_seg/<id>.png).
    """
    src = DATA_DIR / obj_id / "segmented.png"
    if not src.exists():
        raise FileNotFoundError(f"No segmented image at {src}")
    BENCH_SEG_HOST_DIR.mkdir(parents=True, exist_ok=True)
    staged = BENCH_SEG_HOST_DIR / f"{obj_id}.png"
    if not staged.exists() or staged.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, staged)
    return staged


def container_segmented_path(obj_id: str) -> str:
    return f"{BENCH_SEG_CONTAINER_DIR}/{obj_id}.png"


def submit_generation(image_path: Path, engine: str, segmented_container_path: str | None) -> str:
    params = dict(DEFAULT_PARAMS[engine])
    params["engine"] = engine
    if segmented_container_path is not None:
        params["segmented_image_path"] = segmented_container_path

    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        r = requests.post(f"{API_BASE}/generate/image", files=files, data=params, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"submit failed: {r.status_code} {r.text[:500]}")
    return r.json()["task_id"]


def poll_task(task_id: str, engine: str) -> dict:
    timeout = TIMEOUT_PER_TASK[engine]
    start = time.time()
    last_stage = None
    while time.time() - start < timeout:
        r = requests.get(f"{API_BASE}/task/{task_id}", timeout=30)
        r.raise_for_status()
        st = r.json()
        status = st.get("status", "unknown")
        stage = st.get("stage") or st.get("message") or ""
        if stage and stage != last_stage:
            print(f"      [{int(time.time()-start)}s] {status} — {stage}")
            last_stage = stage
        if status == "completed":
            return st
        if status in ("failed", "cancelled"):
            return st
        time.sleep(POLL_INTERVAL)
    return {"status": "timeout"}


def copy_glb_to_output(task_id: str, obj_id: str, condition_key: str) -> Path:
    src = Path(GALLERY_DIR) / task_id / "model.glb"
    if not src.exists():
        raise FileNotFoundError(f"GLB not produced at {src}")
    dst_dir = OUTPUT_DIR / obj_id / condition_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.glb"
    shutil.copy2(src, dst)
    return dst


def run_one(obj: dict, condition: dict, results_index: dict) -> dict:
    obj_id = obj["id"]
    cond_key = condition["key"]
    engine = condition["engine"]
    use_sam3 = condition["use_sam3"]

    record_key = f"{obj_id}::{cond_key}"
    if record_key in results_index:
        existing = results_index[record_key]
        if existing.get("status") == "completed":
            print(f"    skip (already completed) -> {existing.get('output')}")
            return existing

    original = find_original(obj_id)
    segmented_path_for_api: str | None = None
    if use_sam3:
        find_segmented(obj_id)  # stage host-side; raises if missing
        segmented_path_for_api = container_segmented_path(obj_id)

    print(f"    submit ({engine}, sam3={use_sam3})")
    t0 = time.time()
    task_id = submit_generation(original, engine, segmented_path_for_api)
    print(f"    task_id={task_id}")

    st = poll_task(task_id, engine)
    elapsed = time.time() - t0
    status = st.get("status", "unknown")
    print(f"    -> {status} in {elapsed:.0f}s")

    record = {
        "obj_id": obj_id,
        "condition": cond_key,
        "engine": engine,
        "use_sam3": use_sam3,
        "task_id": task_id,
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "generation_time_s": st.get("generation_time_seconds"),
        "error": st.get("error"),
    }
    if status == "completed":
        try:
            out = copy_glb_to_output(task_id, obj_id, cond_key)
            record["output"] = str(out)
            print(f"    glb -> {out}")
        except Exception as e:
            record["status"] = "completed_but_no_glb"
            record["error"] = str(e)
            print(f"    [WARN] {e}")

    return record


def load_results() -> list:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results: list):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def index_records(records: list) -> dict:
    return {f"{r['obj_id']}::{r['condition']}": r for r in records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["trellis", "hunyuan", "sam3d"], default=None,
                        help="Run only this engine pass")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these object ids (or id prefixes)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run conditions that already have completed records")
    args = parser.parse_args()

    manifest = load_manifest()
    objects = manifest["objects"]
    conditions = manifest["conditions"]

    if args.only:
        objects = [o for o in objects if any(o["id"].startswith(a) or o["id"] == a for a in args.only)]
    if args.engine:
        conditions = [c for c in conditions if c["engine"] == args.engine]

    results = [] if args.no_skip else load_results()
    results_index = index_records(results)

    print(f"Engine pass(es): {[c['engine'] for c in conditions]}")
    print(f"Objects: {len(objects)}")
    print(f"Total runs: {len(objects) * len(conditions)}")
    print()

    # Group by engine to minimize swap overhead.
    engines_in_order = []
    for c in conditions:
        if c["engine"] not in engines_in_order:
            engines_in_order.append(c["engine"])

    for engine in engines_in_order:
        engine_conds = [c for c in conditions if c["engine"] == engine]
        print(f"=== ENGINE PASS: {engine.upper()} ({len(engine_conds)} condition(s) x {len(objects)} objects) ===")
        for i, obj in enumerate(objects, 1):
            print(f"  [{i}/{len(objects)}] {obj['id']}")
            for cond in engine_conds:
                print(f"    condition: {cond['key']}")
                rec = run_one(obj, cond, results_index)
                key = f"{rec['obj_id']}::{rec['condition']}"
                results_index[key] = rec
                # Replace existing or append
                replaced = False
                for j, r in enumerate(results):
                    if r["obj_id"] == rec["obj_id"] and r["condition"] == rec["condition"]:
                        results[j] = rec
                        replaced = True
                        break
                if not replaced:
                    results.append(rec)
                save_results(results)  # incremental save
        print()

    print("=" * 60)
    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") in ("failed", "timeout", "cancelled"))
    print(f"Total records: {len(results)}")
    print(f"Completed: {completed}")
    print(f"Failed/timeout: {failed}")
    print(f"Results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
