"""Phase 1 — segmentation preparation for the 14-object benchmark.

For each object in manifest.json, this script:
  1. Uploads the image to /api/segment/start
  2. Issues the SAM 3 prompt (text, points, or text+points) declared in the manifest
  3. Confirms the chosen mask_index (default 0) -> RGBA segmented file on disk
  4. Copies the original image and the segmented PNG into benchmark_v2/data/<id>/
  5. Records the mask details in meta.json

For multi-object APICS scenes, edit the manifest entry to use:
    "prompt": {
        "mode": "text+points",
        "text": "kiosk",
        "points": [[1024, 600], [1800, 580]],
        "labels": [1, 0]
    }

Usage:
    python prepare_masks.py                # process all objects
    python prepare_masks.py 12 14          # process only objects with id starting "12_" and "14_"

Requires: backend container running at API_BASE.
"""

import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api"
DOCKER_CONTAINER = os.environ.get("HTX3D_CONTAINER", "htx-3d")
SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "manifest.json"
DATA_DIR = SCRIPT_DIR / "data"


def fetch_from_container(container_path: str, host_dst: Path):
    """Copy a file from inside the running Docker container to the host.

    The backend runs in Docker and SAM 3 writes segmented files to TEMP_DIR
    inside the container, so a plain shutil.copy2 from the host won't see
    them. We use `docker cp` to bridge the gap.
    """
    if Path(container_path).exists():
        # Either we're not running in Docker, or the path is shared via volume.
        shutil.copy2(container_path, host_dst)
        return
    cmd = ["docker", "cp", f"{DOCKER_CONTAINER}:{container_path}", str(host_dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"docker cp failed (container='{DOCKER_CONTAINER}', src='{container_path}'):\n"
            f"  stdout: {r.stdout}\n  stderr: {r.stderr}"
        )


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def filter_objects(objects: list, args: list[str]) -> list:
    """If user passes prefixes, keep only matching objects."""
    if not args:
        return objects
    keep = []
    for obj in objects:
        for arg in args:
            if obj["id"].startswith(arg) or obj["id"] == arg:
                keep.append(obj)
                break
    return keep


def post(path: str, **kwargs) -> dict:
    r = requests.post(f"{API_BASE}{path}", timeout=120, **kwargs)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()


def start_segment_session(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        return post("/segment/start", files=files)


def segment_text(session_id: str, text: str) -> dict:
    return post("/segment/text", json={"session_id": session_id, "prompt": text})


def segment_points(session_id: str, points: list, labels: list) -> dict:
    return post("/segment/points", json={
        "session_id": session_id,
        "points": points,
        "labels": labels,
    })


def confirm_mask(session_id: str, mask_index: int) -> dict:
    return post("/segment/confirm", json={
        "session_id": session_id,
        "mask_index": mask_index,
    })


def reset_prompts(session_id: str):
    requests.post(f"{API_BASE}/segment/reset", params={"session_id": session_id}, timeout=30)


def process_object(obj: dict, force: bool = False) -> dict:
    obj_id = obj["id"]
    image_path = obj["image"]
    prompt = obj["prompt"]
    mask_index = obj.get("mask_index", 0)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir = DATA_DIR / obj_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy original
    original_dst = out_dir / f"original{os.path.splitext(image_path)[1].lower()}"
    if not original_dst.exists():
        shutil.copy2(image_path, original_dst)

    # 1b. Manual-override path:
    # If a segmented.png already exists in this object's folder (placed there by
    # the operator using the UI for tricky cases), skip the API segmentation and
    # just record a "manual" meta. Pass --force to override.
    seg_dst = out_dir / "segmented.png"
    if seg_dst.exists() and not force:
        print(f"  [manual] segmented.png already present, skipping API segmentation")
        meta = {
            "id": obj_id,
            "image": image_path,
            "original_copy": str(original_dst),
            "segmented_path": str(seg_dst),
            "category": obj.get("category"),
            "target": obj.get("target"),
            "prompt": {"mode": "manual"},
            "manual_override": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_path = out_dir / "meta.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        return meta

    # 2. If prompt mode is "manual", expect the operator to provide segmented.png
    #    via the UI. Tell them what to do and exit.
    if prompt.get("mode") == "manual":
        seg_dst = out_dir / "segmented.png"
        if seg_dst.exists():
            # Operator already placed the file — record manual meta and finish.
            print(f"  [manual] segmented.png present, recording meta")
            meta = {
                "id": obj_id,
                "image": image_path,
                "original_copy": str(original_dst),
                "segmented_path": str(seg_dst),
                "category": obj.get("category"),
                "target": obj.get("target"),
                "prompt": prompt,
                "manual_override": True,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            return meta
        else:
            raise RuntimeError(
                f"{obj_id} declared mode='manual' but {seg_dst} doesn't exist yet. "
                f"Open {image_path} in the SegmentationWorkspace UI, isolate the target "
                f"({obj.get('target')}) using point/box/text prompts, click 'Use Mask', "
                f"then docker cp the result into {seg_dst} and re-run this script."
            )

    # 3. Start session
    print(f"  [start] uploading image...")
    sess = start_segment_session(image_path)
    session_id = sess["session_id"]
    print(f"  [start] session={session_id} ({sess['width']}x{sess['height']})")

    # 4. Run prompt(s)
    mode = prompt["mode"]
    masks_info = None

    if mode == "text":
        print(f"  [text] prompt='{prompt['text']}'")
        result = segment_text(session_id, prompt["text"])
        masks_info = result["masks"]
    elif mode == "points":
        print(f"  [points] {len(prompt['points'])} points")
        result = segment_points(session_id, prompt["points"], prompt["labels"])
        masks_info = result["masks"]
    elif mode == "text+points":
        print(f"  [text] prompt='{prompt['text']}'")
        segment_text(session_id, prompt["text"])
        print(f"  [points] refining with {len(prompt['points'])} points")
        result = segment_points(session_id, prompt["points"], prompt["labels"])
        masks_info = result["masks"]
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    if not masks_info:
        raise RuntimeError(f"SAM3 returned 0 masks for {obj_id}. Adjust prompt and retry.")

    print(f"  [masks] {len(masks_info)} returned, picking index={mask_index}")

    # 4. Confirm
    if mask_index >= len(masks_info):
        raise RuntimeError(f"mask_index={mask_index} but only {len(masks_info)} masks returned")
    conf = confirm_mask(session_id, mask_index)
    segmented_path = conf["segmented_image_path"]

    # 5. Copy segmented file into our folder (uses docker cp if needed)
    seg_dst = out_dir / "segmented.png"
    fetch_from_container(segmented_path, seg_dst)
    print(f"  [done] segmented saved to {seg_dst}")

    meta = {
        "id": obj_id,
        "image": image_path,
        "original_copy": str(original_dst),
        "segmented_path": str(seg_dst),
        "remote_segmented_path": segmented_path,
        "category": obj.get("category"),
        "target": obj.get("target"),
        "prompt": prompt,
        "mask_index": mask_index,
        "masks_returned": len(masks_info),
        "selected_mask": masks_info[mask_index],
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    manifest = load_manifest()
    objects = filter_objects(manifest["objects"], args)

    print(f"Processing {len(objects)} object(s)" + (" (force re-segment)" if force else ""))
    print(f"Output directory: {DATA_DIR}")
    print()

    success = []
    failed = []
    for i, obj in enumerate(objects, 1):
        print(f"[{i}/{len(objects)}] {obj['id']}")
        try:
            meta = process_object(obj, force=force)
            success.append(meta)
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed.append({"id": obj["id"], "error": str(e)})
        print()

    print("=" * 60)
    print(f"Success: {len(success)} / {len(objects)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f['id']}: {f['error']}")
    print(f"Outputs in: {DATA_DIR}")


if __name__ == "__main__":
    main()
