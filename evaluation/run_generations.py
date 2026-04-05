"""Submit all evaluation images to both TRELLIS and Hunyuan engines, wait for results."""

import os
import sys
import time
import json
import requests

API_BASE = "http://localhost:8000/api"
IMAGE_DIR = "/home/cj/HTX-3D/evaluation/input_images/Sample_images_htx"

# Only use convertible image formats (skip .avif)
SKIP_EXT = {".avif"}

ENGINES = ["trellis", "hunyuan"]
EXPORT_FORMATS = "glb,obj,stl"

POLL_INTERVAL = 5  # seconds
TIMEOUT = 600  # 10 minutes max per task


def get_images():
    images = []
    for f in sorted(os.listdir(IMAGE_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in SKIP_EXT:
            continue
        if ext in (".jpg", ".jpeg", ".png", ".webp"):
            images.append(os.path.join(IMAGE_DIR, f))
    return images


def submit_task(image_path, engine):
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        data = {
            "engine": engine,
            "seed": 42,
            "randomize_seed": "false",  # fixed seed for reproducibility
            "formats": EXPORT_FORMATS,
            "texture_size": 1024,
        }
        resp = requests.post(f"{API_BASE}/generate/image", files=files, data=data)
    resp.raise_for_status()
    result = resp.json()
    return result["task_id"]


def poll_task(task_id):
    start = time.time()
    while time.time() - start < TIMEOUT:
        resp = requests.get(f"{API_BASE}/task/{task_id}")
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status", "unknown")

        if status == "completed":
            return result
        elif status == "failed":
            print(f"  FAILED: {result.get('error', 'unknown error')}")
            return result

        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] status={status}...", flush=True)
        time.sleep(POLL_INTERVAL)

    print(f"  TIMEOUT after {TIMEOUT}s")
    return None


def main():
    images = get_images()
    print(f"Found {len(images)} images:")
    for img in images:
        print(f"  - {os.path.basename(img)}")
    print()

    results = []

    for engine in ENGINES:
        print(f"{'='*60}")
        print(f"ENGINE: {engine.upper()}")
        print(f"{'='*60}")

        for img_path in images:
            img_name = os.path.basename(img_path)
            print(f"\n[{engine}] Submitting {img_name}...")

            task_id = submit_task(img_path, engine)
            print(f"  Task ID: {task_id}")

            result = poll_task(task_id)

            if result and result.get("status") == "completed":
                gen_time = result.get("generation_time_seconds", "N/A")
                print(f"  COMPLETED in {gen_time}s")
                results.append({
                    "image": img_name,
                    "engine": engine,
                    "task_id": task_id,
                    "status": "completed",
                    "generation_time": gen_time,
                    "gallery_id": task_id,
                })
            else:
                print(f"  FAILED or TIMED OUT")
                results.append({
                    "image": img_name,
                    "engine": engine,
                    "task_id": task_id,
                    "status": "failed",
                })

    # Save results
    output_path = "/home/cj/HTX-3D/evaluation/generation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Total: {len(results)} generations ({sum(1 for r in results if r['status']=='completed')} completed)")


if __name__ == "__main__":
    main()
