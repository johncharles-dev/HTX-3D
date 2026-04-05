"""
HTX-3D Evaluation Script
Computes quality metrics comparing generated 3D models against input images.

Metrics:
  - SSIM: Structural similarity (higher = better, max 1.0)
  - PSNR: Peak signal-to-noise ratio in dB (higher = better)
  - LPIPS: Learned perceptual similarity (lower = better, 0 = identical)
  - CLIP Score: Semantic similarity (higher = better, max 1.0)
  - Mesh quality: vertex/face count, watertight, bounding box, file size
"""

import os
import sys
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import trimesh

# ── Config ──────────────────────────────────────────────────

GALLERY_DIR = "/app/gallery"
INPUT_DIR = "/app/evaluation/input_images/Sample_images_htx"
RESULTS_FILE = "/app/evaluation/generation_results.json"
OUTPUT_FILE = "/app/evaluation/evaluation_results.json"
REPORT_FILE = "/app/evaluation/evaluation_report.txt"

RENDER_SIZE = 512  # resize both images to this for comparison


# ── Image Metrics ───────────────────────────────────────────

def load_and_resize(path, size=RENDER_SIZE):
    """Load image, convert to RGB, resize to square."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def compute_ssim(img1, img2):
    """SSIM between two RGB images (numpy arrays)."""
    return ssim(img1, img2, channel_axis=2, data_range=255)


def compute_psnr(img1, img2):
    """PSNR between two RGB images."""
    return psnr(img1, img2, data_range=255)


def compute_lpips(img1, img2, lpips_model):
    """LPIPS perceptual distance (lower = more similar)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    t1 = transform(img1).unsqueeze(0).cuda()
    t2 = transform(img2).unsqueeze(0).cuda()
    with torch.no_grad():
        dist = lpips_model(t1, t2)
    return dist.item()


def compute_clip_score(img1, img2, clip_model, clip_processor):
    """CLIP cosine similarity between two images."""
    inputs = clip_processor(images=[Image.fromarray(img1), Image.fromarray(img2)],
                           return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    similarity = (feats[0] @ feats[1]).item()
    return similarity


# ── Mesh Metrics ────────────────────────────────────────────

def compute_mesh_metrics(glb_path):
    """Extract mesh quality metrics from a GLB file."""
    try:
        mesh = trimesh.load(glb_path, force="mesh")
    except Exception as e:
        return {"error": str(e)}

    bounds = mesh.bounds
    bbox_dims = bounds[1] - bounds[0]

    metrics = {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "surface_area": round(float(mesh.area), 4),
        "volume": round(float(mesh.volume), 6) if mesh.is_watertight else None,
        "bbox_x": round(float(bbox_dims[0]), 4),
        "bbox_y": round(float(bbox_dims[1]), 4),
        "bbox_z": round(float(bbox_dims[2]), 4),
        "euler_number": int(mesh.euler_number),
        "file_size_mb": round(os.path.getsize(glb_path) / (1024 * 1024), 2),
    }

    # Check for degenerate faces
    try:
        areas = mesh.area_faces
        metrics["degenerate_faces"] = int(np.sum(areas < 1e-10))
        metrics["min_face_area"] = float(np.min(areas))
        metrics["mean_face_area"] = float(np.mean(areas))
    except Exception:
        pass

    return metrics


# ── Main ────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HTX-3D EVALUATION")
    print("=" * 70)

    # Load generation results
    with open(RESULTS_FILE) as f:
        generations = json.load(f)

    completed = [g for g in generations if g["status"] == "completed"]
    print(f"Evaluating {len(completed)} completed generations\n")

    # Load models (once)
    print("Loading LPIPS model...")
    import lpips
    lpips_model = lpips.LPIPS(net="alex").cuda()
    lpips_model.eval()

    print("Loading CLIP model...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("Models loaded.\n")

    results = []

    for gen in completed:
        img_name = gen["image"]
        engine = gen["engine"]
        task_id = gen["task_id"]
        gen_time = gen.get("generation_time", None)

        print(f"[{engine}] {img_name} (task={task_id})")

        input_path = os.path.join(INPUT_DIR, img_name)
        thumbnail_path = os.path.join(GALLERY_DIR, task_id, "thumbnail.png")
        glb_path = os.path.join(GALLERY_DIR, task_id, "model.glb")

        if not os.path.exists(input_path):
            print(f"  SKIP: input image not found")
            continue
        if not os.path.exists(thumbnail_path):
            print(f"  SKIP: thumbnail not found")
            continue

        # Load images
        input_img = load_and_resize(input_path)
        thumb_img = load_and_resize(thumbnail_path)

        # Image quality metrics
        ssim_val = compute_ssim(input_img, thumb_img)
        psnr_val = compute_psnr(input_img, thumb_img)
        lpips_val = compute_lpips(input_img, thumb_img, lpips_model)
        clip_val = compute_clip_score(input_img, thumb_img, clip_model, clip_processor)

        print(f"  SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}dB  LPIPS={lpips_val:.4f}  CLIP={clip_val:.4f}")

        # Mesh quality metrics
        mesh_metrics = {}
        if os.path.exists(glb_path):
            mesh_metrics = compute_mesh_metrics(glb_path)
            print(f"  Mesh: {mesh_metrics.get('vertices', '?')} verts, "
                  f"{mesh_metrics.get('faces', '?')} faces, "
                  f"watertight={mesh_metrics.get('watertight', '?')}, "
                  f"size={mesh_metrics.get('file_size_mb', '?')}MB")

        result = {
            "image": img_name,
            "engine": engine,
            "task_id": task_id,
            "generation_time_seconds": gen_time,
            "image_metrics": {
                "ssim": round(ssim_val, 4),
                "psnr": round(psnr_val, 2),
                "lpips": round(lpips_val, 4),
                "clip_score": round(clip_val, 4),
            },
            "mesh_metrics": mesh_metrics,
        }
        results.append(result)
        print()

    # Save JSON results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")

    # Generate summary report
    generate_report(results)


def generate_report(results):
    """Generate a formatted comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("HTX-3D EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Group by image
    images = {}
    for r in results:
        img = r["image"]
        if img not in images:
            images[img] = {}
        images[img][r["engine"]] = r

    # Per-image comparison
    for img_name, engines in images.items():
        lines.append(f"Image: {img_name}")
        lines.append("-" * 50)

        header = f"{'Metric':<25} {'TRELLIS':>12} {'Hunyuan':>12}"
        lines.append(header)
        lines.append("-" * 50)

        trellis = engines.get("trellis", {})
        hunyuan = engines.get("hunyuan", {})

        t_img = trellis.get("image_metrics", {})
        h_img = hunyuan.get("image_metrics", {})

        lines.append(f"{'Generation Time (s)':<25} {trellis.get('generation_time_seconds', 'N/A'):>12} {hunyuan.get('generation_time_seconds', 'N/A'):>12}")
        lines.append(f"{'SSIM ↑':<25} {t_img.get('ssim', 'N/A'):>12} {h_img.get('ssim', 'N/A'):>12}")
        lines.append(f"{'PSNR (dB) ↑':<25} {t_img.get('psnr', 'N/A'):>12} {h_img.get('psnr', 'N/A'):>12}")
        lines.append(f"{'LPIPS ↓':<25} {t_img.get('lpips', 'N/A'):>12} {h_img.get('lpips', 'N/A'):>12}")
        lines.append(f"{'CLIP Score ↑':<25} {t_img.get('clip_score', 'N/A'):>12} {h_img.get('clip_score', 'N/A'):>12}")

        t_mesh = trellis.get("mesh_metrics", {})
        h_mesh = hunyuan.get("mesh_metrics", {})

        lines.append(f"{'Vertices':<25} {t_mesh.get('vertices', 'N/A'):>12} {h_mesh.get('vertices', 'N/A'):>12}")
        lines.append(f"{'Faces':<25} {t_mesh.get('faces', 'N/A'):>12} {h_mesh.get('faces', 'N/A'):>12}")
        lines.append(f"{'Watertight':<25} {str(t_mesh.get('watertight', 'N/A')):>12} {str(h_mesh.get('watertight', 'N/A')):>12}")
        lines.append(f"{'File Size (MB)':<25} {t_mesh.get('file_size_mb', 'N/A'):>12} {h_mesh.get('file_size_mb', 'N/A'):>12}")
        lines.append("")

    # Averages
    lines.append("=" * 70)
    lines.append("AVERAGES ACROSS ALL IMAGES")
    lines.append("=" * 70)
    header = f"{'Metric':<25} {'TRELLIS':>12} {'Hunyuan':>12}"
    lines.append(header)
    lines.append("-" * 50)

    for engine in ["trellis", "hunyuan"]:
        pass  # computed below

    for metric_name, key, higher_better in [
        ("Generation Time (s)", "generation_time_seconds", False),
        ("SSIM ↑", "ssim", True),
        ("PSNR (dB) ↑", "psnr", True),
        ("LPIPS ↓", "lpips", False),
        ("CLIP Score ↑", "clip_score", True),
    ]:
        vals = {}
        for engine in ["trellis", "hunyuan"]:
            engine_results = [r for r in results if r["engine"] == engine]
            if key == "generation_time_seconds":
                v = [r.get(key) for r in engine_results if r.get(key) is not None]
            else:
                v = [r["image_metrics"].get(key) for r in engine_results if r.get("image_metrics", {}).get(key) is not None]
            vals[engine] = round(sum(v) / len(v), 4) if v else "N/A"

        t_val = vals.get("trellis", "N/A")
        h_val = vals.get("hunyuan", "N/A")

        # Mark winner
        winner = ""
        if isinstance(t_val, (int, float)) and isinstance(h_val, (int, float)):
            if higher_better:
                winner = " ← TRELLIS" if t_val > h_val else " ← Hunyuan"
            else:
                winner = " ← TRELLIS" if t_val < h_val else " ← Hunyuan"

        lines.append(f"{metric_name:<25} {t_val:>12} {h_val:>12}{winner}")

    lines.append("")

    report = "\n".join(lines)
    print(report)

    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
