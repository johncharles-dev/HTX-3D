"""
HTX-3D Evaluation Script v2
Compares nvdiffrast-rendered views of generated 3D models against input images.

Metrics:
  Image Quality (rendered output vs input):
  - SSIM: Structural similarity (higher = better, max 1.0)
  - PSNR: Peak signal-to-noise ratio in dB (higher = better)
  - LPIPS: Learned perceptual similarity (lower = better, 0 = identical)
  - CLIP Score: Semantic similarity (higher = better, max 1.0)

  Mesh Quality (intrinsic):
  - Vertex/face count, watertight, surface area, file size, degenerate faces
"""

import os
import json
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import trimesh

# ── Config ──────────────────────────────────────────────────

GALLERY_DIR = "/app/gallery"
INPUT_DIR = "/app/evaluation/input_images/Sample_images_htx"
RENDER_DIR = "/app/evaluation/renders"
RESULTS_FILE = "/app/evaluation/generation_results.json"
OUTPUT_FILE = "/app/evaluation/evaluation_results.json"
REPORT_FILE = "/app/evaluation/evaluation_report.txt"

RENDER_SIZE = 512


# ── Image Metrics ───────────────────────────────────────────

def load_and_resize(path, size=RENDER_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def compute_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, data_range=255)


def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=255)


def compute_lpips(img1, img2, lpips_model):
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
    inputs = clip_processor(
        images=[Image.fromarray(img1), Image.fromarray(img2)],
        return_tensors="pt",
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return (feats[0] @ feats[1]).item()


# ── Mesh Metrics ────────────────────────────────────────────

def compute_mesh_metrics(glb_path):
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

    try:
        areas = mesh.area_faces
        metrics["degenerate_faces"] = int(np.sum(areas < 1e-10))
        metrics["mean_face_area"] = round(float(np.mean(areas)), 8)
    except Exception:
        pass

    return metrics


# ── Main ────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HTX-3D EVALUATION v2 (using nvdiffrast renders)")
    print("=" * 70)

    with open(RESULTS_FILE) as f:
        generations = json.load(f)

    completed = [g for g in generations if g["status"] == "completed"]
    print(f"Evaluating {len(completed)} completed generations\n")

    # Load evaluation models
    print("Loading LPIPS model (AlexNet)...")
    import lpips
    lpips_model = lpips.LPIPS(net="alex").cuda().eval()

    print("Loading CLIP model (ViT-B/32)...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Models loaded.\n")

    results = []

    for gen in completed:
        img_name = gen["image"]
        engine = gen["engine"]
        task_id = gen["task_id"]
        gen_time = gen.get("generation_time", None)

        print(f"[{engine}] {img_name} (task={task_id})")

        # Paths
        input_path = os.path.join(INPUT_DIR, img_name)
        render_name = f"{os.path.splitext(img_name)[0]}_{engine}.png"
        render_path = os.path.join(RENDER_DIR, render_name)
        glb_path = os.path.join(GALLERY_DIR, task_id, "model.glb")

        if not os.path.exists(input_path):
            print(f"  SKIP: input not found at {input_path}")
            continue
        if not os.path.exists(render_path):
            print(f"  SKIP: render not found at {render_path}")
            continue

        # Load images
        input_img = load_and_resize(input_path)
        render_img = load_and_resize(render_path)

        # Image metrics
        ssim_val = compute_ssim(input_img, render_img)
        psnr_val = compute_psnr(input_img, render_img)
        lpips_val = compute_lpips(input_img, render_img, lpips_model)
        clip_val = compute_clip_score(input_img, render_img, clip_model, clip_processor)

        print(f"  SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}dB  LPIPS={lpips_val:.4f}  CLIP={clip_val:.4f}")

        # Mesh metrics
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

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")

    generate_report(results)


def generate_report(results):
    lines = []
    lines.append("=" * 70)
    lines.append("HTX-3D EVALUATION REPORT")
    lines.append("GPU: NVIDIA RTX 5090 (32GB, Blackwell, compute 12.0)")
    lines.append("Rendering: nvdiffrast (front-view, 512x512)")
    lines.append("Image Metrics: input image vs rendered 3D model output")
    lines.append("=" * 70)
    lines.append("")

    # Group by image
    images = {}
    for r in results:
        img = r["image"]
        if img not in images:
            images[img] = {}
        images[img][r["engine"]] = r

    for img_name, engines in images.items():
        lines.append(f"Image: {img_name}")
        lines.append("-" * 60)
        lines.append(f"{'Metric':<25} {'TRELLIS':>15} {'Hunyuan':>15}")
        lines.append("-" * 60)

        t = engines.get("trellis", {})
        h = engines.get("hunyuan", {})
        ti = t.get("image_metrics", {})
        hi = h.get("image_metrics", {})
        tm = t.get("mesh_metrics", {})
        hm = h.get("mesh_metrics", {})

        rows = [
            ("Generation Time (s)", t.get("generation_time_seconds", "N/A"), h.get("generation_time_seconds", "N/A")),
            ("SSIM ↑", ti.get("ssim", "N/A"), hi.get("ssim", "N/A")),
            ("PSNR (dB) ↑", ti.get("psnr", "N/A"), hi.get("psnr", "N/A")),
            ("LPIPS ↓", ti.get("lpips", "N/A"), hi.get("lpips", "N/A")),
            ("CLIP Score ↑", ti.get("clip_score", "N/A"), hi.get("clip_score", "N/A")),
            ("Vertices", tm.get("vertices", "N/A"), hm.get("vertices", "N/A")),
            ("Faces", tm.get("faces", "N/A"), hm.get("faces", "N/A")),
            ("Watertight", str(tm.get("watertight", "N/A")), str(hm.get("watertight", "N/A"))),
            ("Surface Area", tm.get("surface_area", "N/A"), hm.get("surface_area", "N/A")),
            ("File Size (MB)", tm.get("file_size_mb", "N/A"), hm.get("file_size_mb", "N/A")),
            ("Degenerate Faces", tm.get("degenerate_faces", "N/A"), hm.get("degenerate_faces", "N/A")),
        ]
        for label, tv, hv in rows:
            lines.append(f"{label:<25} {str(tv):>15} {str(hv):>15}")
        lines.append("")

    # Averages
    lines.append("=" * 70)
    lines.append("AVERAGE METRICS ACROSS ALL IMAGES")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<25} {'TRELLIS':>15} {'Hunyuan':>15} {'Winner':>12}")
    lines.append("-" * 70)

    metrics_config = [
        ("Generation Time (s)", "generation_time_seconds", False, True),
        ("SSIM ↑", "ssim", True, False),
        ("PSNR (dB) ↑", "psnr", True, False),
        ("LPIPS ↓", "lpips", False, False),
        ("CLIP Score ↑", "clip_score", True, False),
        ("Vertices", "vertices", None, False),
        ("Faces", "faces", None, False),
        ("File Size (MB)", "file_size_mb", None, False),
    ]

    for label, key, higher_better, is_top_level in metrics_config:
        vals = {}
        for engine in ["trellis", "hunyuan"]:
            engine_results = [r for r in results if r["engine"] == engine]
            if is_top_level:
                v = [r.get(key) for r in engine_results if r.get(key) is not None]
            elif key in ("vertices", "faces", "file_size_mb"):
                v = [r.get("mesh_metrics", {}).get(key) for r in engine_results
                     if r.get("mesh_metrics", {}).get(key) is not None]
            else:
                v = [r.get("image_metrics", {}).get(key) for r in engine_results
                     if r.get("image_metrics", {}).get(key) is not None]
            vals[engine] = round(sum(v) / len(v), 4) if v else "N/A"

        tv = vals["trellis"]
        hv = vals["hunyuan"]

        winner = ""
        if higher_better is not None and isinstance(tv, (int, float)) and isinstance(hv, (int, float)):
            if higher_better:
                winner = "TRELLIS" if tv > hv else "Hunyuan"
            else:
                winner = "TRELLIS" if tv < hv else "Hunyuan"

        lines.append(f"{label:<25} {str(tv):>15} {str(hv):>15} {winner:>12}")

    lines.append("")
    lines.append("Notes:")
    lines.append("  ↑ = higher is better, ↓ = lower is better")
    lines.append("  SSIM: Structural Similarity Index (0-1)")
    lines.append("  PSNR: Peak Signal-to-Noise Ratio in dB")
    lines.append("  LPIPS: Learned Perceptual Image Patch Similarity (AlexNet)")
    lines.append("  CLIP: Cosine similarity of CLIP ViT-B/32 image embeddings")

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
