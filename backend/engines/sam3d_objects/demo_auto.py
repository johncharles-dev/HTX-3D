# Copyright (c) Meta Platforms, Inc. and affiliates.
# Automated 3D reconstruction: SAM auto-segments objects, SAM3D reconstructs each one.

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sys.path.append("notebook")
from inference import Inference, load_image

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"


def download_sam_checkpoint():
    if os.path.exists(SAM_CHECKPOINT_PATH):
        return
    print(f"Downloading SAM checkpoint to {SAM_CHECKPOINT_PATH}...")
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)
    torch.hub.download_url_to_file(SAM_CHECKPOINT_URL, SAM_CHECKPOINT_PATH)
    print("Download complete.")


def generate_masks(image_rgb, min_area_ratio=0.005, max_area_ratio=0.5):
    """Use SAM to auto-segment all objects in the image."""
    download_sam_checkpoint()

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to("cuda")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=1000,
    )

    masks = mask_generator.generate(image_rgb)

    # Filter by area ratio
    h, w = image_rgb.shape[:2]
    total_pixels = h * w
    filtered = []
    for m in masks:
        area_ratio = m["area"] / total_pixels
        if min_area_ratio <= area_ratio <= max_area_ratio:
            filtered.append(m)

    # Sort by area descending (largest objects first)
    filtered.sort(key=lambda m: m["area"], reverse=True)

    print(f"SAM found {len(masks)} masks, {len(filtered)} after area filtering.")

    # Free SAM model memory
    del sam, mask_generator
    torch.cuda.empty_cache()

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Auto-segment and reconstruct 3D objects")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o", "--output-dir", default="output_auto", help="Output directory")
    parser.add_argument("-n", "--max-objects", type=int, default=5, help="Max objects to reconstruct")
    parser.add_argument("--min-area", type=float, default=0.005, help="Min mask area ratio (0-1)")
    parser.add_argument("--max-area", type=float, default=0.5, help="Max mask area ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-masks", action="store_true", help="Save mask PNGs for inspection")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load image
    image = load_image(args.image)
    print(f"Loaded image: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Auto-segment
    masks = generate_masks(image, args.min_area, args.max_area)

    if not masks:
        print("No objects found. Try lowering --min-area.")
        return

    n = min(args.max_objects, len(masks))
    print(f"Reconstructing top {n} objects...")

    # Load SAM3D model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # Reconstruct each object
    for i, mask_data in enumerate(masks[:n]):
        mask = mask_data["segmentation"].astype(bool)
        area_pct = mask_data["area"] / (image.shape[0] * image.shape[1]) * 100

        print(f"\n[{i+1}/{n}] Reconstructing object (area: {area_pct:.1f}% of image)...")

        if args.save_masks:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(os.path.join(args.output_dir, f"mask_{i}.png"))

        output = inference(image, mask, seed=args.seed)
        output["gs"].save_ply(os.path.join(args.output_dir, f"object_{i}.ply"))
        print(f"  Saved: {args.output_dir}/object_{i}.ply")

    print(f"\nDone! {n} objects reconstructed in {args.output_dir}/")


if __name__ == "__main__":
    main()
