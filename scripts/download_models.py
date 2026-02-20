#!/usr/bin/env python3
"""One-time model weight downloader for HTX 3D Generation Tool.

Downloads TRELLIS model weights from HuggingFace to a local directory.
After downloading, models are loaded from disk — no internet required at runtime.

Usage:
    python scripts/download_models.py                    # Download all models
    python scripts/download_models.py --model image      # Image-to-3D only
    python scripts/download_models.py --model text       # Text-to-3D only
    python scripts/download_models.py --output ./weights  # Custom output dir
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

def download_huggingface_model(repo_id: str, output_dir: str):
    """Download a HuggingFace model repository to a local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub<0.25")
        from huggingface_hub import snapshot_download

    model_name = repo_id.split("/")[-1]
    local_dir = os.path.join(output_dir, model_name)

    if os.path.isdir(local_dir) and any(Path(local_dir).iterdir()):
        print(f"  [SKIP] {model_name} already exists at {local_dir}")
        return local_dir

    print(f"  Downloading {repo_id} -> {local_dir}")
    print(f"  This may take several minutes depending on your connection...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    print(f"  [OK] {model_name} downloaded ({get_dir_size(local_dir)})")
    return local_dir


def get_dir_size(path: str) -> str:
    """Get human-readable directory size."""
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    if total < 1024**2:
        return f"{total / 1024:.1f} KB"
    if total < 1024**3:
        return f"{total / 1024**2:.1f} MB"
    return f"{total / 1024**3:.1f} GB"


MODELS = {
    "image": {
        "repo": "JeffreyXiang/TRELLIS-image-large",
        "description": "TRELLIS Image-to-3D (required)",
    },
    "text": {
        "repo": "JeffreyXiang/TRELLIS-text-large",
        "description": "TRELLIS Text-to-3D (optional, for text prompts)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Download TRELLIS model weights")
    parser.add_argument(
        "--model",
        choices=["image", "text", "all"],
        default="all",
        help="Which model to download (default: all)",
    )
    parser.add_argument(
        "--output",
        default="./weights",
        help="Output directory for model weights (default: ./weights)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("HTX 3D Generation Tool - Model Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    models_to_download = (
        MODELS.keys() if args.model == "all" else [args.model]
    )

    for model_key in models_to_download:
        info = MODELS[model_key]
        print(f"[{model_key.upper()}] {info['description']}")
        download_huggingface_model(info["repo"], output_dir)
        print()

    print("=" * 60)
    print("Download complete!")
    print(f"Models saved to: {output_dir}")
    print()
    print("To use with Docker:")
    print(f"  Mount this directory as a volume: -v {output_dir}:/app/weights")
    print()
    print("To use locally:")
    print(f"  Set WEIGHTS_DIR={output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
