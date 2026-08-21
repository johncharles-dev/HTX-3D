# Machine Requirements

Hardware, software and network requirements for the full HTX-3D stack — five engines,
interactive segmentation and metric auto-scaling.

For installation steps see [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 12 GB VRAM | 24 GB+ |
| GPU compute | CUDA compute 8.0+ | 8.6+ (Ampere / Ada / Blackwell) |
| System RAM | 16 GB | 32 GB |
| **Storage** | **150 GB free** | 250 GB+ |
| NVIDIA driver | 525+ | 570+ (**required** for Blackwell / RTX 50 series) |

Peak VRAM measured is ~8.7 GB (TRELLIS.2), so 12 GB is workable. Only the RTX 5090
configuration has been benchmarked.

### Storage breakdown

The stack is far larger than the models alone — the container image dominates.

| Item | Size |
|---|---|
| Docker image | **35.2 GB** |
| HuggingFace cache (SAM 3, DINOv3, UniDepth, CLIP, Hunyuan) | 40 GB |
| Hunyuan3D cache | 14 GB |
| SAM 3D Objects weights | 12 GB |
| TRELLIS weights | 5.3 GB |
| torch hub cache (DINOv2) | 1.4 GB |
| rembg model | 0.2 GB |
| **Subtotal** | **~108 GB** |
| Gallery | **Unbounded** — reached 14 GB over ~800 generations |

Plan for growth: the gallery stores a GLB, preview video and thumbnail per generation.
`GALLERY_HOST_DIR` in `docker/.env` puts it on a separate volume.

Add ~1.2 GB and a second conda environment if the optional
[TRELLIS.2 host service](../services/trellis2/README.md) is deployed.

## Software

### Linux (native or Docker host)
- Ubuntu 22.04 or 24.04
- NVIDIA driver 525+ — **570-open** for Blackwell; the proprietary variant fails
- CUDA Toolkit 12.8 (Blackwell)
- Python 3.11 (native conda) / 3.12 (container)
- Node.js 20+ (frontend build, native setup only)

### Docker
- Docker Engine 24+
- Compose v2+
- `nvidia-container-toolkit` for GPU passthrough

### Windows
- Windows 10/11 with WSL2
- NVIDIA driver installed on **Windows**, not inside the distro (570+ for Blackwell)
- `nvidia-container-toolkit` inside WSL2

Keep the project on the WSL2 filesystem, not `/mnt/c/` — model loading across the bridge is
dramatically slower.

## Network

**During setup**, internet access is required for substantially more than model weights:

| What | Notes |
|---|---|
| CUDA base image | ~5 GB from Docker Hub |
| Python packages | PyPI, plus git clones from GitHub during the build |
| CUDA extensions | nvdiffrast, diffoctreerast, mip-splatting, pytorch3d — cloned and compiled |
| Model weights | ~73 GB across HuggingFace and one GitHub release asset |

**Weights do not all download automatically.** Four models — SAM 3, SAM 3.1, SAM 3D
Objects and DINOv3 — are **gated** and require a HuggingFace account that has accepted each
model's terms. Access is granted per account and does not transfer with copied files.

```bash
huggingface-cli login
python scripts/download_models.py --model all
```

**At runtime**, no internet is required. All models load from local disk, and the container
reaches TRELLIS.2 only over the host gateway if that service is deployed.

### Air-gapped or restricted networks

Weights can be transferred physically. The HuggingFace cache uses symlinks from
`snapshots/` into `blobs/`, so it must be moved as tar archives and extracted onto a Linux
filesystem — **NTFS and exFAT cannot represent those symlinks**.

The gated licences still apply: the receiving organisation needs its own accepted terms
regardless of how the bytes arrive.

## Licence constraints that affect deployment

| Component | Constraint |
|---|---|
| **UniDepth** (auto-scale) | **CC BY-NC 4.0 — non-commercial.** Blocks commercial use of metric scaling |
| **Hunyuan3D 2.1** | Excludes the **EU, UK and South Korea** — model *and output*. Singapore is inside the Territory |
| **SAM 3 / SAM 3D** | Trade controls: no military/warfare, nuclear or espionage use |

Full audit: [reference/vendoring.md](reference/vendoring.md).
