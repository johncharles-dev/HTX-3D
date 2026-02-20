# Machine Requirements

## Minimum Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 8GB+ VRAM | NVIDIA RTX 3080+ / A5000+ |
| GPU Compute | CUDA Compute Capability 8.0+ | 8.6+ (Ampere/Ada/Blackwell) |
| System RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 50 GB+ (for model weights + gallery) |
| NVIDIA Driver | 525+ | 570+ (required for Blackwell GPUs) |

## Software Requirements

### Linux (Native)
- Ubuntu 22.04 or 24.04
- NVIDIA Driver 525+ (570+ for Blackwell)
- CUDA Toolkit 12.x
- Python 3.11 (conda) / 3.12 (Docker)

### Windows (via Docker)
- Windows 10/11 with WSL2
- Docker Desktop with WSL2 backend
- NVIDIA GPU driver for Windows (570+ for Blackwell)
- nvidia-container-toolkit in WSL2

### Docker
- Docker Engine 24+
- docker-compose v2+
- nvidia-container-toolkit (for GPU passthrough)

## Network Requirements

- **First startup**: Internet access to auto-download model weights (~5 GB, saved to persistent volume)
- **Runtime**: No internet required (all models loaded locally)
