# HTX 3D Generation Tool - Setup & Deployment Guide

Complete guide for setting up and running HTX 3D Generation Tool on Linux, Windows (WSL2), and Docker across different NVIDIA GPU generations.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Setup on Linux (Native)](#2-setup-on-linux-native)
3. [Setup on Windows (WSL2)](#3-setup-on-windows-wsl2)
4. [Setup with Docker](#4-setup-with-docker)
5. [Model Weights Download](#5-model-weights-download)

---

## 1. System Requirements

### Minimum Hardware

| Component      | Minimum              | Recommended            |
|----------------|----------------------|------------------------|
| GPU            | NVIDIA, 8 GB+ VRAM   | NVIDIA RTX 3090 / 4080+ |
| GPU Compute    | CUDA Compute 8.0+    | 8.6+ (Ampere/Ada/Blackwell) |
| System RAM     | 16 GB                | 32 GB                  |
| Storage        | 25 GB free           | 50 GB+ (weights + gallery) |
| NVIDIA Driver  | 525+                 | 570+ (required for RTX 50 series) |

### Software Stack

| Component        | Version               |
|------------------|-----------------------|
| Python           | 3.11 (conda) / 3.12 (Docker) |
| Node.js          | 20+                   |
| CUDA Toolkit     | 12.x (12.8 for Blackwell) |
| PyTorch          | 2.7.0+cu128           |
| OS               | Ubuntu 22.04/24.04 or Windows 10/11 via WSL2 |

---

## 2. Setup on Linux (Native)

### 2.1 Prerequisites

#### Install NVIDIA Driver

**For Ampere/Ada (RTX 30/40 series):**
```bash
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

**For Blackwell (RTX 50 series) -- MUST use open kernel modules:**
```bash
sudo apt update
sudo apt install nvidia-driver-570-open
sudo reboot
```

> The proprietary driver (`nvidia-driver-570` without `-open`) fails on Blackwell. Always use the `-open` variant.

Verify the driver is working:
```bash
nvidia-smi
```

#### Install CUDA Toolkit 12.8

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8
```

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
nvcc --version   # Should show 12.8
```

#### Install Conda (Miniconda)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, restart shell
```

#### Install Node.js 20

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

Or via nvm:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install 20
nvm use 20
```

#### System Libraries

```bash
sudo apt install -y git wget curl build-essential \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

### 2.2 Get the Project

```bash
# If the project is hosted on a git remote:
git clone <repo-url> HTX-3D
cd HTX-3D

# If copying from an existing machine:
# Copy the entire HTX-3D/ directory (excluding weights/ and gallery/)
rsync -av --exclude='weights/' --exclude='gallery/' --exclude='node_modules/' \
  --exclude='frontend/dist/' --exclude='__pycache__/' \
  user@source:/path/to/HTX-3D/ ./HTX-3D/
cd HTX-3D
```

### 2.3 Create Conda Environment

```bash
conda create -n htx-3d python=3.11 -y
conda activate htx-3d
```

### 2.4 Install PyTorch with CUDA 12.8

```bash
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU access:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2.5 Install Backend Dependencies

```bash
# Core backend
pip install -r backend/requirements.txt

# TRELLIS engine dependencies (selectively -- some need special handling)
pip install pillow imageio imageio-ffmpeg opencv-python-headless
pip install trimesh xatlas pyvista pymeshfix open3d
pip install rembg onnxruntime-gpu
pip install tqdm easydict scipy ninja igraph
pip install "huggingface_hub>=0.23,<0.25" "transformers>=4.35.0,<4.50" "pydantic>=2.0,<2.10"
pip install utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# spconv (no cu128 build exists, cu126 works via forward compat)
pip install spconv-cu126==2.3.8

# kaolin (no cu128 wheel, use cu126 index)
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html

# flash-attn is NOT needed when using SDPA backend (which we recommend for all GPUs).
# Do NOT install flash-attn unless you have a specific reason -- it has no pip wheels
# and crashes on Blackwell GPUs.
```

### 2.6 Build CUDA Extensions

CUDA extensions MUST be compiled with the correct architecture list. This is critical -- PTX fallback produces garbage results on newer GPUs (telltale sign: 66 billion GB OOM).

**Option A: Use the TRELLIS setup script (recommended)**

The TRELLIS engine includes a `setup.sh` script that handles cloning, building, and installing all extensions:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0;12.0"
export CUDA_HOME=/usr/local/cuda-12.8

cd backend/engines/trellis
bash setup.sh --all
cd ../../..
```

This installs: nvdiffrast, diffoctreerast, diff-gaussian-rasterization, spconv, kaolin, and all other dependencies in one step.

**Option B: Build each extension manually**

If you prefer to install step by step:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0;12.0"
export CUDA_HOME=/usr/local/cuda-12.8

# 1. nvdiffrast (differentiable rasterization)
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
pip install --no-build-isolation /tmp/nvdiffrast

# 2. diffoctreerast (octree rasterization)
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast
pip install --no-build-isolation /tmp/diffoctreerast

# 3. diff-gaussian-rasterization (Gaussian splatting renderer)
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting
pip install --no-build-isolation /tmp/mip-splatting/submodules/diff-gaussian-rasterization/
```

> **Verify the build succeeded:**
> ```bash
> python -c "import nvdiffrast; import diffoctreerast; import diff_gaussian_rasterization; print('All CUDA extensions OK')"
> ```

### 2.7 Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 2.8 Download Model Weights (Optional)

Model weights are auto-downloaded on first startup. To pre-download:

```bash
python scripts/download_models.py
# Downloads ~5 GB to ./weights/
```

See [Section 5](#5-model-weights-download) for details and options.

### 2.9 Run the Application

**Backend (Terminal 1):**
```bash
conda activate htx-3d
XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa \
  TRELLIS_ENGINE_DIR=./backend/engines/trellis \
  WEIGHTS_DIR=./weights \
  GALLERY_DIR=./gallery \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

**Frontend dev server (Terminal 2):**
```bash
cd frontend
npm run dev
# Opens on http://localhost:5173 (proxies API to :8000)
```

Or build frontend for production and serve everything from the backend:
```bash
cd frontend && npm run build && cd ..
# Now http://localhost:8000 serves both API and frontend
```

---

## 3. Setup on Windows (WSL2)

Windows does not natively support CUDA Linux containers. You have two options:

- **Option A: Docker Desktop + WSL2** (recommended, simpler) -- see [Section 4](#4-setup-with-docker)
- **Option B: Native in WSL2** (more control, no Docker overhead)

### Option B: Native Setup in WSL2

#### 3.1 Install WSL2

Open PowerShell as Administrator:
```powershell
wsl --install
# Reboot if prompted, then:
wsl --set-default-version 2
```

Install Ubuntu 24.04 from the Microsoft Store, then launch it.

#### 3.2 Install NVIDIA GPU Driver (Windows-side only)

Download and install the latest NVIDIA driver **for Windows** from https://www.nvidia.com/drivers

- For RTX 50 series: version 570+
- For RTX 30/40 series: version 525+

> Do NOT install Linux NVIDIA drivers inside WSL2. The Windows driver provides GPU access automatically.

Verify inside WSL2:
```bash
nvidia-smi
```

#### 3.3 Install CUDA Toolkit in WSL2

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8
```

> Use the **wsl-ubuntu** repo, not the regular ubuntu repo.

Add to `~/.bashrc`:
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 3.4 Continue with Linux Setup

From here, follow [Section 2](#2-setup-on-linux-native) starting at step 2.2 (Get the Project). Everything works identically inside WSL2.

#### 3.5 WSL2-Specific Tips

- **File performance**: Clone the repo inside the WSL2 filesystem (`~/HTX-3D`), NOT on `/mnt/c/`. The Windows filesystem is very slow from WSL2.
- **Memory limit**: WSL2 defaults to 50% of system RAM. Create `%UserProfile%\.wslconfig` to increase:
  ```ini
  [wsl2]
  memory=24GB
  swap=8GB
  ```
- **Access from Windows browser**: The app at `http://localhost:5173` (dev) or `http://localhost:8000` (prod) is automatically accessible from Windows.
- **Display server**: Use X11 if you need GUI. Wayland crashes with RTX 50 series GPUs.

---

## 4. Setup with Docker

Docker is the simplest deployment method. It bundles all dependencies (CUDA, Python, Node.js, CUDA extensions) into a single container.

### 4.1 Prerequisites

#### Linux

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in

# Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

#### Windows (Docker Desktop)

1. Install the latest **NVIDIA GPU driver for Windows** (570+ for RTX 50 series)
2. Install **Docker Desktop** from https://docker.com/products/docker-desktop
   - Check "Use WSL 2 based engine" during install
3. In Docker Desktop: Settings > Resources > WSL Integration > Enable for your distro
4. Install `nvidia-container-toolkit` inside WSL2:
   ```bash
   wsl
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   ```
5. **Restart Docker Desktop** after installing the toolkit

### 4.2 Clone and Run

```bash
# Clone or copy the project (see Section 2.2 for options)
cd HTX-3D
```

### 4.3 Build and Run

```bash
cd docker
docker compose up --build
```

**First build takes 20-40 minutes** (downloads CUDA base image ~5 GB, clones and compiles 3 CUDA extensions from GitHub, installs all dependencies). Model weights (~5 GB) are automatically downloaded on first startup and saved to the persistent `weights/` volume -- no manual download step needed. Subsequent starts are fast.

Once running, open: **http://localhost:8000**

### 4.4 Docker Compose Details

The `docker/docker-compose.yml` provides:

| Setting | Value | Purpose |
|---------|-------|---------|
| Port    | `8000:8000` | Backend + frontend served together |
| Volume  | `../weights:/app/weights` | Model weights (persist across rebuilds) |
| Volume  | `../gallery:/app/gallery` | Generated models (persist across rebuilds) |
| GPU     | 1 NVIDIA GPU | Full GPU passthrough |
| Restart | `unless-stopped` | Auto-restart on crash |

### 4.5 Stop and Manage

```bash
# Stop
docker compose down

# Rebuild after code changes
docker compose up --build

# View logs
docker compose logs -f

# Shell into container
docker compose exec HTX-3D bash
```

---

## 5. Model Weights Download

TRELLIS requires pre-trained model weights from HuggingFace (~5 GB total). **With Docker, weights are auto-downloaded on first startup** and saved to the persistent `weights/` volume. No manual steps needed.

For local development or to pre-download weights, use the download script:

### Available Models

| Model | HuggingFace Repo | Size | Purpose |
|-------|-------------------|------|---------|
| Image-to-3D | `JeffreyXiang/TRELLIS-image-large` | ~3 GB | Generate 3D from images |
| Text-to-3D  | `JeffreyXiang/TRELLIS-text-large`  | ~2 GB | Generate 3D from text prompts |

### Download Script (Optional)

```bash
# Pre-download all models
python scripts/download_models.py

# Download only image-to-3D model
python scripts/download_models.py --model image

# Download only text-to-3D model
python scripts/download_models.py --model text

# Custom output directory
python scripts/download_models.py --output /path/to/weights
```

The script:
- Skips models already downloaded
- Installs `huggingface_hub` automatically if missing
- Shows download progress and final size
- Requires `huggingface_hub<0.25` (pinned for compatibility)

### Directory Structure After Download

```
weights/
├── TRELLIS-image-large/    # ~3 GB
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── TRELLIS-text-large/     # ~2 GB
    ├── config.json
    ├── model.safetensors
    └── ...
```

---

## Project Structure

```
HTX-3D/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, WebSocket, health check
│   │   ├── config.py               # Environment config, GPU detection
│   │   ├── dependencies.py         # FastAPI dependency injection
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic request/response schemas
│   │   ├── routers/
│   │   │   ├── generate.py         # /api/generate/* endpoints
│   │   │   └── gallery.py          # /api/gallery/*, /api/download/*
│   │   └── services/
│   │       ├── base.py             # BaseEngine abstract interface
│   │       ├── trellis.py          # TRELLIS engine wrapper
│   │       └── task_manager.py     # Async task queue + progress
│   ├── engines/
│   │   └── trellis/                # TRELLIS source (Blackwell-patched)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Main app (tabs, state, flow)
│   │   ├── api/client.ts           # Backend API client
│   │   ├── types/index.ts          # TypeScript types & constants
│   │   └── components/
│   │       ├── Header.tsx           # Nav tabs, GPU status
│   │       ├── ImageUpload.tsx      # Drag-drop image input
│   │       ├── ModelSelector.tsx    # Model toggle
│   │       ├── SettingsPanel.tsx    # Generation parameters
│   │       ├── ModelViewer.tsx      # Three.js 3D viewer
│   │       ├── ProgressBar.tsx      # Real-time progress
│   │       ├── ExportPanel.tsx      # Format selection & download
│   │       ├── ResultTabs.tsx       # Multi-model results
│   │       └── Gallery.tsx          # Past generations browser
│   ├── package.json
│   └── vite.config.ts
├── docker/
│   ├── Dockerfile                   # Multi-stage (frontend build + CUDA backend)
│   └── docker-compose.yml
├── scripts/
│   └── download_models.py           # HuggingFace model downloader
├── docs/
│   ├── SETUP_GUIDE.md               # This file
│   └── MACHINE_REQUIREMENTS.md
├── weights/                          # Model weights (gitignored)
├── gallery/                          # Generated models (gitignored)
└── README.md
```
