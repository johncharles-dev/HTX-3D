# HTX-3D — Setup and Deployment Guide

Setting up the full stack: five engines, interactive segmentation, metric auto-scaling and
the web UI, on Linux, WSL2 or Docker.

**Docker is the deployment path** and is documented first. Native setup is for development
on the engines themselves.

> **Read [Before you start](#0-before-you-start) first.** Four models are gated behind
> accepted licence terms, one 67 MB checkpoint must be present *before* the image is built,
> and three licences carry real restrictions. Each of these stops an install cold if
> discovered late.

---

## Contents

0. [Before you start](#0-before-you-start)
1. [System requirements](#1-system-requirements)
2. [Setup with Docker](#2-setup-with-docker)
3. [TRELLIS.2 host service (optional)](#3-trellis2-host-service-optional)
4. [Native setup on Linux](#4-native-setup-on-linux)
5. [Windows via WSL2](#5-windows-via-wsl2)
6. [Model weights reference](#6-model-weights-reference)
7. [Verifying the install](#7-verifying-the-install)
8. [Troubleshooting](#8-troubleshooting)

---

## 0. Before you start

### Gated models need an account, not just a download

SAM 3, SAM 3.1, SAM 3D Objects and DINOv3 are gated on HuggingFace. Access is granted
**per account** after accepting each model's terms, and **does not transfer** with copied
files — receiving the weights on a drive does not grant the licence.

```bash
huggingface-cli login          # or export HF_TOKEN=...
python scripts/download_models.py --list    # shows every model and which are gated
```

Accept the terms at `https://huggingface.co/<repo>` for each gated repository, using the
same account your token belongs to.

### One checkpoint must exist before you build

`RealESRGAN_x4plus.pth` (67 MB) drives Hunyuan3D's texture upscaling. It is **not committed
to this repository**, and the Dockerfile `COPY`s the engine tree into the image — so it has
to be on the **build host before `docker compose build`**. Fetching it afterwards does not
help; you must rebuild.

```bash
python scripts/download_models.py --model realesrgan
```

Without it, Hunyuan now fails at engine load with a message naming the file. The other four
engines are unaffected.

### Licences that constrain deployment

| Component | Constraint |
|---|---|
| **UniDepth** (auto-scale) | **CC BY-NC 4.0 — non-commercial.** Blocks commercial use of metric scaling |
| **Hunyuan3D 2.1** | Excludes the **EU, UK and South Korea** — the model *and its output*. Singapore is inside the permitted Territory |
| **SAM 3 / SAM 3D** | Trade controls: no military/warfare, nuclear or espionage use. The licence must travel with the weights |

Full audit: [`docs/reference/vendoring.md`](reference/vendoring.md).

---

## 1. System requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 12 GB VRAM | 24 GB+ |
| GPU compute | CUDA 8.0+ | 8.6+ (Ampere / Ada / Blackwell) |
| System RAM | 16 GB | 32 GB |
| **Storage** | **150 GB free** | 250 GB+ |
| NVIDIA driver | 525+ | 570+ (**required** for RTX 50 series) |

Peak VRAM measured is ~8.7 GB (TRELLIS.2), so 12 GB is workable. Only the RTX 5090
configuration has been benchmarked.

### Storage breakdown — the old figure of 25 GB was badly wrong

| Item | Size |
|---|---|
| Docker image | **35.2 GB** |
| HuggingFace cache | 40 GB |
| Hunyuan3D cache | 14 GB |
| SAM 3D Objects weights | 12 GB |
| TRELLIS weights | 5.3 GB |
| torch hub cache | 1.4 GB |
| rembg model | 0.2 GB |
| **Subtotal** | **~108 GB** |
| Gallery | **Unbounded** — reached 14 GB over ~800 generations |

Put the gallery on a volume with room to grow (see `GALLERY_HOST_DIR`).

### Software

| Component | Version |
|---|---|
| Python | 3.12 (container) / 3.11 (native conda) |
| Node.js | 20+ |
| CUDA toolkit | 12.8 (Blackwell) |
| PyTorch | 2.7.0+cu128 (container) |
| OS | Ubuntu 22.04 / 24.04, or Windows 10/11 via WSL2 |

---

## 2. Setup with Docker

### 2.1 Prerequisites

**NVIDIA driver.** Blackwell (RTX 50 series) **must** use the open kernel modules:

```bash
sudo apt update
sudo apt install nvidia-driver-570-open     # Blackwell — the -open variant is required
# sudo apt install nvidia-driver-535        # Ampere / Ada
sudo reboot
nvidia-smi
```

> The proprietary driver (without `-open`) fails on Blackwell.

**Docker + NVIDIA container toolkit:**

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER      # log out and back in

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# verify GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### 2.2 Get the project

```bash
git clone <repo-url> HTX-3D && cd HTX-3D
```

Copying from an existing machine instead:

```bash
rsync -av --exclude='weights/' --exclude='gallery/' --exclude='node_modules/' \
  --exclude='frontend/dist/' --exclude='__pycache__/' \
  user@source:/path/to/HTX-3D/ ./HTX-3D/
```

### 2.3 Configure host paths

```bash
cp docker/.env.example docker/.env
$EDITOR docker/.env
```

Nothing in the repository hardcodes a machine-specific path any more; everything comes from
here. `docker/.env` is gitignored. The variables:

| Variable | Default | Set it when |
|---|---|---|
| `WEIGHTS_HOST_DIR` | `../weights` | Weights should live on a data volume |
| `GALLERY_HOST_DIR` | `../gallery` | **Usually** — this one grows without bound |
| `SAM3D_HF_DIR` | `../weights/sam3d-objects-hf` | **Almost always** — see below |
| `HF_CACHE_DIR` | `$HOME/.cache/huggingface` | Your HF cache is elsewhere |
| `TORCH_CACHE_DIR` | `$HOME/.cache/torch` | " |
| `HY3DGEN_CACHE_DIR` | `$HOME/.cache/hy3dgen` | " |
| `U2NET_CACHE_DIR` | `$HOME/.u2net` | " |
| `TRELLIS2_SERVICE_URL` | `http://host.docker.internal:8710` | Only if TRELLIS.2 runs elsewhere |

> **`SAM3D_HF_DIR` almost certainly needs setting.** Its default has never been exercised —
> on the development machine those weights lived outside the repository. A wrong value
> surfaces as a `FileNotFoundError` from the SAM 3D engine listing every path it searched.

Check what resolves before building:

```bash
cd docker && docker compose config | grep -A1 source:
```

### 2.4 Fetch model weights

```bash
python scripts/download_models.py --check          # what is already present
python scripts/download_models.py                  # ungated families
python scripts/download_models.py --model all      # including gated (needs a token)
```

The default (`ungated`) succeeds without a token and reports which gated repositories need
terms accepted. `--model all` opts in. See [section 6](#6-model-weights-reference).

**Do not skip `--model realesrgan`** if you want Hunyuan textures — it must run before the
build.

### 2.5 Build and run

```bash
cd docker
docker compose up -d --build
docker compose logs -f
```

A from-scratch build took **~10 minutes** on the development machine (RTX 5090, fast
network). It compiles nvdiffrast, diffoctreerast, diff-gaussian-rasterization, pytorch3d and
the Hunyuan rasterizer against five GPU architectures. Expect longer on slower hardware.

Open **http://localhost:8000** — the container serves both the API and the built frontend.

### 2.6 Compose reference

Seven bind mounts, all host-side paths configurable:

| Container path | Host source | Contents |
|---|---|---|
| `/app/weights` | `${WEIGHTS_HOST_DIR}` | TRELLIS models |
| `/app/gallery` | `${GALLERY_HOST_DIR}` | Generated models |
| `/app/weights/sam3d-objects-hf` | `${SAM3D_HF_DIR}` (**read-only**) | SAM 3D Objects |
| `/root/.cache/huggingface` | `${HF_CACHE_DIR}` | SAM 3, DINOv3, UniDepth, CLIP |
| `/root/.cache/torch` | `${TORCH_CACHE_DIR}` | DINOv2 torch hub |
| `/root/.cache/hy3dgen` | `${HY3DGEN_CACHE_DIR}` | Hunyuan3D |
| `/root/.u2net` | `${U2NET_CACHE_DIR}` | rembg |

Container-side paths (`WEIGHTS_DIR`, `GALLERY_DIR`, the engine dirs, `CORS_ORIGINS`) are
fixed in `docker-compose.yml` — they describe the image layout, not your machine, and
changing them breaks the container.

`extra_hosts: host.docker.internal:host-gateway` is what makes the TRELLIS.2 hostname
resolve on Linux Docker. **Do not remove it.**

### 2.7 Managing the container

```bash
docker compose down                  # stop
docker compose up -d --build         # rebuild after code changes
docker compose logs -f
docker compose exec htx-3d bash      # service name is htx-3d (lowercase)
```

> Application code is `COPY`d into the image, not mounted. Editing `backend/app/` has no
> effect until you rebuild.

---

## 3. TRELLIS.2 host service (optional)

TRELLIS.2 is the highest-scoring engine but **cannot run in the container** — it needs the
host's torch 2.10 / FlashAttention-2 stack for sm_120. It runs as an HTTP service on the
host and the container proxies to it.

The other four engines work without it. The backend probes it at startup and logs a warning
if absent; nothing else is affected.

Full instructions, including the pinned upstream commit and the `transformers` co-pin:
**[`services/trellis2/README.md`](../services/trellis2/README.md)**.

> That directory's own §7 records that nothing in it has been run in its vendored form. A
> clean-machine install is the first thing to do there.

---

## 4. Native setup on Linux

For engine development. The **Dockerfile is the authoritative dependency list** — it
resolves conflicts (huggingface-hub, utils3d, transformers) that the steps below do not.

### 4.1 Toolchain

```bash
# CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install cuda-toolkit-12-8

# ~/.bashrc
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh

# Node 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs

# system libraries
sudo apt install -y git wget curl build-essential \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

### 4.2 Environment and dependencies

```bash
conda create -n htx-3d python=3.11 -y && conda activate htx-3d
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

pip install -r backend/requirements.txt
pip install pillow imageio imageio-ffmpeg opencv-python-headless
pip install trimesh xatlas pyvista pymeshfix open3d
pip install rembg onnxruntime-gpu
pip install tqdm easydict scipy ninja igraph
pip install "huggingface_hub>=0.23,<0.25" "transformers>=4.35.0,<4.50" "pydantic>=2.0,<2.10"
pip install utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install spconv-cu126==2.3.8                    # no cu128 build; cu126 works via forward compat
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html
```

> Do **not** install `flash-attn` for the container stack — SDPA is the supported backend
> and flash-attn crashes on Blackwell. (TRELLIS.2 is the exception; it needs FA2 and lives
> in its own environment.)

The list above covers **TRELLIS only**. Hunyuan3D, SAM 3 and SAM 3D Objects each need
further packages — see the numbered steps in `docker/Dockerfile`, which is kept working.

### 4.3 CUDA extensions

Extensions **must** be compiled for your architecture. PTX fallback produces garbage on
newer GPUs — the telltale sign is a nonsensical multi-billion-GB OOM.

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0;12.0"
cd backend/engines/trellis && bash setup.sh --all && cd ../../..
```

### 4.4 Run

```bash
cd frontend && npm install && npm run dev &     # :5173, proxies the API

XFORMERS_DISABLED=1 ATTN_BACKEND=sdpa \
  TRELLIS_ENGINE_DIR=./backend/engines/trellis \
  WEIGHTS_DIR=./weights GALLERY_DIR=./gallery \
  TRELLIS2_SERVICE_URL=http://localhost:8710 \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

> `TRELLIS2_SERVICE_URL` defaults to `host.docker.internal`, which only resolves inside a
> container. Set it to `localhost` when running natively.

---

## 5. Windows via WSL2

Run everything inside WSL2 — the NVIDIA driver is installed on **Windows**, not in the
distro.

```powershell
wsl --install -d Ubuntu-24.04      # then reboot
```

Install the Windows NVIDIA driver (570+ for RTX 50 series), then inside WSL2:

```bash
nvidia-smi                          # should work with no driver installed in the distro
```

From there follow [section 2](#2-setup-with-docker) or [section 4](#4-native-setup-on-linux)
unchanged. Do **not** install a Linux NVIDIA driver inside WSL2 — it breaks passthrough.

Keep the project on the WSL2 filesystem (`~/HTX-3D`), not `/mnt/c/`. Model loading over the
Windows filesystem bridge is dramatically slower.

---

## 6. Model weights reference

Ten HuggingFace families plus one direct download. Every repo ID is recorded in
`scripts/download_models.py` with the file:line it was read from.

```bash
python scripts/download_models.py --list      # families, sources, gated status
python scripts/download_models.py --check     # what is present, downloads nothing
```

| Family | Repo | Gated | Needed by |
|---|---|---|---|
| `trellis_image` | `JeffreyXiang/TRELLIS-image-large` | | TRELLIS image-to-3D (required) |
| `trellis_text` | `JeffreyXiang/TRELLIS-text-large` | | TRELLIS text-to-3D |
| `hunyuan` | `tencent/Hunyuan3D-2.1` | | Hunyuan3D shape + texture |
| `unidepth` | `lpiccinelli/unidepth-v2-vits14` | | Auto-scale (**CC BY-NC**) |
| `clip` | `openai/clip-vit-base-patch32` | | Auto-scale class priors |
| `trellis2` | `microsoft/TRELLIS.2-4B` | | TRELLIS.2 host service |
| `sam3` | `facebook/sam3` | **Yes** | SAM 3 segmentation |
| `sam3_1` | `facebook/sam3.1` | **Yes** | SAM 3.1 variant |
| `sam3d` | `facebook/sam-3d-objects` | **Yes** | SAM 3D Objects |
| `dinov3` | `facebook/dinov3-vitl16-…` | **Yes** | TRELLIS.2 image conditioner |
| `realesrgan` | GitHub release asset | | Hunyuan texture upscaling |

Where they land: `trellis_image` and `trellis_text` go to `--output` (default `./weights`);
everything else goes to the HuggingFace cache (`$HF_HOME`); `realesrgan` goes into the
engine tree.

Six repositories present in the development cache are **not** fetched because their role
could not be established from this repository — `--list` names them and why.

### Air-gapped installs

Weights transfer as tar archives with a manifest. The HF cache uses symlinks from
`snapshots/` into `blobs/`, so it must be extracted onto a Linux filesystem — **not NTFS or
exFAT**, which cannot represent them.

---

## 7. Verifying the install

```bash
curl -s localhost:8000/api/health | python3 -m json.tool
```

Expect `status: ok`, your GPU detected, and four engines registered:

```json
{"status":"ok",
 "gpu":{"name":"...","is_blackwell":true},
 "engines_registered":["trellis","hunyuan","sam3d","trellis2"],
 "models_loaded":["trellis"]}
```

`engines_registered` only means the objects were constructed. Hunyuan, SAM 3D and TRELLIS.2
lazy-load, so the real test is one generation per engine through the UI. The container logs
show each swap:

```
docker compose logs -f | grep -iE 'loading engine|reachable|registered'
```

TRELLIS.2, if deployed:

```bash
systemctl is-active trellis2-service
curl -s localhost:8710/health          # {"status":"ok","model_loaded":false}
```

---

## 8. Troubleshooting

| Symptom | Cause |
|---|---|
| Multi-billion-GB OOM | CUDA extensions built without your architecture in `TORCH_CUDA_ARCH_LIST` — PTX fallback. Rebuild |
| `FileNotFoundError` naming `RealESRGAN_x4plus.pth` | Checkpoint absent at build time. Fetch it, then **rebuild** |
| SAM 3D: `FileNotFoundError` listing search paths | `SAM3D_HF_DIR` wrong. The message lists every path tried |
| 401 / 403 on a `facebook/*` repo | Token valid, but terms not accepted on that account |
| TRELLIS.2 unreachable at startup | Expected if not deployed — non-fatal. Otherwise check the service and `TRELLIS2_SERVICE_URL` |
| Code edits have no effect | Code is baked into the image. Rebuild |
| `docker compose exec HTX-3D` fails | Service name is lowercase `htx-3d` |
| Blackwell driver problems | Must be the `-open` driver variant |

---

## Related documentation

| Document | Contents |
|---|---|
| [`docker/.env.example`](../docker/.env.example) | Every operator-set variable |
| [`services/trellis2/README.md`](../services/trellis2/README.md) | TRELLIS.2 host service |
| [`docs/reference/vendoring.md`](reference/vendoring.md) | Engine provenance and licences |
| [`docs/reference/dependency-pins.md`](reference/dependency-pins.md) | Pinned commits and observed drift |
| [`docs/MACHINE_REQUIREMENTS.md`](MACHINE_REQUIREMENTS.md) | Hardware and network requirements |
