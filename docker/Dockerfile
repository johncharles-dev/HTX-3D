## --- Stage 1: Build Frontend -----------------------------
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

## --- Stage 2: Backend + Runtime --------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies (Ubuntu 24.04 ships Python 3.12)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git wget curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libopengl0 \
    && rm -rf /var/lib/apt/lists/*

# Ensure 'python' command works
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Allow pip to install globally (Ubuntu 24.04 PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Install backend Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install TRELLIS engine runtime dependencies
# (not the full upstream requirements.txt which includes torch, flash-attn, gradio, etc.)
RUN pip install --no-cache-dir \
    easydict \
    scipy \
    ninja \
    igraph \
    tqdm \
    trimesh==4.11.1 \
    pyvista==0.46.5 \
    pymeshfix==0.17.0 \
    xatlas==0.0.10 \
    open3d==0.19.0 \
    rembg>=2.0.72 \
    onnxruntime-gpu==1.24.1 \
    imageio-ffmpeg>=0.5.1 \
    opencv-python-headless>=4.11.0 \
    "huggingface_hub>=0.23,<0.25" \
    "transformers>=4.35.0,<4.50" \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Install Hunyuan3D engine runtime dependencies
RUN pip install --no-cache-dir \
    diffusers==0.30.0 \
    omegaconf \
    pymeshlab \
    pygltflib \
    cupy-cuda12x \
    realesrgan \
    basicsr \
    timm \
    torchdiffeq \
    einops \
    pytorch-lightning

# Install CUDA extensions with Blackwell support
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0;12.0"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# Copy TRELLIS engine source
COPY backend/engines/trellis/ ./engines/trellis/

# Copy Hunyuan3D engine source
COPY backend/engines/hunyuan/ ./engines/hunyuan/

# Copy SAM3 segmentation engine source
COPY backend/engines/sam3/ ./engines/sam3/

# Copy SAM 3D Objects engine source
COPY backend/engines/sam3d_objects/ ./engines/sam3d_objects/

# Build CUDA extensions from source (cloned from GitHub)
# These MUST be compiled with the correct TORCH_CUDA_ARCH_LIST for Blackwell support.
# PTX fallback produces garbage results on newer GPUs (telltale sign: 66 billion GB OOM).

# 1. nvdiffrast - differentiable rasterization
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && pip install --no-build-isolation /tmp/extensions/nvdiffrast

# 2. diffoctreerast - octree rasterization
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast \
    && pip install --no-build-isolation /tmp/extensions/diffoctreerast

# 3. diff-gaussian-rasterization - Gaussian splatting renderer (from mip-splatting)
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting \
    && pip install --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# 4. spconv (CUDA 12.6 forward compat -- no cu128 build exists)
RUN pip install --no-cache-dir spconv-cu126==2.3.8

# 5. kaolin (cu126 wheel -- no cu128 wheel exists)
RUN pip install --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html

# 6. Hunyuan custom rasterizer (build from source for Blackwell)
RUN cd /app/engines/hunyuan/hy3dpaint/custom_rasterizer \
    && pip install --no-build-isolation .

# 7. Rebuild mesh_inpaint_processor for Python 3.12 (source ships with 3.11 .so)
RUN pip install --no-cache-dir pybind11 \
    && cd /app/engines/hunyuan/hy3dpaint/DifferentiableRenderer \
    && rm -f mesh_inpaint_processor*.so \
    && python -c "import pybind11; print(pybind11.get_cmake_dir())" \
    && c++ -O3 -Wall -shared -std=c++17 -fPIC \
       $(python -m pybind11 --includes) \
       mesh_inpaint_processor.cpp \
       -o mesh_inpaint_processor$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 8. SAM3 segmentation dependencies
RUN pip install --no-cache-dir \
    "ftfy==6.1.1" \
    "iopath>=0.1.10" \
    "seaborn" \
    "pycocotools" \
    "psutil"

# 9. SAM 3D Objects dependencies (pytorch3d + hydra + runtime deps)
RUN pip install --no-cache-dir \
    "hydra-core==1.3.2" \
    "moge @ git+https://github.com/microsoft/MoGe.git" \
    "gsplat==1.5.0" \
    loguru optree astor lightning
RUN pip install --no-cache-dir \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git"

# 10. Pin versions back after SAM3D deps may have upgraded them
# - huggingface-hub <1.0 required by TRELLIS transformers
# - utils3d pinned to TRELLIS-compatible commit (MoGe upgrades to 1.3 which breaks TRELLIS)
RUN pip install --no-cache-dir \
    "huggingface_hub>=0.23,<1.0" \
    "transformers>=4.35.0,<4.50" \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Cleanup build artifacts
RUN rm -rf /tmp/extensions

# Copy backend application code
COPY backend/app/ ./app/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist/ ./frontend/dist/

# Environment defaults for Blackwell compatibility
ENV XFORMERS_DISABLED=1
ENV ATTN_BACKEND=sdpa
ENV TRELLIS_ENGINE_DIR=/app/engines/trellis
ENV HUNYUAN_ENGINE_DIR=/app/engines/hunyuan
ENV SAM3_DIR=/app/engines/sam3
ENV SAM3_BPE_PATH=/app/engines/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
ENV SAM3D_OBJECTS_DIR=/app/engines/sam3d_objects
ENV WEIGHTS_DIR=/app/weights
ENV GALLERY_DIR=/app/gallery

# Volumes for persistent data
VOLUME ["/app/weights", "/app/gallery"]

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
