#!/usr/bin/env bash
# Launch the host-side TRELLIS.2 microservice in the trellis2 conda env.
#
# Every machine-specific value is an environment variable with a default. Override
# them in /etc/default/trellis2-service (read by the systemd unit) rather than
# editing this file, so the vendored copy stays identical across deployments.
set -uo pipefail

TRELLIS2_ROOT="${TRELLIS2_ROOT:-/opt/trellis2}"
TRELLIS2_CONDA_SH="${TRELLIS2_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
TRELLIS2_CONDA_ENV="${TRELLIS2_CONDA_ENV:-trellis2}"

source "$TRELLIS2_CONDA_SH"
conda activate "$TRELLIS2_CONDA_ENV"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
# 12.0 = sm_120 (Blackwell: RTX 5090, RTX PRO 6000). Change only for another arch.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
# Correct Blackwell (sm120) backend: FA2. NOT flash_attn_3.
export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-flash_attn}"
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$TRELLIS2_ROOT"

# HF auth for the gated DINOv3 image conditioner. No credential is stored in this
# repository and none is defaulted: run `huggingface-cli login` on the host, or
# export HF_TOKEN in the environment / EnvironmentFile.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_HOME/token" ]; then
    HF_TOKEN="$(cat "$HF_HOME/token")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

export TRELLIS2_SERVICE_PORT="${TRELLIS2_SERVICE_PORT:-8710}"
cd "$TRELLIS2_ROOT"
exec python -u "$TRELLIS2_ROOT/service/trellis2_service.py"
