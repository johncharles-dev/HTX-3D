#!/usr/bin/env bash
# Run the TRELLIS.2 image-to-3D example on Blackwell (sm_120), outside the service.
# Used as the post-install acceptance test — it exercises the CUDA extension build
# directly, without going through the HTTP service.
set -uo pipefail

TRELLIS2_ROOT="${TRELLIS2_ROOT:-/opt/trellis2}"
TRELLIS2_CONDA_SH="${TRELLIS2_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
TRELLIS2_CONDA_ENV="${TRELLIS2_CONDA_ENV:-trellis2}"

source "$TRELLIS2_CONDA_SH"
conda activate "$TRELLIS2_CONDA_ENV"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
# Needed for nvdiffrast/nvdiffrec JIT plugin compiles at runtime -> real Blackwell
# SASS, no PTX garbage.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
# FA2, matching run_service.sh. flash_attn_3 (installed as 3.0.0) was tried and
# found wrong on Blackwell sm_120; the 3.79 benchmark ran through the service on
# FA2, so the acceptance test must exercise that same path.
export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-flash_attn}"
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HF auth for the gated DINOv3 image conditioner. No credential is stored in this
# repository and none is defaulted: run `huggingface-cli login` on the host, or
# export HF_TOKEN in the environment.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HF_HOME/token" ]; then
    HF_TOKEN="$(cat "$HF_HOME/token")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

cd "$TRELLIS2_ROOT"
python -u example.py
