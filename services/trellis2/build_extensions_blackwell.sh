#!/usr/bin/env bash
# Build TRELLIS.2 CUDA extensions for Blackwell (sm_120: RTX 5090, RTX PRO 6000).
#
# Machine-specific values are environment variables with defaults; override them in
# the environment rather than editing this file.
#
# REPRODUCIBILITY CAVEAT: upstream setup.sh clones CuMesh and FlexGEMM from the
# default branch tip with no commit pin, so a build today may not match the one
# benchmarked at 3.79. The commits used for that build are not recoverable (the
# source trees under /tmp/extensions were cleared, and pip recorded only a local
# file:// path). Once a known-good build is established, pin it by exporting
# CUMESH_REF / FLEXGEMM_REF. See README.md.
set -uo pipefail

TRELLIS2_ROOT="${TRELLIS2_ROOT:-/opt/trellis2}"
TRELLIS2_CONDA_SH="${TRELLIS2_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
TRELLIS2_CONDA_ENV="${TRELLIS2_CONDA_ENV:-trellis2}"

# Empty = clone the default branch tip (upstream behaviour). Set to a commit hash
# or tag to pin.
CUMESH_REF="${CUMESH_REF:-}"
FLEXGEMM_REF="${FLEXGEMM_REF:-}"

source "$TRELLIS2_CONDA_SH"
conda activate "$TRELLIS2_CONDA_ENV"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export FORCE_CUDA=1

EXTDIR="${TRELLIS2_EXTDIR:-/tmp/extensions}"
mkdir -p "$EXTDIR"

step() { echo; echo "========== $1 =========="; }
result() { if [ "$1" -eq 0 ]; then echo ">>> OK: $2"; else echo ">>> FAIL($1): $2"; fi; }

# Clone a repo unless it is already present, optionally at a pinned ref. A clone
# failure is fatal: continuing would pip-install a non-existent directory and
# report a confusing error several steps later.
clone_ext() {
    local name="$1" url="$2" ref="$3" dest="$EXTDIR/$1"
    if [ -d "$dest" ]; then
        echo "  reusing existing $dest"
        return 0
    fi
    git clone -q --recursive "$url" "$dest" || {
        echo ">>> FAIL: could not clone $name from $url" >&2
        exit 1
    }
    if [ -n "$ref" ]; then
        git -C "$dest" checkout -q "$ref" || {
            echo ">>> FAIL: could not check out $name at pinned ref '$ref'" >&2
            exit 1
        }
        git -C "$dest" submodule update --init --recursive -q || {
            echo ">>> FAIL: could not update $name submodules after checkout" >&2
            exit 1
        }
    fi
    echo "  $name at $(git -C "$dest" rev-parse HEAD)"
}

echo "nvcc: $(which nvcc)  |  $(nvcc --version | tail -2 | head -1)"
echo "ARCH_LIST=$TORCH_CUDA_ARCH_LIST  CUDA_HOME=$CUDA_HOME  MAX_JOBS=$MAX_JOBS"
echo "TRELLIS2_ROOT=$TRELLIS2_ROOT  EXTDIR=$EXTDIR"

# 1. o-voxel (ships inside the TRELLIS.2 tree; eigen submodule must be pulled)
step "o-voxel"
cp -r "$TRELLIS2_ROOT/o-voxel" "$EXTDIR/o-voxel" 2>/dev/null || true
rm -rf "$EXTDIR/o-voxel/build"
pip install "$EXTDIR/o-voxel" --no-build-isolation 2>&1 | tail -8
result ${PIPESTATUS[0]} "o-voxel"

# 2. nvdiffrast v0.4.0 (python install; CUDA plugin JITs at first use)
step "nvdiffrast"
[ -d "$EXTDIR/nvdiffrast" ] || git clone -q -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$EXTDIR/nvdiffrast"
pip install "$EXTDIR/nvdiffrast" --no-build-isolation 2>&1 | tail -5
result ${PIPESTATUS[0]} "nvdiffrast"

# 3. nvdiffrec (JeffreyXiang renderutils branch -> nvdiffrec_render)
step "nvdiffrec"
[ -d "$EXTDIR/nvdiffrec" ] || git clone -q -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXTDIR/nvdiffrec"
pip install "$EXTDIR/nvdiffrec" --no-build-isolation 2>&1 | tail -5
result ${PIPESTATUS[0]} "nvdiffrec"

# 4. CuMesh (cubvh + eigen submodules; URL matches upstream setup.sh:125)
step "CuMesh"
clone_ext CuMesh https://github.com/JeffreyXiang/CuMesh.git "$CUMESH_REF"
rm -rf "$EXTDIR/CuMesh/build"
pip install "$EXTDIR/CuMesh" --no-build-isolation 2>&1 | tail -10
result ${PIPESTATUS[0]} "CuMesh"

# 5. FlexGEMM (CUTLASS-based, riskiest for Blackwell; upstream setup.sh:131)
step "FlexGEMM"
clone_ext FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git "$FLEXGEMM_REF"
rm -rf "$EXTDIR/FlexGEMM/build"
pip install "$EXTDIR/FlexGEMM" --no-build-isolation 2>&1 | tail -15
result ${PIPESTATUS[0]} "FlexGEMM"

step "IMPORT CHECK"
python - <<'PY' 2>&1 | tail -20
import importlib
for m in ["o_voxel","cumesh","nvdiffrast.torch","nvdiffrec_render.light","flex_gemm","flash_attn_interface"]:
    try:
        importlib.import_module(m)
        print("OK  ", m)
    except Exception as e:
        print("FAIL", m, "->", repr(e)[:160])
PY
echo; echo "===== BUILD SCRIPT DONE ====="
