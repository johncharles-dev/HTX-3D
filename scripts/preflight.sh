#!/usr/bin/env bash
# HTX-3D deployment preflight — READ ONLY. Installs nothing, changes nothing.
#
# Checks every assumption docs/RUNBOOK_UNGATED.md and docs/RUNBOOK_GATED.md make, and
# reports which phase would fail before you spend an hour finding out.
#
#     bash scripts/preflight.sh
#
# Exit: 0 all clear · 1 warnings only · 2 at least one blocking failure.

set -uo pipefail

PASS=0; WARN=0; FAIL=0
say()  { printf '  %-6s %-34s %s\n' "$1" "$2" "$3"; }
ok()   { PASS=$((PASS+1)); say "[PASS]" "$1" "$2"; }
warn() { WARN=$((WARN+1)); say "[WARN]" "$1" "$2"; }
bad()  { FAIL=$((FAIL+1)); say "[FAIL]" "$1" "$2"; }
hdr()  { printf '\n== %s\n' "$1"; }

echo "HTX-3D preflight — read only, nothing is installed or modified"
echo "run at: $(date -u '+%Y-%m-%d %H:%M:%SZ')"

# ---------------------------------------------------------------- environment
hdr "Environment"

IS_WSL=no
if grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=yes
  ok "WSL2" "yes — WSL-specific guidance applies"
else
  ok "WSL2" "no — native Linux; skip the .wslconfig notes"
fi

if [ -r /etc/os-release ]; then
  . /etc/os-release
  case "${VERSION_ID:-}" in
    24.04|22.04) ok "Ubuntu version" "${VERSION_ID} — supported" ;;
    "")          warn "Ubuntu version" "unknown — check the CUDA repo URL by hand" ;;
    *)           warn "Ubuntu version" "${VERSION_ID} — untested; runbook assumes 22.04/24.04" ;;
  esac
  if [ "$IS_WSL" = yes ]; then
    say "[INFO]" "CUDA repo to use" "wsl-ubuntu (same for both Ubuntu versions)"
  else
    say "[INFO]" "CUDA repo to use" "ubuntu$(echo "${VERSION_ID:-}" | tr -d .)"
  fi
else
  bad "Ubuntu version" "/etc/os-release missing"
fi

if sudo -n true 2>/dev/null; then
  ok "sudo" "available without a password prompt"
elif sudo -v 2>/dev/null; then
  ok "sudo" "available (password accepted)"
else
  bad "sudo" "NO SUDO — Phase 1 blocks. Needs an administrator."
fi

# ------------------------------------------------------------------------ GPU
hdr "GPU  (Phase 2.1 / Phase 5)"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
  ok "nvidia-smi" "$GPU  ($VRAM)"
  case "$CC" in
    12.0) ok "compute capability" "12.0 — Blackwell, matches the runbook" ;;
    "")   warn "compute capability" "could not read" ;;
    *)    warn "compute capability" "$CC — NOT 12.0. TORCH_CUDA_ARCH_LIST assumptions differ." ;;
  esac
  DRV_MAJ=${DRV%%.*}
  if [ "${DRV_MAJ:-0}" -ge 570 ] 2>/dev/null; then
    ok "driver" "$DRV"
  else
    bad "driver" "$DRV — Blackwell needs 570+. On WSL, fix on the WINDOWS side."
  fi
  if [ "$IS_WSL" = yes ]; then
    if ls /usr/lib/wsl/lib/libcuda.so* >/dev/null 2>&1; then
      ok "WSL GPU libs" "/usr/lib/wsl/lib present"
    else
      bad "WSL GPU libs" "/usr/lib/wsl/lib/libcuda.so missing — do NOT apt install nvidia-driver-*"
    fi
  fi
else
  bad "nvidia-smi" "not working. On WSL this is a WINDOWS driver problem."
fi

# --------------------------------------------------------------------- capacity
hdr "Capacity  (Phase 2.2 / 2.3)"

FREE_G=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${FREE_G:-}" ]; then
  if   [ "$FREE_G" -ge 170 ]; then ok   "disk free on /" "${FREE_G} GB — enough for gated (170 GB)"
  elif [ "$FREE_G" -ge 150 ]; then warn "disk free on /" "${FREE_G} GB — enough for ungated, tight for gated"
  else                             bad  "disk free on /" "${FREE_G} GB — need 150 GB minimum"
  fi
  [ "$IS_WSL" = yes ] && [ "$FREE_G" -lt 150 ] && \
    say "[INFO]" "disk fix" "virtual disk: fix from Windows, not Linux. wsl --manage <distro> --resize"
else
  warn "disk free on /" "could not determine"
fi

RAM_G=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
if [ -n "${RAM_G:-}" ]; then
  if   [ "$RAM_G" -ge 24 ]; then ok   "RAM" "${RAM_G} GB"
  elif [ "$RAM_G" -ge 16 ]; then warn "RAM" "${RAM_G} GB — builds may OOM; 24 GB recommended"
  else                           bad  "RAM" "${RAM_G} GB — extension builds will OOM"
  fi
  [ "$IS_WSL" = yes ] && [ "$RAM_G" -lt 24 ] && \
    say "[INFO]" "RAM fix" "set memory=24GB in C:\\Users\\<you>\\.wslconfig, then wsl --shutdown"
else
  warn "RAM" "could not determine"
fi

NPROC=$(nproc 2>/dev/null || echo "?")
say "[INFO]" "cores" "$NPROC  (build parallelism is pinned in the Dockerfile, not derived from this)"

# ----------------------------------------------------------------------- docker
hdr "Docker  (Phase 4 / Phase 5)"

INIT=$(ps -p 1 -o comm= 2>/dev/null || echo "?")
if [ "$INIT" = systemd ]; then
  ok "init system" "systemd — use systemctl"
else
  warn "init system" "$INIT — no systemd. Use 'sudo service docker start'."
  [ "$IS_WSL" = yes ] && \
    say "[INFO]" "systemd" "enable via [boot] systemd=true in /etc/wsl.conf, then wsl --shutdown"
fi

if command -v docker >/dev/null 2>&1; then
  ok "docker installed" "$(docker --version 2>/dev/null | cut -d, -f1)"
  if docker ps >/dev/null 2>&1; then
    ok "docker usable" "daemon running, no sudo needed"
    if docker compose version >/dev/null 2>&1; then
      ok "compose v2" "$(docker compose version --short 2>/dev/null)"
    else
      bad "compose v2" "missing — 'docker compose' is required, not docker-compose"
    fi
    if timeout 180 docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 \
         nvidia-smi -L >/dev/null 2>&1; then
      ok "GPU in docker" "--gpus all works — Phase 5 satisfied"
    else
      bad "GPU in docker" "--gpus all FAILED. Phase 5 is not done. (May also be a slow first pull.)"
    fi
  else
    if sudo -n docker ps >/dev/null 2>&1; then
      warn "docker usable" "works with sudo only — add yourself to the docker group and re-login"
    else
      bad "docker usable" "daemon not reachable. Start it (Phase 4.2)."
    fi
  fi
else
  warn "docker installed" "absent — Phase 4 will install it"
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
  ok "nvidia-ctk" "$(nvidia-ctk --version 2>/dev/null | head -1)"
  [ -f /etc/cdi/nvidia.yaml ] && ok "CDI spec" "/etc/cdi/nvidia.yaml present" \
                              || warn "CDI spec" "absent — Phase 5.3 generates it"
else
  warn "nvidia-ctk" "absent — Phase 5.1 installs it"
fi

# -------------------------------------------------------------------- toolchain
hdr "Host toolchain  (Phase 3 / Phase 6 — TRELLIS.2 only)"

if command -v nvcc >/dev/null 2>&1; then
  NVCC=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | awk '{print $2}')
  if [ "$NVCC" = "12.8" ]; then ok   "nvcc" "12.8"
  else                          warn "nvcc" "$NVCC — TRELLIS.2 expects 12.8; container is unaffected"
  fi
else
  warn "nvcc" "absent — only needed for TRELLIS.2. Container ships its own CUDA."
fi

if command -v conda >/dev/null 2>&1; then
  CB=$(conda info --base 2>/dev/null)
  ok "conda" "$CB"
  [ -f "$CB/etc/profile.d/conda.sh" ] \
    && say "[INFO]" "TRELLIS2_CONDA_SH" "$CB/etc/profile.d/conda.sh" \
    || warn "conda.sh" "missing at $CB/etc/profile.d/conda.sh"
else
  warn "conda" "absent — only needed for TRELLIS.2"
fi

# ------------------------------------------------------------------------- repo
hdr "Repository  (Phase 6 / 7 / 8)"

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
say "[INFO]" "repo root" "$ROOT"

for f in docker/Dockerfile docker/docker-compose.yml docker/.env.example scripts/download_models.py; do
  [ -f "$ROOT/$f" ] && ok "$f" "present" || bad "$f" "MISSING — wrong directory, or an incomplete copy"
done

if [ -f "$ROOT/docker/.env" ]; then
  ok "docker/.env" "exists"
else
  warn "docker/.env" "absent — cp docker/.env.example docker/.env  (Phase 6.2)"
fi

CKPT="$ROOT/backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
if [ -f "$CKPT" ]; then
  SZ=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
  if [ "$SZ" -gt 60000000 ]; then ok  "RealESRGAN ckpt" "$((SZ/1024/1024)) MB"
  else                            bad "RealESRGAN ckpt" "only $((SZ/1024)) KB — truncated, refetch"
  fi
else
  warn "RealESRGAN ckpt" "absent — fetch BEFORE the build (Phase 8), or Hunyuan textures break"
fi

if command -v python3 >/dev/null 2>&1; then
  ok "python3" "$(python3 --version 2>&1 | awk '{print $2}')"
  python3 -c "import huggingface_hub" 2>/dev/null \
    && ok "huggingface_hub" "importable" \
    || warn "huggingface_hub" "absent — needed by scripts/download_models.py"
else
  bad "python3" "absent — needed to fetch weights"
fi

# --------------------------------------------------------------------- verdict
hdr "Verdict"
printf '  %d pass, %d warn, %d fail\n\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "  BLOCKED. Fix every [FAIL] above before starting the runbook."
  echo "  A [FAIL] on sudo, nvidia-smi or GPU-in-docker cannot be worked around."
  exit 2
elif [ "$WARN" -gt 0 ]; then
  echo "  READY, with warnings. Each [WARN] names the phase that resolves it."
  exit 1
else
  echo "  READY. Every precondition satisfied."
  exit 0
fi
