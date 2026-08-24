# HTX-3D — Deployment Runbook: GATED

**From a bare WSL2 instance. No CUDA, no Docker, no conda installed.**
**HuggingFace gated access available from the start.** All four engines plus TRELLIS.2.

One person, one terminal. Follow top to bottom. Do not skip Phase 0.

There is **no `TROUBLESHOOTING.md`** in this repository. Every fix is inline here.

## What you get

| Delivered |
|---|
| TRELLIS image-to-3D |
| Hunyuan3D 2.1 (shape + PBR texture) |
| SAM 3 interactive segmentation |
| SAM 3D Objects |
| **TRELLIS.2** (host service — the highest-scoring engine, 3.79 blinded) |
| Metric auto-scale, logo bake, UI, gallery, export |

## Before you start — read these three

1. **Confirm the gated terms are actually accepted before you begin.** "We have access"
   often means a token exists, not that the four licences were accepted on that account.
   Acceptance is **per account** and does not transfer with a copied token. Verify in
   Phase 7.1 — it is the one blocker no terminal work fixes.

2. **TRELLIS.2 has never been installed from this repo in its vendored form.**
   `services/trellis2/README.md` §7 records four things that have never executed. It is
   Phase 13, it is 2–4 hours, and it is the first thing you cut.

3. **Timings in this repo were measured on an RTX 5090 at 575 W.** The target card is
   300 W. Generation will be slower. No recorded `gen_s` is a threshold.

**Realistic plan: Phases 1–12 in one day. Phase 13 on a second day.**

---

## Phase 0 — Collect these first

**Two minutes. Do not skip.** Later phases reference these by letter.

```bash
cd ~ 2>/dev/null
echo "== A repo root      "; echo "(fill in after Phase 6)"
echo "== B user           "; whoami
echo "== C home           "; echo $HOME
echo "== D ubuntu version "; . /etc/os-release; echo "$VERSION_ID  ($PRETTY_NAME)"
echo "== E wsl?           "; grep -qi microsoft /proc/version && echo "YES - WSL2" || echo "NO - native Linux"
echo "== F init system    "; ps -p 1 -o comm=
echo "== G gpu            "; nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi NOT WORKING"
echo "== H disk           "; df -h / | tail -1
echo "== I ram            "; free -g | awk '/^Mem:/{print $2" GB total"}'
echo "== J docker         "; docker --version 2>/dev/null || echo "not installed (expected)"
echo "== K nvcc           "; nvcc --version 2>/dev/null | grep release || echo "not installed (expected)"
echo "== L conda          "; conda info --base 2>/dev/null || echo "not installed (expected)"
```

Write these down:

| ID | Value | Example | Used in |
|---|---|---|---|
| **A** repo root | `____________` | `/home/htx/HTX-3D` | Phases 6–13 |
| **B** user | `____________` | `htx` | Phase 4, Phase 13.7 systemd `User=` |
| **C** home | `____________` | `/home/htx` | Phase 6 cache paths |
| **D** Ubuntu | `____________` | `24.04` or `22.04` | **Phase 3 — picks the CUDA repo URL** |
| **E** WSL | `____________` | `YES - WSL2` | Phases 2, 4 |
| **F** init | `____________` | `systemd` or `init` | **Phase 4 and Phase 13.7** |
| **G** GPU | `____________` | `NVIDIA RTX PRO 6000 Blackwell Max-Q, 12.0, 97887 MiB, 590.48` | Phase 2, Phase 13 |
| **H** disk free | `____________` | `1007G  420G avail` | Phase 2 — need **170 GB** |
| **I** RAM | `____________` | `30 GB total` | Phases 9, 13 |
| **L** conda base | `____________` | `/home/htx/miniconda3` | **Phase 13.7** `TRELLIS2_CONDA_SH` = this **+ `/etc/profile.d/conda.sh`** |

> **`export A` matters.** Later phases paste `$A/...` into commands. After Phase 6, run
> `export A=$(pwd)` from the repo root. Re-export in any new terminal, and after `sudo -i`.

**Compute capability (G) must read `12.0`.** Anything else and every `TORCH_CUDA_ARCH_LIST`
value in Phase 13 is wrong — note it and do the container only.

---

## Phase 1 — sudo

**30 seconds.**

```bash
sudo -v && echo "SUDO OK"
```

**Correct output:** `SUDO OK`.

- **`<user> is not in the sudoers file` →** **Stop.** Needs a Windows-side administrator to
  reset the distro's default user. Nothing here works without it.

---

## Phase 2 — WSL2 sanity checks

**5 minutes.**

### 2.1 The GPU comes from Windows — do NOT install a Linux driver

```bash
nvidia-smi
ls -la /usr/lib/wsl/lib/libcuda.so* 2>/dev/null | head -3
```

**Correct output:** the GPU table, and at least one `libcuda.so` under `/usr/lib/wsl/lib`.

- **Works →** continue. **Never `apt install nvidia-driver-*` inside WSL.** It overwrites the
  WSL GPU stubs and breaks CUDA for the whole distro.
- **Not working →** the **Windows** driver is missing or old. Install it on Windows (570+ for
  Blackwell), then from Windows PowerShell `wsl --shutdown` and reopen. **Nothing inside
  Linux fixes this.**

### 2.2 Disk — the WSL virtual disk

```bash
df -h /
```

**Need 170 GB free** — 20 GB more than the ungated variant, for the gated weights.

**The WSL2 filesystem is a virtual disk (`ext4.vhdx`) on the Windows drive.** If `df` shows
under 170 GB free, **the fix is on the Windows side, not inside Linux.** Deleting files in
Linux will not help once the vhdx has hit its ceiling.

From **Windows** PowerShell:

```powershell
Get-PSDrive C                 # the real limit is free space on the Windows drive
wsl --shutdown
wsl --manage Ubuntu --resize 500GB
```

Replace `Ubuntu` with the name from `wsl -l -v`.

### 2.3 RAM — set it now, before any build

```bash
free -g
```

WSL2 defaults to roughly half of Windows' RAM. Set it explicitly in
`C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=8
swap=16GB
```

Then from Windows PowerShell:

```powershell
wsl --shutdown
```

Reopen and re-check `free -g`.

> **Why this matters:** the `flash_attn` build OOM'd at 3 parallel jobs on 30 GB. This
> runbook hardcodes **`MAX_JOBS=2` for flash_attn** and **`MAX_JOBS=4` for the CUDA
> extensions**. **Do not raise them, and do not use `$(nproc)`.**

### ✅ Definition of done — Phase 2

`nvidia-smi` works inside WSL. 170 GB free. RAM confirmed.

---

## Phase 3 — CUDA 12.8 toolkit

**15 minutes, ~3 GB. Required in this variant** — Phase 13 compiles CUDA extensions on the
host.

### 3.1 Check your Ubuntu version — this picks the repo URL

```bash
. /etc/os-release; echo "$VERSION_ID"
lsb_release -a 2>/dev/null
```

**Correct output:** `24.04` or `22.04`. That is value **D**.

### 3.2 Add the CUDA repository

**On WSL2, use the `wsl-ubuntu` repo.** Same for both Ubuntu versions, and it deliberately
excludes the driver packages you must not install inside WSL.

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Native Linux instead (value **E** said not WSL):

```bash
# Ubuntu 24.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```

**Correct output:** keyring unpacked; `apt update` lists a `developer.download.nvidia.com`
source with no GPG errors.

- **`NO_PUBKEY` →** the keyring did not install. Re-run `dpkg -i`.
- **404 →** wrong distro string. Re-check **D**.

### 3.3 Install the toolkit only

```bash
sudo apt install -y cuda-toolkit-12-8
```

**Install `cuda-toolkit-12-8`, never `cuda` or `cuda-drivers`.**

**Timing: 8–15 minutes.**

### 3.4 Put nvcc on PATH

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version | grep release
ls -d /usr/local/cuda-12.8
```

**Correct output:** `Cuda compilation tools, release 12.8, V12.8.61` and the directory.

- **`nvcc: command not found` →** check `ls /usr/local/` for the real name, adjust
  `CUDA_HOME`.
- **Reports a version below 12.8 →** an older toolkit is shadowing it. Confirm with
  `which nvcc`; it must be under `/usr/local/cuda-12.8/bin`.

### ✅ Definition of done — Phase 3

`nvcc --version` reports 12.8 and `$CUDA_HOME` is `/usr/local/cuda-12.8`.

---

## Phase 4 — Docker inside WSL

**10 minutes.**

### 4.1 Install

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

Apply the group change — open a **new** WSL terminal, or:

```bash
newgrp docker
```

```bash
docker --version
docker ps
```

**Correct output:** a version string, then an empty container table with headers.

- **`permission denied … docker.sock` →** close every WSL terminal, `wsl --shutdown` from
  Windows, reopen.

### 4.2 Start the daemon — depends on the init system

```bash
ps -p 1 -o comm=
```

- **`systemd` →**
  ```bash
  sudo systemctl enable --now docker
  systemctl is-active docker
  ```
  **Correct output:** `active`.

- **`init` →** systemd is not enabled.

  **Option A — manual, fastest today:**
  ```bash
  sudo service docker start
  sudo service docker status
  ```
  **Correct output:** `Docker is running`. Re-run after every WSL restart.

  **Option B — enable systemd (required if you want the TRELLIS.2 systemd unit in
  Phase 13.7):**
  ```bash
  sudo tee /etc/wsl.conf >/dev/null <<'EOF'
  [boot]
  systemd=true
  EOF
  ```
  Then from **Windows** PowerShell `wsl --shutdown`, reopen, and confirm
  `ps -p 1 -o comm=` now says `systemd`.

  > **In this variant, take Option B.** Phase 13.7 installs a systemd unit. Without systemd
  > you must run TRELLIS.2 in a foreground terminal instead — workable, but it dies when the
  > terminal closes.

### ✅ Definition of done — Phase 4

`docker ps` works without sudo; the daemon is running.

---

## Phase 5 — GPU passthrough into Docker

**8 minutes. The step that most often fails on WSL.**

### 5.1 Install the container toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
nvidia-ctk --version
```

**Correct output:** `NVIDIA Container Toolkit CLI version 1.x`.

### 5.2 Configure the Docker runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

**Correct output:** a line naming `/etc/docker/daemon.json`.

`docker-compose.yml` requests the GPU via `deploy.resources.reservations.devices` with
`driver: nvidia`, which uses this runtime.

### 5.3 Generate the CDI spec

On WSL the GPU libraries live in `/usr/lib/wsl/lib`; the CDI spec records that explicitly.

```bash
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
```

**Correct output:** device names including `nvidia.com/gpu=all` and `nvidia.com/gpu=0`.

- **Errors about no devices →** `nvidia-smi` is not working in this shell. Phase 2.1.

### 5.4 Restart Docker and test

```bash
sudo systemctl restart docker    # or: sudo service docker restart
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

**Timing:** ~90 seconds first time (~200 MB pull).

**Correct output:** the host GPU table, printed from inside the container.

| Symptom | Do this |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | 5.2 did not take. Re-run, restart Docker, retry. |
| `nvidia-container-cli: initialization error` | Try `docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`. If that works but `--gpus all` does not, add `"features": {"cdi": true}` to `/etc/docker/daemon.json` and restart Docker. |
| `unknown or invalid runtime name: nvidia` | `daemon.json` not written. `cat` it, re-run 5.2. |
| Works with sudo only | Group change not applied. `wsl --shutdown`. |

### ✅ Definition of done — Phase 5

`docker run --rm --gpus all … nvidia-smi` prints your GPU. **Do not start Phase 10 until this
passes.**

---

## Phase 6 — Miniconda

**5 minutes. Required in this variant** — TRELLIS.2 needs its own environment with torch
2.10, which must not collide with anything else.

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
source ~/.bashrc
conda info --base
```

**Correct output:** `conda info --base` prints `/home/<you>/miniconda3`. **That is value L.**

`-b` is batch mode: no prompts, accepts the licence, no PATH surprises beyond what
`conda init` writes.

**Phase 13.7 needs `L + /etc/profile.d/conda.sh`.** Confirm that file exists now:

```bash
ls -l "$(conda info --base)/etc/profile.d/conda.sh"
```

**Correct output:** the file listed. Write the full path down.

- **`conda: command not found` after sourcing →** open a new terminal, or
  `source "$HOME/miniconda3/etc/profile.d/conda.sh"`.

### ✅ Definition of done — Phase 6

`conda info --base` resolves and `conda.sh` exists at the recorded path.

---

## Phase 7 — Repository, configuration, and the gating check

**15 minutes. Do 7.1 first thing in the morning — it may need another human.**

### 7.1 Verify gated access BEFORE anything else

```bash
pip install -U huggingface_hub 2>/dev/null || pip install -U huggingface_hub --break-system-packages
huggingface-cli login
```

Paste a token from the account that accepted the terms. Then verify each of the four
individually — this fails fast and tells you exactly which one is missing:

```bash
for r in facebook/sam3 facebook/sam3.1 facebook/sam-3d-objects \
         facebook/dinov3-vitl16-pretrain-lvd1689m; do
  python3 - "$r" <<'PY'
import sys
from huggingface_hub import HfApi
r = sys.argv[1]
try:
    HfApi().model_info(r)
    print(f"OK      {r}")
except Exception as e:
    print(f"BLOCKED {r}  -> {type(e).__name__}")
PY
done
```

**Correct output:** four `OK` lines.

- **All four `OK` →** continue. This is the good day.
- **Any `BLOCKED` →** the token is valid but that repo's terms are **not accepted on this
  account**. Open the URL, log in as that account, accept:
  - `https://huggingface.co/facebook/sam3`
  - `https://huggingface.co/facebook/sam3.1`
  - `https://huggingface.co/facebook/sam-3d-objects`
  - `https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m`
  ~5 minutes if the account is yours. **If it is not, escalate now** — this is the one
  blocker that cannot be fixed in the terminal.
- **Still blocked and nobody can accept today →** switch to
  [`RUNBOOK_UNGATED.md`](RUNBOOK_UNGATED.md) and run the ungated path. Its final section
  covers adding these later, and **none of them needs a rebuild.**

### 7.2 Get the code

```bash
cd <where you put it>
export A="$(pwd)"
echo "$A"
git log --oneline -1 2>/dev/null || echo "(no .git — fine, but you cannot verify the version)"
```

> **Keep the repo on the WSL filesystem** (`/home/...`), **not** `/mnt/c/`. Model loading
> across the Windows bridge is dramatically slower.

### 7.3 Create the env file

```bash
cd "$A"
cp docker/.env.example docker/.env
```

### 7.4 The eight variables

| Variable | Default | Today |
|---|---|---|
| `WEIGHTS_HOST_DIR` | `../weights` | keep |
| `GALLERY_HOST_DIR` | `../gallery` | keep, or a bigger path |
| `SAM3D_HF_DIR` | `../weights/sam3d-objects-hf` | **set in 7.6 after the download** |
| `HF_CACHE_DIR` | `$HOME/.cache/huggingface` | keep |
| `TORCH_CACHE_DIR` | `$HOME/.cache/torch` | keep |
| `HY3DGEN_CACHE_DIR` | `$HOME/.cache/hy3dgen` | keep |
| `U2NET_CACHE_DIR` | `$HOME/.u2net` | keep |
| `TRELLIS2_SERVICE_URL` | `http://host.docker.internal:8710` | **keep** |

Create the mount targets so Docker does not create them root-owned:

```bash
mkdir -p "$A/weights" "$A/gallery" \
         "$HOME/.cache/huggingface" "$HOME/.cache/torch" \
         "$HOME/.cache/hy3dgen" "$HOME/.u2net"
```

### 7.5 Download everything

```bash
cd "$A"
python3 scripts/download_models.py --check
python3 scripts/download_models.py --model all
```

**Correct output:** a `GATED REPOSITORIES` block, then `Token found. Proceeding.`, then
`[GET ]`/`[OK  ]` lines for all ten families.

**~73 GB. 25 minutes on a fast link, 3+ hours on a slow one. Do not kill it** — watch the
size counter, not the clock.

| Symptom | Do this |
|---|---|
| `NO TOKEN FOUND` | 7.1 did not complete. `huggingface-cli login` again. |
| 401 / 403 on a `facebook/*` repo | Terms not accepted on this account. Back to 7.1. |
| Interrupted | Re-run; completed files are skipped. |
| `No space left on device` | Phase 2.2 — grow the vhdx from Windows. |

### 7.6 Point SAM3D_HF_DIR at the real weights

Its default has **never been exercised**. Find where the weights actually landed:

```bash
find "$HOME/.cache/huggingface" -name "pipeline.yaml" -path "*sam*3d*" 2>/dev/null
```

**Correct output:** one path ending in `pipeline.yaml`.

Set `SAM3D_HF_DIR` to the directory **containing** it:

```bash
cd "$A"
sed -i "s|^SAM3D_HF_DIR=.*|SAM3D_HF_DIR=/full/path/from/find|" docker/.env
grep SAM3D_HF_DIR docker/.env
```

- **`find` returns nothing →** leave the default. SAM 3D will fail at first use with a
  `FileNotFoundError` that **lists every path it searched** — pick the right one from that
  list and set it then.

### 7.7 Verify what resolves

```bash
cd "$A/docker" && docker compose config | grep -A1 "source:" ; cd "$A"
```

**Correct output:** **seven** `source:` entries, all absolute paths that exist. Any literal
`${` is a broken variable.

### ✅ Definition of done — Phase 7

Four gated repos `OK`. All ten families downloaded. Seven mounts resolve.

---

## Phase 8 — RealESRGAN, before the build

**1 minute.**

Not on HuggingFace, not in git, and `COPY`d into the image — so it **must exist before
Phase 9**.

```bash
cd "$A"
mkdir -p backend/engines/hunyuan/hy3dpaint/ckpt
wget -O backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
ls -lh backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth
```

**Correct output:** ~67 MB.

- **A few KB →** an HTML error page. Delete and retry.

### ✅ Definition of done — Phase 8

The file exists and is ~67 MB.

---

## Phase 9 — Build and start the container

**~10 minutes native. On WSL allow 45. Do not kill before 60 minutes.**

```bash
cd "$A/docker"
docker compose up -d --build 2>&1 | tee ~/htx-build-$(date +%Y%m%d-%H%M).log
```

**These lines must appear:**

```
RESOLVED nvdiffrast 253ac4fcea7de5f396371124af597e6cc957bfae
RESOLVED diffoctreerast b09c20b84ec3aace4729e6e18a613112320eca3a
RESOLVED mip-splatting dda02ab5ecf45d6edb8c540d9bb65c7e451345a9
```

and near the end:

```
utils3d.pt shim installed
UniDepth OK
```

**Correct output at the end:** `Container htx-3d  Started`.

**Expected noise — ignore:** the `gradio 6.24.0 requires huggingface-hub<2.0` conflict.
Long-standing, inert, nothing imports gradio. Multi-minute silences in `pytorch3d` and the
extensions are normal.

| Symptom | Do this |
|---|---|
| Killed / OOM during a compile | WSL RAM. Phase 2.3, `wsl --shutdown`, retry. **Most likely WSL failure.** |
| `no space left on device` | `docker system prune -af`, then Phase 2.2. |
| Network timeout in a clone | Re-run; completed layers are reused. |
| `FileNotFoundError` … `RealESRGAN_x4plus.pth` | Phase 8 skipped. Do it, rebuild. |
| CUDA extension compile error | Capture the last 50 lines. **Stop and note it.** |

> **`docker/Dockerfile` already sets `ENV MAX_JOBS=8`** (line 89, before every extension
> build). That caps nvcc parallelism, which is what prevents the OOM that otherwise lands
> 20-30 minutes into the build. If it still OOMs after raising WSL memory, lower it to `4`
> and rebuild. **Do not raise it.**

```bash
docker compose logs -f
```

**Correct output:** ends with `Uvicorn running on http://0.0.0.0:8000`.

### ✅ Definition of done — Phase 9

`docker compose ps` shows `htx-3d` as `Up`; logs show `Uvicorn running`.

---

## Phase 10 — Smoke test

**2 minutes.**

```bash
curl -s localhost:8000/api/health | python3 -m json.tool
```

**Correct output:**

```json
{
    "status": "ok",
    "gpu": {
        "available": true,
        "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q",
        "compute_capability": "12.0",
        "is_blackwell": true
    },
    "models_loaded": [],
    "engines_registered": ["trellis", "hunyuan", "sam3d", "trellis2"],
    "active_engine": null,
    "queue_size": 0
}
```

Check: `status: ok`, `available: true`, four engines. `models_loaded` empty is **correct** —
engines lazy-load.

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/
```

**Correct output:** `200`. Then open **http://localhost:8000** from Windows — WSL2 forwards
localhost automatically. If it does not resolve:

```bash
hostname -I | awk '{print $1}'
```

and browse `http://<that-ip>:8000`.

| Symptom | Do this |
|---|---|
| `Connection refused` | `docker compose ps`, then `docker compose logs htx-3d`. |
| `"available": false` | Passthrough lost. Phase 5.4, then `docker compose up -d`. |
| Warning naming `host.docker.internal:8710` | **Expected until Phase 13.** Non-fatal. |

### ✅ Definition of done — Phase 10

`status: ok`, GPU available, four engines, UI returns 200.

**This is the point at which you have something to show.**

---

## Phase 11 — Prove the container engines

**25–40 minutes. First use of each engine loads a model, so first runs are slow.**

Watch swaps in a second terminal:

```bash
cd "$A/docker" && docker compose logs -f | grep -iE 'loading engine|unloading|registered|reachable'
```

### 11.1 TRELLIS — the one that must work

Upload a photograph, engine **TRELLIS**, generate.

**Correct:** a mesh appears. First run ~1–2 minutes including load.

- **Multi-billion-GB OOM** → extensions built without your architecture. **Stop and note it.**
- **Works →** this is your fallback demo.

### 11.2 SAM 3 interactive segmentation

Upload a cluttered photo. **Segment Object.** Click a point on the target.

**Correct:** a blue mask overlay, a **Mask %** figure, and a **Use Mask** button.

- **401/403 in the logs →** gated terms. Phase 7.1.
- **Works →** confirm the whole flow: refine with a second point, **Use Mask**, then generate.
  The cut-out is passed to the engine as `segmented_image_path`.

### 11.3 SAM 3D Objects

Engine **SAM 3D Objects**, generate. First load is slow — 12 GB of weights.

- **`FileNotFoundError` listing search paths →** `SAM3D_HF_DIR` is wrong. **The error lists
  every path it tried.** Pick the one containing `pipeline.yaml`, set it in `docker/.env`,
  then:
  ```bash
  cd "$A/docker" && docker compose up -d
  ```
  **`up -d`, not `restart`** — bind mounts are fixed at container creation, so the container
  must be recreated. **No rebuild.** ~30 seconds.

### 11.4 Hunyuan3D

Engine **Hunyuan3D**, generate. Slowest; allow ~2 minutes after load.

- **`FileNotFoundError` naming `RealESRGAN_x4plus.pth` →** Phase 8 skipped. Fetch it and
  **rebuild** (~15 min from cache).

### 11.5 Auto-scale

After any generation, the panel should show dimensions in metres with a confidence badge.

### 11.6 Logo bake, gallery, export

Bake a logo onto a result. Open the **Gallery** tab. Export GLB, then OBJ.

- **Gallery empty despite successful generations →** check the browser Network tab for
  `/api/gallery?page=1&per_page=500`. `Gallery.tsx` swallows fetch errors silently and shows
  the same "No generations yet" state either way.

### ✅ Definition of done — Phase 11

TRELLIS, SAM 3, SAM 3D and Hunyuan all produce output. Auto-scale reports metres. Gallery
lists results. Export downloads.

**Stop here for day one. Phase 13 is a separate day.**

---

## Phase 12 — Checkpoint before TRELLIS.2

**Do not start Phase 13 unless all of these are true:**

```bash
nvcc --version | grep release                 # 12.8
conda info --base                             # value L
free -g                                       # 24 GB+
df -h / | tail -1                             # 40 GB+ still free
curl -s localhost:8000/api/health | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])"
```

- All pass, **and it is before 14:00** → Phase 13.
- Anything fails, **or it is after 14:00** → **stop.** The container stack is a complete,
  defensible handover on its own. TRELLIS.2 is an increment.

---

## Phase 13 — TRELLIS.2 host service

**2–4 hours. Highest risk in the runbook. Never been run from this repo in its vendored
form** — `services/trellis2/README.md` §7 lists four things that have never executed: the
clone path in the build script, the acceptance test on FA2, the parameterised launcher, and
the systemd unit. Expect to debug.

TRELLIS.2 cannot run in the container — it needs the host's torch 2.10 / FlashAttention-2
stack for sm_120.

### 13.1 Clone at the pin

```bash
export TRELLIS2_ROOT=/opt/trellis2
sudo mkdir -p "$TRELLIS2_ROOT" && sudo chown "$(whoami)" "$TRELLIS2_ROOT"
git clone https://github.com/microsoft/TRELLIS.2 "$TRELLIS2_ROOT"
git -C "$TRELLIS2_ROOT" checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C "$TRELLIS2_ROOT" submodule update --init --recursive
git -C "$TRELLIS2_ROOT" log --oneline -1
```

**Correct output:**
`75fbf018 Merge pull request #166 from microsoft/copilot/fix-failing-github-actions-job`

**Timing:** 3–5 minutes.

### 13.2 Apply the patch

```bash
cd "$TRELLIS2_ROOT"
git apply --check "$A/services/trellis2/patches/image_feature_extractor.patch" && echo "PATCH APPLIES"
git apply         "$A/services/trellis2/patches/image_feature_extractor.patch"
```

**Correct output:** `PATCH APPLIES`, then silence.

- **`--check` fails →** the tree is not at the pinned commit. Redo 13.1. **Do not force it.**

The patch makes `DinoV3FeatureExtractor` tolerate the transformers 5.x block layout. Without
it, every generation raises `AttributeError`.

### 13.3 Create the environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n trellis2 python=3.12 -y
conda activate trellis2
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.13.1
pip install fastapi uvicorn python-multipart pillow
python -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"
```

**Correct output:** `2.10.0+cu128` and an arch list **containing `sm_120`**.

**Timing:** 10–15 minutes (torch is ~2.5 GB).

- **`sm_120` absent →** wrong wheel. Re-run the torch install with that exact index URL.

### 13.4 flash_attn — MAX_JOBS=2, not more

```bash
MAX_JOBS=2 pip install flash_attn==2.8.3.post1 --no-build-isolation
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
```

**Correct output:** `flash_attn 2.8.3.post1`.

**Timing:** 2 minutes if a prebuilt wheel matches; **up to 90 minutes** if it compiles from
source. Do not kill it.

> **`MAX_JOBS=2` is deliberate. The build OOM'd at 3 jobs on 30 GB.** Do not use `$(nproc)`.

- **Killed / OOM →** you exceeded 2 jobs, or WSL memory is below 24 GB. Phase 2.3, then retry
  with `MAX_JOBS=1`.
- **FA2 is correct, FA3 is wrong.** `flash_attn_3` was tried on sm_120 and found wrong. Do
  not substitute it.

### 13.5 Confirm transformers survived

```bash
pip show transformers | grep Version
```

**Correct output:** `Version: 5.13.1`.

- **Anything else →** something upgraded it as a transitive. `pip install transformers==5.13.1`
  again. **It is a co-pin, not a suggestion** — the patched extractor is verified against
  5.13.1 and no other version.

### 13.6 Build the CUDA extensions — MAX_JOBS=4

```bash
cp "$A/services/trellis2/build_extensions_blackwell.sh" "$TRELLIS2_ROOT/"
CUDA_HOME=/usr/local/cuda-12.8 \
MAX_JOBS=4 \
TORCH_CUDA_ARCH_LIST=12.0 \
TRELLIS2_ROOT="$TRELLIS2_ROOT" \
bash "$TRELLIS2_ROOT/build_extensions_blackwell.sh" 2>&1 | tee ~/trellis2-build.log
```

**Timing: 30–90 minutes. Do not kill it.**

**Correct output — the final block must show all six:**

```
OK   o_voxel
OK   cumesh
OK   nvdiffrast.torch
OK   nvdiffrec_render.light
OK   flex_gemm
OK   flash_attn_interface
```

> **`MAX_JOBS=4` is deliberate. Do not use `$(nproc)`.**

| Symptom | Do this |
|---|---|
| Any module missing or `FAIL` | TRELLIS.2 will not work. **Stop, note which module, keep the log.** The container stack is unaffected. |
| Killed / OOM | Retry with `MAX_JOBS=2`. If it still dies, raise WSL memory. |
| `nvcc: not found` | Phase 3 incomplete, or you are in a shell without the PATH export. `source ~/.bashrc`. |

### 13.7 Install the service and the systemd unit

```bash
mkdir -p "$TRELLIS2_ROOT/service"
cp "$A/services/trellis2/trellis2_service.py" "$TRELLIS2_ROOT/service/"
cp "$A/services/trellis2/run_service.sh"      "$TRELLIS2_ROOT/service/"
chmod +x "$TRELLIS2_ROOT/service/run_service.sh"
```

systemd does **not** expand variables in `User=` or `ExecStart=`. Use real values:

```bash
sudo cp "$A/services/trellis2/trellis2-service.service" /etc/systemd/system/
sudo sed -i "s|^User=.*|User=$(whoami)|" /etc/systemd/system/trellis2-service.service
sudo sed -i "s|^ExecStart=.*|ExecStart=${TRELLIS2_ROOT}/service/run_service.sh|" \
  /etc/systemd/system/trellis2-service.service
grep -E "^(User|ExecStart)=" /etc/systemd/system/trellis2-service.service
```

**Correct output:** your real username and a real path — no placeholder `trellis2` user
unless that user exists.

Environment file. **`TRELLIS2_CONDA_SH` is value L + `/etc/profile.d/conda.sh`:**

```bash
sudo tee /etc/default/trellis2-service >/dev/null <<EOF
TRELLIS2_ROOT=${TRELLIS2_ROOT}
TRELLIS2_CONDA_SH=$(conda info --base)/etc/profile.d/conda.sh
TRELLIS2_CONDA_ENV=trellis2
TRELLIS2_SERVICE_PORT=8710
HF_HOME=${HOME}/.cache/huggingface
EOF
cat /etc/default/trellis2-service
```

**Correct output:** five lines, all absolute paths, no `${` left.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now trellis2-service
systemctl is-active trellis2-service
curl -s localhost:8710/health
```

**Correct output:** `active`, then `{"status":"ok","model_loaded":false}`.

**`model_loaded:false` is correct** — the 4B model lazy-loads on the first generation, so the
service starts instantly.

| Symptom | Do this |
|---|---|
| `inactive` / `failed` | `journalctl -u trellis2-service -n 50 --no-pager`. Usually a wrong `TRELLIS2_CONDA_SH`, or `User=` cannot read `$TRELLIS2_ROOT`. |
| `System has not been booted with systemd` | Phase 4.2 Option B was skipped. Run it in the foreground instead: `TRELLIS2_ROOT=/opt/trellis2 bash /opt/trellis2/service/run_service.sh` — it dies when the terminal closes. |

### 13.8 Connect the container

```bash
cd "$A/docker" && docker compose restart htx-3d
sleep 20 && docker compose logs --tail 40 htx-3d | grep -i trellis2
```

**Correct output:** no warning about an unreachable TRELLIS.2.

> The container needs **no rebuild** for this. `TrellisTwoEngine.load()` only probes the
> URL — there are no local weights (`backend/app/services/trellis2.py:63-67`). The restart
> just clears the startup warning.

Generate once with engine **TRELLIS.2**. First run loads the 4B model — allow **3–5 minutes**.

**Pass criteria — structural, not timing** (from `services/trellis2/README.md` §8):
- A recognisable object in the viewer.
- GLB in the **tens of MB**, not kilobytes.
- Face count in the **millions**.
- All six modules reported `OK` in 13.6.

**Do not fail the deployment for slow generation.** The reference timings are from a 575 W
RTX 5090; the target is 300 W.

### ✅ Definition of done — Phase 13

`systemctl is-active` → `active`; `/health` → ok; one TRELLIS.2 generation completes.

---

## Cut-off rule

**Check the clock at 14:00.**

| State at 14:00 | Do this |
|---|---|
| Phase 11 done | Phase 13 **only if** Phase 12 passes and it is before 14:00. Otherwise stop. |
| Phase 10 done, Phase 11 in progress | Finish Phase 11. **Do not start Phase 13 today.** |
| Phase 9 still building | Let it finish. Nothing else today. |
| Phase 7 blocked on gated access | Switch to `RUNBOOK_UNGATED.md`. Record the blocker in writing. |

**Hard rule: do not begin Phase 13 after 14:00.** A half-installed TRELLIS.2 with a failed
systemd unit is worse than none — it leaves a service that fails on every boot. If you
started and it is not working by 16:00:

```bash
sudo systemctl disable --now trellis2-service
```

Leave the container stack running.

**Last 30 minutes, whatever state you are in:**

```bash
cd "$A/docker" && docker compose ps > ~/htx-handover-ps.txt
curl -s localhost:8000/api/health | python3 -m json.tool > ~/htx-handover-health.json
systemctl is-active trellis2-service > ~/htx-handover-trellis2.txt 2>&1
cp "$A/docker/.env" ~/htx-handover-env.txt
docker images | grep htx-3d > ~/htx-handover-images.txt
free -g > ~/htx-handover-mem.txt; df -h / >> ~/htx-handover-mem.txt
```

Write down: which engines generated, which failed and with what error, and anything to
escalate.

---

## Minimum viable handover

Cut in this order — top goes first.

| # | Cut | Saves | You lose | Still works |
|---|---|---|---|---|
| 1 | **Phase 13 — TRELLIS.2** | 2–4 h | The highest-scoring engine | Everything else. Container logs a non-fatal warning. |
| 2 | **Phases 3 + 6** (CUDA toolkit, Miniconda) | 20 min | Only prerequisites for Phase 13 | The whole container stack |
| 3 | **SAM 3D Objects** (skip 7.6, 11.3) | 20 min | One engine | TRELLIS, Hunyuan, SAM 3 |
| 4 | **Hunyuan** (skip Phase 8) | 1 min, avoids a rebuild | Hunyuan texturing — treat the engine as gone | TRELLIS + SAM 3 + auto-scale |

**Never cut:** TRELLIS weights, the build, the health check.

**Absolute minimum path**, assuming Phases 1, 2, 4, 5 passed:

```bash
cd "$A"
cp docker/.env.example docker/.env
python3 scripts/download_models.py --model trellis_image
mkdir -p backend/engines/hunyuan/hy3dpaint/ckpt
wget -O backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
cd docker && docker compose up -d --build
curl -s localhost:8000/api/health | python3 -m json.tool
```

Then one TRELLIS generation. **That is a defensible handover.**

---

## Every value in one place

### Ports

| Port | Service | Bound |
|---|---|---|
| `8000` | HTX-3D API + UI | container → host |
| `8710` | TRELLIS.2 host service | host `0.0.0.0` |

### Build parallelism — hardcoded, do not use `$(nproc)`

| Where | Value | Why |
|---|---|---|
| `flash_attn` (13.4) | **`MAX_JOBS=2`** | OOM'd at 3 jobs on 30 GB |
| TRELLIS.2 CUDA extensions (13.6) | **`MAX_JOBS=4`** | |
| Container build | **`MAX_JOBS=8`** | Set in `docker/Dockerfile:89`. Lower to 4 if it OOMs; never raise |

### Container versions (fixed in `docker/Dockerfile`)

| Item | Value |
|---|---|
| Base image | `nvidia/cuda:12.8.0-devel-ubuntu24.04` |
| Python | 3.12 |
| torch | `2.7.0` |
| torch index URL | `https://download.pytorch.org/whl/cu128` |
| transformers | `>=4.35.0,<4.50` |
| huggingface_hub | `>=0.23,<1.0` |
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.6;8.9;10.0;12.0` |
| spconv | `spconv-cu126==2.3.8` |
| kaolin | `0.18.0` |

### TRELLIS.2 host service

| Item | Value |
|---|---|
| Upstream commit | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| transformers | `5.13.1` — **co-pin, do not float** |
| torch | `2.10.0` |
| torch index URL | `https://download.pytorch.org/whl/cu128` |
| flash_attn | `2.8.3.post1` (FA2 — **not** flash_attn_3) |
| Python | 3.12 |
| `TORCH_CUDA_ARCH_LIST` | `12.0` |
| `ATTN_BACKEND` | `flash_attn` |
| `SPARSE_ATTN_BACKEND` | `flash_attn` |
| `SPARSE_CONV_BACKEND` | `flex_gemm` |
| Six modules | `o_voxel`, `cumesh`, `nvdiffrast.torch`, `nvdiffrec_render.light`, `flex_gemm`, `flash_attn_interface` |

### Pinned commits (in `docker/Dockerfile`)

| Package | Commit | Reproduces the benchmarked stack? |
|---|---|---|
| `pytorch3d` | `b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881` | **Yes** |
| `moge` | `07444410f1e33f402353b99d6ccd26bd31e469e8` | **Yes** |
| `nvdiffrast` | `253ac4fcea7de5f396371124af597e6cc957bfae` | No — observed 2026-08-14 only |
| `diffoctreerast` | `b09c20b84ec3aace4729e6e18a613112320eca3a` | No — observed only |
| `mip-splatting` | `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9` | No — observed only |
| `utils3d` | `9a4eb15e4021b67b12c460c7057d642626897ec8` | Pinned throughout |
| `unidepth` | `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` | Pinned throughout |

### Model repositories — all ten

| Family | Repo ID | Gated |
|---|---|---|
| TRELLIS image | `JeffreyXiang/TRELLIS-image-large` | No |
| TRELLIS text | `JeffreyXiang/TRELLIS-text-large` | No |
| Hunyuan3D | `tencent/Hunyuan3D-2.1` | No |
| UniDepth | `lpiccinelli/unidepth-v2-vits14` | No |
| CLIP | `openai/clip-vit-base-patch32` | No |
| TRELLIS.2 | `microsoft/TRELLIS.2-4B` | No |
| SAM 3 | `facebook/sam3` | **Yes** |
| SAM 3.1 | `facebook/sam3.1` | **Yes** |
| SAM 3D | `facebook/sam-3d-objects` | **Yes** |
| DINOv3 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | **Yes** |

### Direct downloads

| File | URL |
|---|---|
| RealESRGAN | `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth` |
| u2net (air-gapped) | `https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx` |

### CUDA repo URLs

| Target | URL |
|---|---|
| **WSL2 (use this)** | `https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb` |
| Ubuntu 24.04 native | `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb` |
| Ubuntu 22.04 native | `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb` |

Package: `cuda-toolkit-12-8`. **Never `cuda` or `cuda-drivers` inside WSL.**

### Miniconda

`https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
Install with `-b -p "$HOME/miniconda3"`.

### Storage budget

| Item | Size |
|---|---|
| Docker image | 35.2 GB |
| HuggingFace cache (all ten families) | 40 GB |
| Hunyuan3D cache | 14 GB |
| SAM 3D Objects weights | 12 GB |
| TRELLIS weights | 5.3 GB |
| torch hub | 1.4 GB |
| TRELLIS.2 conda env + extensions | ~8 GB |
| Gallery | unbounded — 14 GB per ~800 generations |

---

## Things that cannot be fixed on site

1. **Gated terms not accepted and the account holder unavailable.** No workaround. Fall back
   to `RUNBOOK_UNGATED.md`.
2. **A CUDA extension fails to compile** (container or TRELLIS.2). Keep the log.
3. **Multi-billion-GB OOM at generation.** Extensions lack your architecture.
4. **`nvidia-smi` never works inside WSL.** A Windows-side driver problem.
5. **No sudo.** Needs a Windows administrator.
6. **Compute capability is not `12.0`.** Every Blackwell assumption in Phase 13 is void.

## Licence constraints — do not let anyone quietly ignore these

| Component | Constraint |
|---|---|
| **UniDepth** | **CC BY-NC 4.0 — non-commercial.** This powers auto-scale. Any commercial or non-research deployment needs legal review or a replacement depth backbone. |
| **Hunyuan3D 2.1** | Territory-limited: excludes EU, UK, South Korea — model *and* output. **Singapore is inside the Territory.** |
| **SAM 3 / SAM 3D** | Trade controls: no military or warfare purposes, nuclear, or espionage. HTX is home-affairs, so the ordinary reading is satisfied — but the wording is broad and deserves a deliberate read. |
| **TRELLIS.2-4B weights** | Code is MIT; **the weights carry their own terms and were never verified.** Confirm before operational use. |
| **DINOv3** | Gated; terms accepted per account and not transferable by copying a token. |

Full audit: [`reference/vendoring.md`](reference/vendoring.md).
