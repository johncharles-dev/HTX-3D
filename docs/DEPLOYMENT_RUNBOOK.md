# HTX-3D — Deployment Runbook

**One person, one terminal, one day.** Follow top to bottom. Do not skip Phase 0.

Target machine: **RTX PRO 6000 Blackwell Max-Q** (GB202, sm_120, 96 GB, 300 W).

There is **no `TROUBLESHOOTING.md`** in this repository. Every fix is inline here.

Companion docs, only if you have time to read them: [`SETUP_GUIDE.md`](SETUP_GUIDE.md),
[`MACHINE_REQUIREMENTS.md`](MACHINE_REQUIREMENTS.md),
[`../services/trellis2/README.md`](../services/trellis2/README.md).

---

## Read this before you start

Three things that will cost you the day if you learn them at 15:00.

1. **Gated weights need an account, not a token.** Four repos require a HuggingFace
   account that has *accepted each licence*. Acceptance is per account and does **not**
   transfer with copied files or a copied token. If nobody at HTX has accepted them,
   Phase 3 stops dead and no amount of terminal work fixes it. **Do Phase 3's gating check
   first thing in the morning**, before anything else, because it may need another human.

2. **TRELLIS.2 has never been installed from this repo in its vendored form.** Its README
   §7 says so explicitly. It is the highest-scoring engine and the highest-risk phase.
   It is Phase 8, it is optional, and it is the first thing you cut.

3. **Timings in this repo were measured on an RTX 5090 at 575 W.** Your card is 300 W.
   Generation times will be slower. Do not treat any recorded `gen_s` as a threshold.

---

## Phase 0 — Collect these first

**Two minutes. Do not skip.** Every later phase references these.

Paste this whole block. Write the answers in the table below.

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"   # be at the repo root first
export A="$(pwd)"          # every later phase uses $A. Re-export in any new terminal.
echo "== A repo root      "; echo "$A"
echo "== B user           "; whoami
echo "== C home           "; echo $HOME
echo "== D cores          "; nproc
echo "== E gpu            "; nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
echo "== F cuda toolkit   "; ls -d /usr/local/cuda-12.8 2>/dev/null || echo "MISSING /usr/local/cuda-12.8"
echo "== G nvcc           "; nvcc --version 2>/dev/null | grep release || echo "nvcc NOT ON PATH"
echo "== H conda base     "; conda info --base 2>/dev/null || echo "conda NOT INSTALLED"
echo "== I docker         "; docker --version 2>/dev/null || echo "docker NOT INSTALLED"
echo "== J compose        "; docker compose version 2>/dev/null || echo "compose v2 MISSING"
echo "== K disk here      "; df -h . | tail -1
echo "== L block devices  "; lsblk -f | grep -vE "loop|^$"
echo "== M big mounts     "; findmnt -t ext4,xfs,btrfs -o TARGET,SIZE,AVAIL --noheadings
echo "== N hf cache       "; du -sh $HOME/.cache/huggingface 2>/dev/null || echo "absent (fine)"
```

Write these down:

| ID | Value | Example on a typical machine | Used in |
|---|---|---|---|
| **A** repo root | `________________` | `/home/htx/HTX-3D` | Every `cd`. Phase 2, 5, 8 |
| **B** user | `________________` | `htx` | Phase 8.6 systemd `User=` |
| **C** home | `________________` | `/home/htx` | Phase 2 cache paths |
| **D** cores | `________________` | `32` | Phase 8.4 `MAX_JOBS` (auto via `$(nproc)`) |
| **E** GPU | `________________` | `NVIDIA RTX PRO 6000 Blackwell Max-Q, 12.0, 97887 MiB, 570.86.15` | Phase 1 gate |
| **F** CUDA dir | `________________` | `/usr/local/cuda-12.8` | Phase 8.4 `CUDA_HOME` |
| **G** nvcc | `________________` | `Cuda compilation tools, release 12.8, V12.8.61` | Phase 1 gate |
| **H** conda base | `________________` | `/home/htx/miniconda3` | Phase 8.6 `TRELLIS2_CONDA_SH` = this **+ `/etc/profile.d/conda.sh`** |
| **I/J** docker | `________________` | `Docker version 27.x` / `Docker Compose version v5.0.2` | Phase 1 gate |
| **K** free disk | `________________` | `1.8T  400G avail` | Phase 1 gate — need **150 GB** |
| **L/M** data volume | `________________` | `/mnt/data  3.6T  3.1T` | Phase 2 `GALLERY_HOST_DIR` |

> **`export A` matters.** Phase 8 pastes `$A/services/...` into commands. If you open a new
> terminal, or `sudo` into another shell, run `export A=/your/repo/root` again first.

**Compute capability (E) must read `12.0`.** If it reads anything else, `TORCH_CUDA_ARCH_LIST`
assumptions in Phase 8 are wrong — note it and continue with the container only.

---

## Phase 1 — Preflight gates

**5 minutes if the machine is prepared. Up to 40 minutes plus a reboot if not.**

### 1.1 Driver

```bash
nvidia-smi
```

**Correct output:** a table, driver version **570 or higher**, your GPU listed, no error.

- **Driver ≥ 570 and GPU listed →** continue.
- **`command not found` or "no devices" →** driver missing. Install the **open** variant:
  ```bash
  sudo apt update && sudo apt install -y nvidia-driver-570-open && sudo reboot
  ```
  Blackwell **requires** `-open`. The proprietary build fails. Reboot costs ~3 minutes.
- **Driver below 570 →** same install command. Blackwell will not work below 570.

### 1.2 CUDA toolkit

```bash
nvcc --version | grep release
```

**Correct output:** `Cuda compilation tools, release 12.8, V12.8.61` (V-number may differ).

- **Reports 12.8 →** continue.
- **Reports below 12.8, or not on PATH →** the **container does not care** — it ships its own
  CUDA 12.8 from `nvidia/cuda:12.8.0-devel-ubuntu24.04`. Continue to Phase 2.
  Only **Phase 8 (TRELLIS.2)** needs a host toolkit. If you will do Phase 8, install it, or
  note "host CUDA < 12.8, TRELLIS.2 deferred" and plan to cut Phase 8.

### 1.3 Docker + GPU passthrough

```bash
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

**Correct output:** compose reports `v2` or later (v5 is current); the second command prints the same GPU table as
the host. Takes ~60 s the first time (pulls a ~200 MB base image).

- **Both work →** continue.
- **`docker: command not found` →**
  ```bash
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
  ```
  **Then log out and back in.** The group change does not apply to your current shell.
- **`could not select device driver "" with capabilities: [[gpu]]` →** container toolkit
  missing:
  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt update && sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
  Then re-run the passthrough test. ~4 minutes.
- **Permission denied on the docker socket →** you skipped the log-out. Do it now.

### 1.4 Disk

```bash
df -h .
```

**Need 150 GB free** on the filesystem holding the repo — 35 GB image, ~108 GB weights and
caches, plus room for the gallery.

- **≥150 GB free →** continue.
- **Less →** find a bigger volume from **L/M** and use it for `GALLERY_HOST_DIR`,
  `WEIGHTS_HOST_DIR` and `HF_CACHE_DIR` in Phase 2. The **image** still lands on
  `/var/lib/docker` — check that separately with `df -h /var/lib/docker`; it needs 35 GB
  free on its own.

### ✅ Definition of done — Phase 1

`nvidia-smi` works on the host **and** inside a container. `docker compose version` reports
v2 or later. 150 GB free. **Do not start Phase 5 until all four are true.**

---

## Phase 2 — Repository and configuration

**10 minutes.**

### 2.1 Get the code

If the repo is not already on the machine, copy it in — the source machine path and the
transport are yours to choose. Then:

```bash
cd "$A"
pwd                        # must match A
git log --oneline -1
```

**Correct output:** a commit line. If `git` reports "not a repository", you copied the tree
without `.git` — that is fine for deployment, but note it: you cannot verify the version.

### 2.2 Create the env file

```bash
cd "$A"
cp docker/.env.example docker/.env
```

**Correct output:** silence. Confirm with `ls -l docker/.env`.

### 2.3 Set the eight variables

`docker/.env` is the **only** file with machine-specific values. Relative paths resolve
against `docker/`, **not** the repo root.

| Variable | Default | Set it to | Set it when |
|---|---|---|---|
| `WEIGHTS_HOST_DIR` | `../weights` | keep, or a data volume | weights go elsewhere |
| `GALLERY_HOST_DIR` | `../gallery` | **`<M>/htx3d-gallery`** | **usually — grows without bound** |
| `SAM3D_HF_DIR` | `../weights/sam3d-objects-hf` | see 2.4 | **almost always** |
| `HF_CACHE_DIR` | `$HOME/.cache/huggingface` | keep, using **C** | cache is elsewhere |
| `TORCH_CACHE_DIR` | `$HOME/.cache/torch` | keep | " |
| `HY3DGEN_CACHE_DIR` | `$HOME/.cache/hy3dgen` | keep | " |
| `U2NET_CACHE_DIR` | `$HOME/.u2net` | keep | " |
| `TRELLIS2_SERVICE_URL` | `http://host.docker.internal:8710` | **keep** | only if TRELLIS.2 is on another host |

Gallery on a data volume, using **M**:

```bash
sudo mkdir -p /mnt/data/htx3d-gallery && sudo chown "$(whoami)" /mnt/data/htx3d-gallery
sed -i 's|^# GALLERY_HOST_DIR=.*|GALLERY_HOST_DIR=/mnt/data/htx3d-gallery|' docker/.env
grep GALLERY_HOST_DIR docker/.env
```

Replace `/mnt/data` with your real **M**. If **M** is empty, leave the default and note
"gallery on system disk — monitor free space".

Do **not** move `WEIGHTS_DIR`, `GALLERY_DIR`, the engine dirs or `CORS_ORIGINS` into this
file. They are container-internal, fixed in `docker-compose.yml`, and moving them breaks
the container.

### 2.4 SAM3D_HF_DIR — the one that is usually wrong

Its default has **never been exercised**. Decide now:

```bash
ls -d "$HOME/.cache/huggingface/hub/models--facebook--sam-3d-objects" 2>/dev/null && echo "IN HF CACHE"
find / -maxdepth 6 -name "pipeline.yaml" -path "*sam*3d*" 2>/dev/null | head
```

- **Weights not yet downloaded (normal on a clean machine) →** leave the default. Phase 3
  puts them in the HF cache and you will revisit this in Phase 7.3.
- **`find` returns a path →** set `SAM3D_HF_DIR` to the directory **containing**
  `pipeline.yaml`:
  ```bash
  sed -i "s|^SAM3D_HF_DIR=.*|SAM3D_HF_DIR=/full/path/from/find|" docker/.env
  ```

### 2.5 Verify what resolves

```bash
cd docker && docker compose config | grep -A1 "source:" ; cd ..
```

**Correct output:** seven `source:` entries, all **absolute host paths that exist**. Any path
containing a literal `${` or pointing somewhere absurd is a broken variable — fix it now, not
after the build.

### ✅ Definition of done — Phase 2

`docker compose config` resolves seven bind mounts to real absolute paths. No `${` left.

---

## Phase 3 — Model weights

**Longest phase. 30 minutes on a fast link, 3+ hours on a slow one. ~73 GB.**
**Start the gating check before anything else in the day.**

### 3.1 Gating check — do this first

```bash
cd "$A"
python3 scripts/download_models.py --check
```

**Correct output:** a per-family report of what is present. Nothing downloads.

Then:

```bash
python3 scripts/download_models.py --model all
```

If you have no token, it prints:

```
GATED REPOSITORIES
...
  NO TOKEN FOUND.
  Authenticate first, then re-run:
      huggingface-cli login
```

**This is the decision point of the whole day.**

- **Token present, downloads proceed →** continue to 3.2.
- **No token →** log in:
  ```bash
  huggingface-cli login
  ```
  Paste a token from an account that has accepted **all four**:
  - `https://huggingface.co/facebook/sam3`
  - `https://huggingface.co/facebook/sam3.1`
  - `https://huggingface.co/facebook/sam-3d-objects`
  - `https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m`
- **401 or 403 on a `facebook/*` repo →** the token is valid but **terms are not accepted on
  that account**. Open the four URLs in a browser, log in as that account, accept, re-run.
  ~5 minutes if the account holder is you. **If it is not you, escalate immediately** — this
  is the one blocker you cannot fix in the terminal.
- **Nobody at HTX can accept the terms today →** **stop this phase.** Go to
  [Minimum viable handover](#minimum-viable-handover) and run the TRELLIS-only path. SAM 3
  segmentation, SAM 3D Objects and TRELLIS.2 are all unavailable without gated access.

### 3.2 Ungated families

```bash
python3 scripts/download_models.py
```

**Correct output:** `[GET ]` lines followed by `[OK  ]` with a size, or `[SKIP]` if present.

Fetches: `JeffreyXiang/TRELLIS-image-large`, `JeffreyXiang/TRELLIS-text-large`,
`tencent/Hunyuan3D-2.1`, `lpiccinelli/unidepth-v2-vits14`,
`openai/clip-vit-base-patch32`, `microsoft/TRELLIS.2-4B`.

**Timing:** ~59 GB. 15 min at 1 Gbit, ~90 min at 100 Mbit. **Do not kill it.** Watch the
size counter, not the clock.

- **A family fails mid-download →** re-run the same command. It skips what is complete.
- **Disk fills →** stop, move `HF_CACHE_DIR` to volume **M**, re-run.

### 3.3 Verify

```bash
python3 scripts/download_models.py --check
du -sh "$HOME/.cache/huggingface" weights 2>/dev/null
```

**Correct output:** every needed family reported present. HF cache in the tens of GB.

### ✅ Definition of done — Phase 3

`--check` reports all six ungated families present. Gated families present, **or** explicitly
noted as blocked with the reason.

---

## Phase 4 — RealESRGAN, before the build

**1 minute. Skipping this silently breaks Hunyuan textures.**

This checkpoint is **not** on HuggingFace and is **not** in git. The Dockerfile `COPY`s the
engine tree into the image, so it must exist **on the build host before Phase 5**.

```bash
mkdir -p backend/engines/hunyuan/hy3dpaint/ckpt
wget -O backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
ls -lh backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth
```

**Correct output:** a file of **~67 MB**.

- **File is a few KB →** you downloaded an HTML error page. Delete it and retry.
- **No internet on the build host →** copy the file in by hand. **Do not build without it** —
  fixing it later costs a full rebuild.

### ✅ Definition of done — Phase 4

`RealESRGAN_x4plus.pth` exists and is ~67 MB.

---

## Phase 5 — Build and start the container

**~10 minutes on the development machine. Allow 60. Do not kill it before 60 minutes.**

The build compiles nvdiffrast, diffoctreerast, diff-gaussian-rasterization, pytorch3d and the
Hunyuan rasterizer against five GPU architectures. Long silences are normal.

```bash
cd docker
docker compose up -d --build 2>&1 | tee ~/htx-build-$(date +%Y%m%d-%H%M).log
```

**Correct output while running:** step-by-step layer output. These lines confirm the pins
resolved — they must appear:

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

**Expected noise — ignore all of these:**
- `gradio 6.24.0 requires huggingface-hub<2.0,>=1.16.0, but you have huggingface-hub 0.36.2`
  — long-standing and inert. Nothing imports gradio.
- Long pauses during `pytorch3d` and the CUDA extensions. Normal.

**Failures:**

| Symptom | Do this |
|---|---|
| `no space left on device` | `docker system prune -af`, then check `df -h /var/lib/docker` needs 35 GB. Rebuild. |
| Network timeout on a `git clone` | Re-run the same command. Docker reuses completed layers. |
| `FileNotFoundError` … `RealESRGAN_x4plus.pth` | You skipped Phase 4. Do it, then rebuild. |
| Compile error in a CUDA extension | Capture the last 50 log lines. **Stop and note it — this cannot be fixed on site.** |
| Build finishes but no container | `docker compose logs htx-3d` — read the Python traceback. |

Then watch it come up:

```bash
docker compose logs -f
```

**Correct output:** ends with `Uvicorn running on http://0.0.0.0:8000`. Press `Ctrl-C` to
stop following — that does not stop the container.

### ✅ Definition of done — Phase 5

`docker compose ps` shows `htx-3d` as `Up`. Logs show `Uvicorn running`.

---

## Phase 6 — Smoke test

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
        "vram_gb": 95.8,
        "is_blackwell": true
    },
    "models_loaded": [],
    "engines_registered": ["trellis", "hunyuan", "sam3d", "trellis2"],
    "active_engine": null,
    "queue_size": 0
}
```

Check three things:
1. `"status": "ok"`
2. `"available": true` and `"is_blackwell": true`
3. `engines_registered` lists **four** names

`models_loaded` empty is **correct** — engines lazy-load on first use.

Then the UI:

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/
```

**Correct output:** `200`. Open **http://localhost:8000** in a browser on the machine.

**Failures:**

| Symptom | Do this |
|---|---|
| `Connection refused` | `docker compose ps`. If not `Up`, `docker compose logs htx-3d`. |
| `"available": false` | GPU passthrough lost. Re-run Phase 1.3, then `docker compose up -d`. |
| Fewer than four engines | Read logs for the engine that failed to register. Note it; continue. |
| A warning naming `host.docker.internal:8710` | **Expected.** TRELLIS.2 is not deployed yet. Non-fatal. |

### ✅ Definition of done — Phase 6

`status: ok`, `is_blackwell: true`, four engines registered, UI returns 200.

**This is the point at which you have something to show. If the day goes wrong after here,
you still have a working system.**

---

## Phase 7 — Prove each engine

**15–30 minutes. First use of each engine loads a model, so the first run of each is slow.**

Do these **through the UI** at http://localhost:8000. Use any photograph of a vehicle.

Watch engine swaps in a second terminal:

```bash
cd docker && docker compose logs -f | grep -iE 'loading engine|unloading|registered|reachable'
```

### 7.1 TRELLIS — the one that must work

Upload an image, engine **TRELLIS**, generate.

**Correct:** a mesh appears in the viewer. First run ~1–2 min including model load.

- **Multi-billion-GB OOM** (e.g. "tried to allocate 66000000000.00 GiB") → CUDA extensions
  built without your architecture. **Stop and note it.** Not fixable on site.
- **Works →** TRELLIS is the fallback demo. Everything after this is a bonus.

### 7.2 SAM 3 segmentation

Upload a cluttered photo. Click **Segment Object**. Click a point on the target.

**Correct:** a blue mask overlay appears, with a **Mask %** figure and a **Use Mask** button.

- **401/403 in the logs →** gated terms not accepted. See Phase 3.1.

### 7.3 SAM 3D Objects

Engine **SAM 3D Objects**, generate. First load is slow — 12 GB of weights.

- **`FileNotFoundError` listing search paths →** `SAM3D_HF_DIR` is wrong. **The error lists
  every path it tried.** Pick the one containing `pipeline.yaml`, set it in `docker/.env`,
  then `docker compose up -d` (no rebuild needed — it is a bind mount).
- **Works →** tick it off.

### 7.4 Hunyuan3D

Engine **Hunyuan3D**, generate. Slowest engine; allow ~2 min after load.

- **`FileNotFoundError` naming `RealESRGAN_x4plus.pth` →** Phase 4 was skipped. Fetch it and
  **rebuild** (Phase 5). ~15 minutes. Decide against the clock.

### 7.5 Auto-scale

After any generation, check the auto-scale panel shows dimensions in metres with a
confidence badge.

- **Blank or errors →** check logs for `UniDepth`. Non-fatal; the mesh is still valid.

### ✅ Definition of done — Phase 7

TRELLIS generates. Note the pass/fail of each of the other four. **Partial is acceptable —
record exactly which engines work.**

---

## Phase 8 — TRELLIS.2 host service (OPTIONAL, CUT FIRST)

**2–4 hours. Highest risk in the whole runbook. Never been run from this repo.**

**Do not start Phase 8 unless Phases 6 and 7 are done and it is before 14:00.**

`services/trellis2/README.md` §7 lists four things that have never executed: the clone path
in the build script, the acceptance test on FA2, the parameterised launcher, and the systemd
unit. Expect to debug.

### 8.1 Clone at the pin

```bash
export TRELLIS2_ROOT=/opt/trellis2
sudo mkdir -p "$TRELLIS2_ROOT" && sudo chown "$(whoami)" "$TRELLIS2_ROOT"
git clone https://github.com/microsoft/TRELLIS.2 "$TRELLIS2_ROOT"
git -C "$TRELLIS2_ROOT" checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C "$TRELLIS2_ROOT" submodule update --init --recursive
git -C "$TRELLIS2_ROOT" log --oneline -1
```

**Correct output:** `75fbf01 Merge pull request #166 from microsoft/copilot/fix-failing-github-actions-job`

### 8.2 Patch

```bash
cd "$TRELLIS2_ROOT"
git apply --check "$A/services/trellis2/patches/image_feature_extractor.patch" && echo "PATCH APPLIES"
git apply         "$A/services/trellis2/patches/image_feature_extractor.patch"
```

**Correct output:** `PATCH APPLIES`, then silence.

- **`--check` fails →** the tree is not at the pinned commit. Re-do 8.1. Do **not** force it.

### 8.3 Environment

```bash
conda create -n trellis2 python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trellis2
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.13.1
pip install flash_attn==2.8.3.post1
pip install fastapi uvicorn python-multipart pillow
python -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"
```

**Correct output:** `2.10.0+cu128` and an arch list **containing `sm_120`**.

- **`sm_120` absent →** wrong wheel. Re-run the torch install with the exact index URL above.
- **transformers gets upgraded later →** `pip install transformers==5.13.1` again. It is a
  **co-pin**, not a suggestion. Check with `pip show transformers`.

### 8.4 Build the CUDA extensions

Using **D** for cores and **F** for CUDA:

```bash
cp "$A/services/trellis2/build_extensions_blackwell.sh" "$TRELLIS2_ROOT/"
CUDA_HOME=/usr/local/cuda-12.8 \
MAX_JOBS=$(nproc) \
TORCH_CUDA_ARCH_LIST=12.0 \
TRELLIS2_ROOT="$TRELLIS2_ROOT" \
bash "$TRELLIS2_ROOT/build_extensions_blackwell.sh" 2>&1 | tee ~/trellis2-build.log
```

**Timing: 30–90 minutes.** Do not kill it.

**Correct output — the last block must show all six:**

```
OK   o_voxel
OK   cumesh
OK   nvdiffrast.torch
OK   nvdiffrec_render.light
OK   flex_gemm
OK   flash_attn_interface
```

- **Any module missing or `FAIL` →** TRELLIS.2 will not work. **Stop Phase 8, note which
  module failed, keep the log.** The container stack is unaffected.
- **`nvcc: not found` →** your host CUDA is missing (Phase 1.2). Cut Phase 8.

### 8.5 Install service files

```bash
mkdir -p "$TRELLIS2_ROOT/service"
cp "$A/services/trellis2/trellis2_service.py" "$TRELLIS2_ROOT/service/"
cp "$A/services/trellis2/run_service.sh"      "$TRELLIS2_ROOT/service/"
chmod +x "$TRELLIS2_ROOT/service/run_service.sh"
```

### 8.6 systemd

systemd does **not** expand variables in `User=` or `ExecStart=`. Use **B** (user) and
**H** (conda base) as literals.

```bash
sudo cp "$A/services/trellis2/trellis2-service.service" /etc/systemd/system/
sudo sed -i "s|^User=.*|User=$(whoami)|" /etc/systemd/system/trellis2-service.service
sudo sed -i "s|^ExecStart=.*|ExecStart=${TRELLIS2_ROOT}/service/run_service.sh|" \
  /etc/systemd/system/trellis2-service.service
grep -E "^(User|ExecStart)=" /etc/systemd/system/trellis2-service.service
```

**Correct output:** your real username and a real path — no `<`, no `trellis2` placeholder
user unless that user genuinely exists.

Environment file — `TRELLIS2_CONDA_SH` is **H + `/etc/profile.d/conda.sh`**:

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

**Correct output:** five lines, all absolute paths, no `${` remaining.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now trellis2-service
systemctl is-active trellis2-service
curl -s localhost:8710/health
```

**Correct output:** `active`, then `{"status":"ok","model_loaded":false}`.

**`model_loaded:false` is correct** — the 4B model lazy-loads on the first generation.

- **`inactive` / `failed` →** `journalctl -u trellis2-service -n 50 --no-pager`. Most likely
  a wrong `TRELLIS2_CONDA_SH` or a `User=` that cannot read `$TRELLIS2_ROOT`.

### 8.7 Connect the container

```bash
cd "$A/docker" && docker compose restart htx-3d
sleep 20 && docker compose logs --tail 40 htx-3d | grep -i trellis2
```

**Correct output:** no warning about an unreachable TRELLIS.2. Then generate once with
engine **TRELLIS.2** through the UI. First run loads the 4B model — allow **3–5 minutes**.

**Pass criteria** (structural, from the README §8 — **not** timing):
- A recognisable object in the viewer.
- GLB in the **tens of MB**, not kilobytes.
- Face count in the **millions**.

### ✅ Definition of done — Phase 8

`systemctl is-active` → `active`; `/health` → ok; one generation completes through the UI.

---

## Cut-off rule

**Check the clock at 14:00.**

| State at 14:00 | Do this |
|---|---|
| Phase 6 done (container up, health ok) | Continue to Phase 7. Start Phase 8 **only if Phase 7 finishes by 15:00**. |
| Phase 5 still building | Let it finish. Do **not** start Phase 8 today. |
| Phase 5 failed twice | **Stop building.** Go to Minimum viable handover. |
| Phase 3 still blocked on gated access | **Stop.** Run the TRELLIS-only path below. Record the blocker in writing. |

**Hard rule: do not begin Phase 8 after 15:00.** A half-installed TRELLIS.2 with a failed
systemd unit is worse than none — it leaves a service that fails on every boot. If you have
started and it is not working by 16:00:

```bash
sudo systemctl disable --now trellis2-service
```

Leave the container stack running. Note where you stopped.

**Last 30 minutes of the day, whatever state you are in:**

```bash
cd "$A/docker" && docker compose ps
curl -s localhost:8000/api/health | python3 -m json.tool > ~/htx-handover-health.json
docker images | grep htx-3d > ~/htx-handover-images.txt
cp docker/.env ~/htx-handover-env.txt
```

Write down: which engines generated successfully, which failed and with what error, and
anything you were told to escalate.

---

## Minimum viable handover

**The shortest path to something that works.** Cut in this order — top of the list goes
first.

### Cut order

| # | Cut | Saves | You lose | Still works |
|---|---|---|---|---|
| 1 | **Phase 8 — TRELLIS.2** | 2–4 h | The highest-scoring engine | Everything else. Container logs a non-fatal warning. |
| 2 | **SAM 3D Objects** (skip `facebook/sam-3d-objects`, skip 7.3) | ~12 GB, 20 min | One of four engines | TRELLIS, Hunyuan, SAM 3 segmentation |
| 3 | **Hunyuan textures** (skip Phase 4) | 1 min, avoids a rebuild | Hunyuan PBR texturing | Hunyuan geometry may still fail — treat Hunyuan as gone |
| 4 | **SAM 3 segmentation** (skip gated `facebook/sam3`, `sam3.1`) | ~20 min | Interactive segmentation; rembg fallback remains | TRELLIS + auto-scale |

**Never cut:** TRELLIS (`JeffreyXiang/TRELLIS-image-large`), UniDepth + CLIP (auto-scale), the
container itself. That is the demonstrable core.

### Absolute minimum path

If it is 15:00 and nothing works, this is the whole job:

```bash
# 1. gates
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# 2. config
cp docker/.env.example docker/.env

# 3. the one required model — ungated, no token needed
python3 scripts/download_models.py --model trellis_image

# 4. the checkpoint, before the build
mkdir -p backend/engines/hunyuan/hy3dpaint/ckpt
wget -O backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 5. build
cd docker && docker compose up -d --build

# 6. prove it
curl -s localhost:8000/api/health | python3 -m json.tool
```

Then one TRELLIS generation through the UI. **That is a defensible handover.** Everything
else is an increment on top of it.

---

## Every value in one place

Copy this table if you copy nothing else.

### Ports

| Port | Service | Bound |
|---|---|---|
| `8000` | HTX-3D API + UI | container → host |
| `8710` | TRELLIS.2 host service | host `0.0.0.0` |
| `5173` | frontend dev server | dev only, commented out in compose |

### Versions — container (fixed in `docker/Dockerfile`)

| Item | Value |
|---|---|
| Base image | `nvidia/cuda:12.8.0-devel-ubuntu24.04` |
| Python | 3.12 (Ubuntu 24.04) |
| torch | `2.7.0` |
| torch index URL | `https://download.pytorch.org/whl/cu128` |
| transformers | `>=4.35.0,<4.50` |
| huggingface_hub | `>=0.23,<1.0` |
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.6;8.9;10.0;12.0` |
| spconv | `spconv-cu126==2.3.8` |
| kaolin | `0.18.0` (cu126 wheel) |

### Versions — TRELLIS.2 host service

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
| `SPARSE_CONV_BACKEND` | `flex_gemm` |

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

### Model repositories

| Family | Repo ID | Gated | Needed by |
|---|---|---|---|
| TRELLIS image | `JeffreyXiang/TRELLIS-image-large` | No | **TRELLIS — required** |
| TRELLIS text | `JeffreyXiang/TRELLIS-text-large` | No | text-to-3D (optional) |
| Hunyuan3D | `tencent/Hunyuan3D-2.1` | No | Hunyuan engine |
| UniDepth | `lpiccinelli/unidepth-v2-vits14` | No | auto-scale (**CC BY-NC**) |
| CLIP | `openai/clip-vit-base-patch32` | No | auto-scale class prior |
| TRELLIS.2 | `microsoft/TRELLIS.2-4B` | No | TRELLIS.2 service |
| SAM 3 | `facebook/sam3` | **Yes** | interactive segmentation |
| SAM 3.1 | `facebook/sam3.1` | **Yes** | segmentation variant |
| SAM 3D | `facebook/sam-3d-objects` | **Yes** | SAM 3D Objects engine |
| DINOv3 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | **Yes** | TRELLIS.2 conditioner |

### Direct downloads

| File | URL | Destination |
|---|---|---|
| RealESRGAN | `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth` | `backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth` (~67 MB, **before build**) |
| u2net (air-gapped only) | `https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx` | `$HOME/.u2net/` |

### Storage budget

| Item | Size |
|---|---|
| Docker image | 35.2 GB |
| HuggingFace cache | 40 GB |
| Hunyuan3D cache | 14 GB |
| SAM 3D Objects weights | 12 GB |
| TRELLIS weights | 5.3 GB |
| torch hub (DINOv2) | 1.4 GB |
| rembg | 0.2 GB |
| **Gallery** | **unbounded — 14 GB per ~800 generations** |

---

## Things that cannot be fixed on site

Note these and stop. Do not improvise.

1. **Gated terms not accepted, and the account holder is unavailable.** No workaround.
2. **A CUDA extension fails to compile.** Keep the log. Needs the development machine.
3. **Multi-billion-GB OOM at generation.** Extensions were built without your architecture.
   Needs a rebuild with a corrected `TORCH_CUDA_ARCH_LIST`, and diagnosis first.
4. **Compute capability is not `12.0`.** Every Blackwell assumption in Phase 8 is void.
5. **`nvidia-smi` works on the host but never inside a container** after the Phase 1.3 fix.
   Container toolkit / driver mismatch.

## Licence constraints — do not let anyone quietly ignore these

| Component | Constraint |
|---|---|
| **UniDepth** | **CC BY-NC 4.0 — non-commercial.** This is what powers auto-scale. Any commercial or non-research deployment needs legal review or a replacement depth backbone. |
| **Hunyuan3D 2.1** | Territory-limited: excludes EU, UK, South Korea — model *and* output. **Singapore is inside the Territory.** |
| **SAM 3 / SAM 3D** | Trade controls: no military or warfare purposes, nuclear, or espionage. HTX is home-affairs, so the ordinary reading is satisfied — but the wording is broad and deserves a deliberate read. |
| **TRELLIS.2-4B weights** | Code is MIT; **the weights carry their own terms and were never verified.** Confirm before operational use. |

Full audit: [`reference/vendoring.md`](reference/vendoring.md).
