# HTX-3D — Deployment Runbook: UNGATED

**From a bare WSL2 instance. No CUDA, no Docker, no conda installed.**
**No HuggingFace gated access.** This is the primary route.

One person, one terminal, one day. Follow top to bottom. Do not skip Phase 0.

There is **no `TROUBLESHOOTING.md`** in this repository. Every fix is inline here.

## What you get

| Delivered | Not delivered today |
|---|---|
| TRELLIS image-to-3D | SAM 3 interactive segmentation |
| Hunyuan3D 2.1 (shape + PBR texture) | SAM 3D Objects |
| Metric auto-scale | TRELLIS.2 |
| Logo-on-surface bake | |
| Web UI, gallery, export (GLB/OBJ/STL/PLY) | |

Background removal still works without SAM 3 — `rembg` is the automatic fallback and needs
no account.

**The health endpoint will still report four engines registered.** That means the Python
objects were constructed, not that all four work. SAM 3D will fail at first use. That is
expected today.

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
```

Write these down:

| ID | Value | Example | Used in |
|---|---|---|---|
| **A** repo root | `____________` | `/home/htx/HTX-3D` | Phases 6–11 |
| **B** user | `____________` | `htx` | Phase 4 docker group |
| **C** home | `____________` | `/home/htx` | Phase 6 cache paths |
| **D** Ubuntu | `____________` | `24.04` or `22.04` | **Phase 3 — picks the CUDA repo URL** |
| **E** WSL | `____________` | `YES - WSL2` | Phases 2, 4 |
| **F** init | `____________` | `systemd` or `init` | **Phase 4 — decides how Docker starts** |
| **G** GPU | `____________` | `NVIDIA RTX PRO 6000 Blackwell Max-Q, 12.0, 97887 MiB, 590.48` | Phase 2 gate |
| **H** disk free | `____________` | `1007G  420G avail` | Phase 2 gate — need **150 GB** |
| **I** RAM | `____________` | `30 GB total` | Phase 9 |

> **`export A` matters.** Later phases paste `$A/...` into commands. After Phase 6, run
> `export A=$(pwd)` from the repo root. Re-export in any new terminal.

---

## Phase 1 — sudo

**30 seconds. Do this before anything else.**

```bash
sudo -v && echo "SUDO OK"
```

**Correct output:** `SUDO OK` (possibly after a password prompt).

- **`SUDO OK` →** continue.
- **`<user> is not in the sudoers file` →** **Stop.** You cannot install anything. This needs
  a Windows-side administrator to reset the WSL distro's default user, or an existing admin
  to add you. Nothing in this runbook works without it.

Keep the session warm — this refreshes the timestamp so long steps don't stall on a prompt:

```bash
sudo -v
```

---

## Phase 2 — WSL2 sanity checks

**5 minutes. Three things that are different from native Linux.**

### 2.1 The GPU comes from Windows — do NOT install a Linux driver

```bash
nvidia-smi
ls -la /usr/lib/wsl/lib/libcuda.so* 2>/dev/null | head -3
```

**Correct output:** the normal `nvidia-smi` table showing your GPU, and at least one
`libcuda.so` under `/usr/lib/wsl/lib`.

- **Works →** continue. **Never run `apt install nvidia-driver-*` inside WSL.** It will
  overwrite the WSL GPU stub libraries and break CUDA for the whole distro.
- **`nvidia-smi: command not found` or "no devices" →** the **Windows** driver is missing or
  too old. Fix on the Windows side: install the NVIDIA driver for your card from
  nvidia.com (570+ for Blackwell), then from a Windows terminal:
  ```
  wsl --shutdown
  ```
  and reopen the distro. **Nothing inside Linux fixes this.**

### 2.2 Disk — the WSL virtual disk

```bash
df -h /
```

**Need 150 GB free.**

**The WSL2 filesystem is a virtual disk (`ext4.vhdx`) living on the Windows drive.** If `df`
shows under 150 GB free, **the fix is on the Windows side, not inside Linux.** Deleting files
in Linux will not help if the vhdx has hit its ceiling.

From a **Windows** PowerShell (not WSL):

```powershell
# free space on the Windows drive is the real limit
Get-PSDrive C

# grow the virtual disk if it has hit its maximum
wsl --shutdown
wsl --manage Ubuntu --resize 500GB
```

Replace `Ubuntu` with the distro name shown by `wsl -l -v`.

### 2.3 RAM — set it now, before the build

```bash
free -g
```

**Correct output:** ideally 24 GB or more.

WSL2 defaults to about half of Windows' RAM. The CUDA extension compiles are memory-hungry.
Set it explicitly from Windows in `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=8
swap=16GB
```

Then, from Windows PowerShell:

```powershell
wsl --shutdown
```

Reopen the distro and re-check `free -g`.

> **Why this matters:** parallel CUDA builds OOM'd at 30 GB on the development machine.
> This runbook hardcodes low job counts to compensate. Do not raise them.

### ✅ Definition of done — Phase 2

`nvidia-smi` works inside WSL. 150 GB free. RAM confirmed.

---

## Phase 3 — CUDA 12.8 toolkit

**15 minutes, ~3 GB download.**

> **You can skip this entire phase today.** The container ships its own CUDA 12.8 from
> `nvidia/cuda:12.8.0-devel-ubuntu24.04`. The **host** toolkit is only needed if you later
> add the TRELLIS.2 host service. Skip it if you are short on time; come back when you need
> TRELLIS.2.

### 3.1 Check your Ubuntu version — this picks the repo URL

```bash
. /etc/os-release; echo "$VERSION_ID"
lsb_release -a 2>/dev/null
```

**Correct output:** `24.04` or `22.04`. That is value **D**.

### 3.2 Add the CUDA repository

**On WSL2, use the `wsl-ubuntu` repo.** It is the same for both Ubuntu versions and
deliberately excludes the driver packages, which you must not install inside WSL.

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

If you are **not** on WSL (value **E** said native Linux), use the matching distro repo
instead:

```bash
# Ubuntu 24.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```

**Correct output:** `dpkg` reports the keyring unpacked; `apt update` lists a
`developer.download.nvidia.com` source with no GPG errors.

- **`NO_PUBKEY` or GPG error →** the keyring did not install. Re-run the `dpkg -i`.
- **404 on the `.deb` →** you used the wrong distro string. Re-check **D**.

### 3.3 Install the toolkit only

```bash
sudo apt install -y cuda-toolkit-12-8
```

**Install `cuda-toolkit-12-8`, never `cuda` or `cuda-drivers`.** The bare `cuda` package
pulls the Linux driver, which breaks WSL GPU access.

**Timing: 8–15 minutes.**

### 3.4 Put nvcc on PATH

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version | grep release
ls -d /usr/local/cuda-12.8
```

**Correct output:** `Cuda compilation tools, release 12.8, V12.8.61` and the directory path.

- **`nvcc: command not found` after sourcing →** check `ls /usr/local/` for the real
  directory name and adjust `CUDA_HOME`.

### ✅ Definition of done — Phase 3

`nvcc --version` reports 12.8, **or** you consciously skipped this phase.

---

## Phase 4 — Docker inside WSL

**10 minutes.**

### 4.1 Install

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

**Correct output:** ends with a Docker version banner.

**Now apply the group change.** Either open a **new** WSL terminal, or:

```bash
newgrp docker
```

Verify without sudo:

```bash
docker --version
docker ps
```

**Correct output:** a version string, then an empty container table with headers.

- **`permission denied … /var/run/docker.sock` →** the group change has not applied. Close
  every WSL terminal and from Windows PowerShell run `wsl --shutdown`, then reopen.

### 4.2 Start the daemon — this differs by init system

Check value **F**:

```bash
ps -p 1 -o comm=
```

- **Reports `systemd` →** normal:
  ```bash
  sudo systemctl enable --now docker
  systemctl is-active docker
  ```
  **Correct output:** `active`.

- **Reports `init` →** systemd is **not** enabled in this distro. Two options.

  **Option A — start it manually each session (fastest today):**
  ```bash
  sudo service docker start
  sudo service docker status
  ```
  **Correct output:** `Docker is running`.
  You must re-run `sudo service docker start` after every WSL restart.

  **Option B — enable systemd (better, costs one restart):**
  ```bash
  sudo tee /etc/wsl.conf >/dev/null <<'EOF'
  [boot]
  systemd=true
  EOF
  ```
  Then from **Windows** PowerShell:
  ```powershell
  wsl --shutdown
  ```
  Reopen the distro, confirm `ps -p 1 -o comm=` now says `systemd`, then use Option A's
  systemctl commands.

  **Take Option B if it is before midday. Take Option A if it is later.**

### ✅ Definition of done — Phase 4

`docker ps` works without sudo and the daemon is running.

---

## Phase 5 — GPU passthrough into Docker

**8 minutes. This is the step that most often fails on WSL.**

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

This is what `docker-compose.yml` needs — it requests the GPU through
`deploy.resources.reservations.devices` with `driver: nvidia`, which uses this runtime.

### 5.3 Generate the CDI spec

On WSL the GPU libraries live in `/usr/lib/wsl/lib` rather than the usual places, and the
CDI spec records that explicitly.

```bash
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
```

**Correct output:** `nvidia-ctk cdi list` prints device names including `nvidia.com/gpu=all`
and `nvidia.com/gpu=0`.

- **`cdi generate` errors about no devices →** `nvidia-smi` is not working in this shell.
  Go back to Phase 2.1.

### 5.4 Restart Docker and test

```bash
# systemd
sudo systemctl restart docker
# or, non-systemd
sudo service docker restart

docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

**Timing:** ~90 seconds the first time — it pulls a ~200 MB image.

**Correct output:** the same GPU table you saw on the host, printed from inside the
container.

**Failures:**

| Symptom | Do this |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | 5.2 did not take. Re-run it, restart Docker, retry. |
| `nvidia-container-cli: initialization error` | Try the CDI addressing instead: `docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`. If **that** works but `--gpus all` does not, add `"features": {"cdi": true}` to `/etc/docker/daemon.json` and restart Docker. |
| `unknown or invalid runtime name: nvidia` | `/etc/docker/daemon.json` was not written. Check it with `cat`, re-run 5.2. |
| Works with sudo, fails without | Group change not applied. `wsl --shutdown` from Windows. |

### ✅ Definition of done — Phase 5

`docker run --rm --gpus all … nvidia-smi` prints your GPU. **Do not start Phase 9 until this
passes.**

---

## Phase 6 — Repository and configuration

**10 minutes.**

### 6.1 Get the code

Copy the repo onto the machine — transport is yours. Then:

```bash
cd <where you put it>
export A="$(pwd)"
echo "$A"
git log --oneline -1 2>/dev/null || echo "(no .git — fine, but you cannot verify the version)"
```

> **Keep the repo on the WSL filesystem** (`/home/...`), **not** on `/mnt/c/`. Model loading
> across the Windows bridge is dramatically slower.

### 6.2 Create the env file

```bash
cd "$A"
cp docker/.env.example docker/.env
ls -l docker/.env
```

### 6.3 The eight variables

Relative paths resolve against `docker/`, **not** the repo root.

| Variable | Default | Today |
|---|---|---|
| `WEIGHTS_HOST_DIR` | `../weights` | keep |
| `GALLERY_HOST_DIR` | `../gallery` | keep, or point at a bigger path |
| `SAM3D_HF_DIR` | `../weights/sam3d-objects-hf` | **keep — unused today** |
| `HF_CACHE_DIR` | `$HOME/.cache/huggingface` | keep |
| `TORCH_CACHE_DIR` | `$HOME/.cache/torch` | keep |
| `HY3DGEN_CACHE_DIR` | `$HOME/.cache/hy3dgen` | keep |
| `U2NET_CACHE_DIR` | `$HOME/.u2net` | keep |
| `TRELLIS2_SERVICE_URL` | `http://host.docker.internal:8710` | **keep — unreachable today, non-fatal** |

The defaults are correct for this variant. **Create the mount targets so Docker does not
create them as root-owned:**

```bash
mkdir -p "$A/weights" "$A/gallery" \
         "$HOME/.cache/huggingface" "$HOME/.cache/torch" \
         "$HOME/.cache/hy3dgen" "$HOME/.u2net" \
         "$A/weights/sam3d-objects-hf"
```

The last one is an empty placeholder. Compose mounts it read-only; SAM 3D will fail at first
use with a `FileNotFoundError`, which is correct and expected today.

### 6.4 Verify what resolves

```bash
cd "$A/docker" && docker compose config | grep -A1 "source:" ; cd "$A"
```

**Correct output:** **seven** `source:` entries, every one an absolute path that exists. Any
literal `${` is a broken variable — fix it now, not after the build.

### ✅ Definition of done — Phase 6

Seven bind mounts resolve to real absolute paths. `$A` is exported.

---

## Phase 7 — Model weights (ungated only)

**~59 GB. 20 minutes on a fast link, 2+ hours on a slow one. No account needed.**

```bash
cd "$A"
python3 scripts/download_models.py --check
```

**Correct output:** a per-family report. Nothing downloads.

```bash
python3 scripts/download_models.py
```

That command fetches **only ungated** families and needs no token:

| Family | Repo |
|---|---|
| TRELLIS image | `JeffreyXiang/TRELLIS-image-large` |
| TRELLIS text | `JeffreyXiang/TRELLIS-text-large` |
| Hunyuan3D | `tencent/Hunyuan3D-2.1` |
| UniDepth | `lpiccinelli/unidepth-v2-vits14` |
| CLIP | `openai/clip-vit-base-patch32` |
| TRELLIS.2 | `microsoft/TRELLIS.2-4B` |

**Correct output:** `[GET ]` lines followed by `[OK  ]` with sizes, or `[SKIP]`.

It will also print a **GATED REPOSITORIES** block naming `facebook/sam3`, `facebook/sam3.1`,
`facebook/sam-3d-objects` and `facebook/dinov3-vitl16-pretrain-lvd1689m`, then
`Skipping gated downloads.` **That is the expected outcome today. Not an error.**

**Failures:**

| Symptom | Do this |
|---|---|
| Interrupted mid-download | Re-run the same command. Completed files are skipped. |
| `No space left on device` | Phase 2.2 — the vhdx is full. Fix from Windows. |
| `huggingface_hub is not installed` | `pip install huggingface_hub` (add `--break-system-packages` on 24.04). |

Verify:

```bash
python3 scripts/download_models.py --check
du -sh "$HOME/.cache/huggingface" weights
```

**Correct output:** six ungated families present; cache in the tens of GB.

> `microsoft/TRELLIS.2-4B` downloads even though TRELLIS.2 is not deployed. Harmless — it is
> ungated and it means the weights are already there when you add the service later.

### ✅ Definition of done — Phase 7

All six ungated families report present.

---

## Phase 8 — RealESRGAN, before the build

**1 minute. Skipping this silently breaks Hunyuan textures.**

Not on HuggingFace, not in git, and the Dockerfile `COPY`s the engine tree into the image —
so it **must exist before Phase 9**.

```bash
cd "$A"
mkdir -p backend/engines/hunyuan/hy3dpaint/ckpt
wget -O backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
ls -lh backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth
```

**Correct output:** a file of **~67 MB**.

- **A few KB →** you saved an HTML error page. Delete and retry.
- **No internet on this machine →** copy the file across by hand. **Do not build without it.**
  Fixing it later costs a full rebuild.

### ✅ Definition of done — Phase 8

`RealESRGAN_x4plus.pth` exists and is ~67 MB.

---

## Phase 9 — Build and start the container

**~10 minutes native. On WSL allow 45. Do not kill it before 60 minutes.**

```bash
cd "$A/docker"
docker compose up -d --build 2>&1 | tee ~/htx-build-$(date +%Y%m%d-%H%M).log
```

**These lines must appear** — they confirm the pins resolved:

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

**Expected noise — ignore:**
- `gradio 6.24.0 requires huggingface-hub<2.0,>=1.16.0, but you have huggingface-hub 0.36.2`
  — long-standing and inert. Nothing imports gradio.
- Multi-minute silences during `pytorch3d` and the CUDA extensions. Normal.

**Failures:**

| Symptom | Do this |
|---|---|
| Killed / `signal: killed` / OOM during a compile | WSL ran out of RAM. Phase 2.3 — raise `memory=` in `.wslconfig`, `wsl --shutdown`, retry. **This is the most likely WSL failure.** |
| `no space left on device` | `docker system prune -af`, then Phase 2.2 to grow the vhdx. |
| Network timeout in a `git clone` | Re-run the same command; completed layers are reused. |
| `FileNotFoundError` … `RealESRGAN_x4plus.pth` | Phase 8 was skipped. Do it, rebuild. |
| Compile error inside a CUDA extension | Capture the last 50 log lines. **Stop and note it — not fixable on site.** |

> **`docker/Dockerfile` already sets `ENV MAX_JOBS=8`** (line 89, before every extension
> build). That caps nvcc parallelism, which is what prevents the OOM that otherwise lands
> 20-30 minutes into the build. If it still OOMs after raising WSL memory, lower it to `4`
> and rebuild. **Do not raise it.**

Watch it come up:

```bash
docker compose logs -f
```

**Correct output:** ends with `Uvicorn running on http://0.0.0.0:8000`. `Ctrl-C` stops
following, not the container.

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

Check three things:
1. `"status": "ok"`
2. `"available": true`
3. `engines_registered` lists four names

`models_loaded` empty is **correct** — engines lazy-load.
`sam3d` and `trellis2` being listed does **not** mean they work today.

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/
```

**Correct output:** `200`.

**Open the UI.** From WSL, use the Windows browser:

```bash
echo "http://localhost:8000"
```

WSL2 forwards localhost to Windows automatically. If it does not resolve, get the WSL IP:

```bash
hostname -I | awk '{print $1}'
```

and browse to `http://<that-ip>:8000` from Windows.

**Failures:**

| Symptom | Do this |
|---|---|
| `Connection refused` | `docker compose ps`; if not `Up`, read `docker compose logs htx-3d`. |
| `"available": false` | GPU passthrough lost. Re-run Phase 5.4, then `docker compose up -d`. |
| Warning naming `host.docker.internal:8710` | **Expected.** TRELLIS.2 not deployed. Non-fatal. |

### ✅ Definition of done — Phase 10

`status: ok`, GPU available, four engines registered, UI returns 200.

**This is the point at which you have something to show.**

---

## Phase 11 — Prove what you shipped

**20–30 minutes. First use of each engine loads a model, so first runs are slow.**

Watch swaps in a second terminal:

```bash
cd "$A/docker" && docker compose logs -f | grep -iE 'loading engine|unloading|registered|reachable'
```

### 11.1 TRELLIS — the one that must work

Upload a photograph in the UI, engine **TRELLIS**, generate.

**Correct:** a mesh appears in the viewer. First run ~1–2 minutes including model load.

- **Multi-billion-GB OOM** (e.g. "tried to allocate 66000000000.00 GiB") → CUDA extensions
  built without your architecture. **Stop and note it.** Not fixable on site.
- **Works →** this is your fallback demo. Everything after is a bonus.

### 11.2 Background removal (rembg, no account needed)

Upload a photo with a background and generate **without** touching Segment Object.

**Correct:** the background is removed automatically. First use downloads the u2net model to
`$HOME/.u2net` — allow an extra minute.

- **Air-gapped →** seed it by hand:
  `https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx` into `$HOME/.u2net/`

### 11.3 Hunyuan3D

Engine **Hunyuan3D**, generate. Slowest engine; allow ~2 minutes after load.

- **`FileNotFoundError` naming `RealESRGAN_x4plus.pth` →** Phase 8 was skipped. Fetch it and
  **rebuild** (Phase 9, ~15 min from cache). Decide against the clock.

### 11.4 Auto-scale

After any generation, the auto-scale panel should show dimensions in metres with a
confidence badge.

- **Blank →** check logs for `UniDepth`. Non-fatal; the mesh is still valid.

### 11.5 Logo bake

Load a result, open the logo tool, place a logo, bake.

**Correct:** the logo appears baked into the texture, and a new gallery entry is created.

### 11.6 Gallery

Open the **Gallery** tab.

**Correct:** your generations appear as cards with thumbnails.

- **Empty despite successful generations →** open the browser dev tools Network tab and check
  `/api/gallery?page=1&per_page=500` returns 200. `Gallery.tsx` swallows fetch errors
  silently and renders the same "No generations yet" state either way, so the network tab is
  the only way to tell the two apart.

### 11.7 Export

Export a model as GLB. Then OBJ.

**Correct:** files download and open in a viewer.

### ✅ Definition of done — Phase 11

TRELLIS generates, auto-scale reports metres, gallery lists the results, export downloads.
Record pass/fail for Hunyuan and logo bake.

---

## Cut-off rule

**Check the clock at 14:00.**

| State at 14:00 | Do this |
|---|---|
| Phase 10 done | Continue to Phase 11 at leisure. |
| Phase 9 still building | Let it finish. Do not start anything else. |
| Phase 9 failed twice with OOM | Raise `.wslconfig` memory, `wsl --shutdown`, one more attempt. If that fails, stop. |
| Still stuck before Phase 5 | **Stop installing.** Write down exactly which step fails and its output. |

**Last 30 minutes, whatever state you are in:**

```bash
cd "$A/docker" && docker compose ps > ~/htx-handover-ps.txt
curl -s localhost:8000/api/health | python3 -m json.tool > ~/htx-handover-health.json
cp "$A/docker/.env" ~/htx-handover-env.txt
docker images | grep htx-3d > ~/htx-handover-images.txt
free -g > ~/htx-handover-mem.txt; df -h / >> ~/htx-handover-mem.txt
```

Write down: which engines generated, which failed and with what error, and anything to
escalate.

---

## Minimum viable handover

If time runs out, cut in this order:

| # | Cut | Saves | You lose |
|---|---|---|---|
| 1 | Phase 3 (CUDA toolkit) | 15 min | Nothing today. Only needed for TRELLIS.2 later. |
| 2 | Phase 11.5 (logo bake) | 10 min | A demo feature. |
| 3 | Phase 11.3 (Hunyuan) + Phase 8 | 20 min | One engine. |

**Never cut:** TRELLIS weights, RealESRGAN if you intend Hunyuan at all, the build, the
health check.

**Absolute minimum path** — the whole job in six commands, assuming Phases 1–5 passed:

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

Then one TRELLIS generation through the UI. **That is a defensible handover.**

---

## Later — enabling SAM 3, SAM 3D and TRELLIS.2 when access arrives

**Verified against the code. None of the three needs an image rebuild.**

Here is why, so you can say it with confidence:

| Engine | How it resolves at runtime | Verified in |
|---|---|---|
| SAM 3 | `hf_hub_download(repo_id="facebook/sam3", …)` — reads the HuggingFace cache, which is a **bind mount** | `backend/engines/sam3/sam3/model_builder.py:664-672` |
| SAM 3D | Searches for `pipeline.yaml`, including `/app/weights/sam3d-objects-hf/` — a **bind mount** | `backend/app/services/sam3d_objects.py:57-77` |
| TRELLIS.2 | `load()` only probes `http://host.docker.internal:8710`. **No local weights at all** | `backend/app/services/trellis2.py:63-67` |

The engine **code** is already inside the image. Only data and configuration are outside it.

### Step 1 — Accept the terms

On a HuggingFace account, accept all four:

- `https://huggingface.co/facebook/sam3`
- `https://huggingface.co/facebook/sam3.1`
- `https://huggingface.co/facebook/sam-3d-objects`
- `https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m`

**Acceptance is per account and does not transfer with a copied token or copied files.**

### Step 2 — Download the weights on the host

```bash
cd "$A"
huggingface-cli login          # paste the token from that account
python3 scripts/download_models.py --model all
```

**Correct output:** `Token found. Proceeding.` then `[OK  ]` for the four gated repos.

- **401 / 403 on a `facebook/*` repo →** token is valid but terms are not accepted **on that
  account**. Go back to Step 1.

They land in `$HOME/.cache/huggingface`, which is already mounted into the container.

### Step 3a — SAM 3: restart only

```bash
cd "$A/docker"
docker compose restart htx-3d
```

**Time: 30 seconds. No rebuild.** Test with Segment Object in the UI.

### Step 3b — SAM 3D Objects: point at the weights, then recreate

Find where the weights actually landed:

```bash
find "$HOME/.cache/huggingface" -name "pipeline.yaml" -path "*sam*3d*" 2>/dev/null
```

Set `SAM3D_HF_DIR` to the directory **containing** that file:

```bash
cd "$A"
sed -i "s|^SAM3D_HF_DIR=.*|SAM3D_HF_DIR=/full/path/from/find|" docker/.env
grep SAM3D_HF_DIR docker/.env
cd docker && docker compose up -d          # NOT restart — see below
```

**Use `up -d`, not `restart`.** Bind mounts are fixed when the container is *created*, so a
change to `SAM3D_HF_DIR` needs the container recreated. `docker compose up -d` detects the
changed config and recreates it. **Still no rebuild.** ~30 seconds.

If SAM 3D still fails, its `FileNotFoundError` **lists every path it searched** — pick the
right one from that list.

### Step 3c — TRELLIS.2: a host-side install, no container change

TRELLIS.2 never runs in the container. It needs its own conda environment, torch 2.10 and
five CUDA extensions compiled for sm_120 — **2 to 4 hours**, and
`services/trellis2/README.md` §7 records that it has never been installed from this repo in
its vendored form.

Do it on a separate day, following the **GATED** runbook's TRELLIS.2 phase. It requires the
Phase 3 CUDA toolkit, which is why that phase is worth doing eventually.

Once the host service answers on `:8710`, the container needs **nothing** — `load()` probes
the URL at first use. A `docker compose restart htx-3d` only clears the startup warning from
the logs.

### Summary

| To enable | Rebuild? | Command | Time |
|---|---|---|---|
| SAM 3 | **No** | `docker compose restart htx-3d` | 30 s |
| SAM 3D Objects | **No** | edit `SAM3D_HF_DIR`, `docker compose up -d` | 30 s |
| TRELLIS.2 | **No** (container) | host install + optional restart | 2–4 h host-side |

The only thing that ever needs a rebuild is a change to **application code**, because
`backend/app/` is `COPY`d into the image rather than mounted.

---

## Every value in one place

### Ports

| Port | Service |
|---|---|
| `8000` | HTX-3D API + UI |
| `8710` | TRELLIS.2 host service (not used in this variant) |

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

### Pinned commits

| Package | Commit |
|---|---|
| `pytorch3d` | `b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881` |
| `moge` | `07444410f1e33f402353b99d6ccd26bd31e469e8` |
| `nvdiffrast` | `253ac4fcea7de5f396371124af597e6cc957bfae` |
| `diffoctreerast` | `b09c20b84ec3aace4729e6e18a613112320eca3a` |
| `mip-splatting` | `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9` |
| `utils3d` | `9a4eb15e4021b67b12c460c7057d642626897ec8` |
| `unidepth` | `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` |

### Repos fetched today (all ungated)

| Family | Repo ID |
|---|---|
| TRELLIS image | `JeffreyXiang/TRELLIS-image-large` |
| TRELLIS text | `JeffreyXiang/TRELLIS-text-large` |
| Hunyuan3D | `tencent/Hunyuan3D-2.1` |
| UniDepth | `lpiccinelli/unidepth-v2-vits14` |
| CLIP | `openai/clip-vit-base-patch32` |
| TRELLIS.2 | `microsoft/TRELLIS.2-4B` |

### Gated repos (later)

`facebook/sam3` · `facebook/sam3.1` · `facebook/sam-3d-objects` ·
`facebook/dinov3-vitl16-pretrain-lvd1689m`

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

### Storage budget

| Item | Size |
|---|---|
| Docker image | 35.2 GB |
| HuggingFace cache (ungated only) | ~45 GB |
| Hunyuan3D cache | 14 GB |
| TRELLIS weights | 5.3 GB |
| torch hub | 1.4 GB |
| Gallery | unbounded — 14 GB per ~800 generations |

---

## Things that cannot be fixed on site

1. **A CUDA extension fails to compile.** Keep the log; needs the development machine.
2. **Multi-billion-GB OOM at generation.** Extensions lack your architecture. Needs
   diagnosis, then a rebuild.
3. **`nvidia-smi` never works inside WSL.** A Windows-side driver problem.
4. **No sudo.** Needs a Windows administrator.
5. **Compute capability is not `12.0`.** Blackwell assumptions are void; note it.

## Licence constraints

| Component | Constraint |
|---|---|
| **UniDepth** | **CC BY-NC 4.0 — non-commercial.** This powers auto-scale. Commercial or non-research deployment needs legal review or a replacement depth backbone. |
| **Hunyuan3D 2.1** | Territory-limited: excludes EU, UK, South Korea — model *and* output. **Singapore is inside the Territory.** |
| **TRELLIS** | MIT (Microsoft). |

Full audit: [`reference/vendoring.md`](reference/vendoring.md).
