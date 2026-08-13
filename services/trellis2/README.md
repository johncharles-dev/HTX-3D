# TRELLIS.2 host service (HTX layer)

TRELLIS.2 is the highest-scoring image-to-3D pipeline in this project's evaluation
(3.79 combined, blinded manual scoring — see `evaluation/manual_eval/`). It does **not**
run inside the HTX-3D container: it needs the host's torch 2.10 / CUDA 12.8 /
FlashAttention-2 stack for Blackwell sm_120. It therefore runs as a small HTTP service
on the host, and the container's `trellis2` engine
(`backend/app/services/trellis2.py`) forwards images to it over the docker bridge.

---

## 1 · What this directory is

**Upstream TRELLIS.2 is not vendored here.** This directory contains only the
HTX-authored service layer plus a single patch against upstream:

| File | Purpose |
|---|---|
| `trellis2_service.py` | FastAPI service. Holds the 4B model resident, `GET /health`, `POST /generate` → GLB bytes |
| `run_service.sh` | Launcher. Activates the conda env and sets the sm_120 / FA2 environment |
| `trellis2-service.service` | systemd unit — **contains placeholders, see §3.6** |
| `build_extensions_blackwell.sh` | Builds the five CUDA extensions for sm_120 |
| `run_example_blackwell.sh` | Runs one generation outside the service — the acceptance test |
| `patches/image_feature_extractor.patch` | One-file compatibility fix against upstream |

### Why the patch exists

Upstream `DinoV3FeatureExtractor.extract_features` iterates `self.model.layer`.
From **transformers 5.x** the blocks are nested one level deeper, in a
`DINOv3ViTEncoder` at `.model`. The verified environment runs **transformers 5.13.1**,
so without this patch TRELLIS.2 raises `AttributeError` on every generation. The patch
is a defensive `getattr` that works on both layouts:

```python
encoder = getattr(self.model, 'model', self.model)
for i, layer_module in enumerate(encoder.layer):
```

It changes no model behaviour — the 3.79 result is upstream's, not a consequence of
local modification.

---

## 2 · Upstream pin

The reproducible unit is **two** pins, not one: the upstream commit **and** the
`transformers` version. The commit alone does not give a working install.

| Pin | Value |
|---|---|
| `microsoft/TRELLIS.2` commit | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` — 2026-06-05, *"Merge pull request #166 from microsoft/copilot/fix-failing-github-actions-job"* |
| `transformers` | `5.13.1` |

```bash
git clone https://github.com/microsoft/TRELLIS.2
cd TRELLIS.2
git checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
pip install transformers==5.13.1
```

### Why transformers is a co-pin

The patched function in `trellis2/modules/image_feature_extractor.py` is coupled to the
`transformers` DINOv3 API in more places than the patch itself touches. The patch's
`getattr(self.model, 'model', self.model)` is deliberately tolerant and handles both the
pre-5.x and 5.x block layouts. What is **not** version-tolerant is the surrounding code
it sits in:

- `self.model.embeddings(image, bool_masked_pos=None)`
- `self.model.rope_embeddings(image)`
- `layer_module(hidden_states, position_embeddings=...)`

None of those is verified against any version other than **5.13.1**. A different
`transformers` release can change any of them and break the extractor in a way the patch
does not address — so pinning the commit while letting `transformers` float will not
reliably reproduce this install.

> **Do not track upstream `main`, and do not let `transformers` resolve freely.** The
> patch in §3.2 is verified against this commit and this `transformers` version, and no
> other combination. If you need a newer upstream, re-verify the patch first
> (`git apply --check`) and re-run the whole acceptance test — upstream may have fixed
> the transformers 5.x issue itself, in which case the patch should be dropped rather
> than forced.

---

## 3 · Setup

Assumes: an sm_120 GPU, CUDA 12.8 toolkit, conda, and a HuggingFace account that has
accepted the gated **DINOv3** conditioner terms. Set `TRELLIS2_ROOT` to wherever you
cloned upstream (this guide uses `/opt/trellis2`).

### 3.1 Clone at the pinned commit

```bash
export TRELLIS2_ROOT=/opt/trellis2
git clone https://github.com/microsoft/TRELLIS.2 "$TRELLIS2_ROOT"
git -C "$TRELLIS2_ROOT" checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C "$TRELLIS2_ROOT" submodule update --init --recursive
```

### 3.2 Apply the patch

```bash
cd "$TRELLIS2_ROOT"
git apply --check /path/to/HTX-3D/services/trellis2/patches/image_feature_extractor.patch
git apply         /path/to/HTX-3D/services/trellis2/patches/image_feature_extractor.patch
```

`--check` first: it fails loudly if the tree is not at the pinned commit.

### 3.3 Create the environment

Create a conda env named `trellis2` with Python 3.12 and the verified stack:

```bash
conda create -n trellis2 python=3.12 -y
conda activate trellis2
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.13.1     # co-pin — see §2, do not let this float
pip install flash_attn==2.8.3.post1
pip install fastapi uvicorn python-multipart pillow
```

If a later step upgrades `transformers` as a transitive dependency, pin it back before
building extensions — check with `pip show transformers`.

Upstream's own `setup.sh` (`--new-env`, `--basic`, …) covers the remaining Python
dependencies; consult it for anything the service reports missing. The versions above
are those read from the working install, not guesses — see §5.

### 3.4 Build the CUDA extensions

```bash
cp /path/to/HTX-3D/services/trellis2/build_extensions_blackwell.sh "$TRELLIS2_ROOT/"
TRELLIS2_ROOT="$TRELLIS2_ROOT" bash "$TRELLIS2_ROOT/build_extensions_blackwell.sh"
```

Builds o-voxel, nvdiffrast v0.4.0, nvdiffrec (renderutils), CuMesh and FlexGEMM against
`TORCH_CUDA_ARCH_LIST=12.0`. The script ends with an import check — **all six modules
must report `OK`** before continuing. See §7 for a reproducibility caveat on CuMesh and
FlexGEMM.

### 3.5 Install the service files

`run_service.sh` expects the service to live at `$TRELLIS2_ROOT/service/`:

```bash
mkdir -p "$TRELLIS2_ROOT/service"
cp /path/to/HTX-3D/services/trellis2/trellis2_service.py "$TRELLIS2_ROOT/service/"
cp /path/to/HTX-3D/services/trellis2/run_service.sh      "$TRELLIS2_ROOT/service/"
chmod +x "$TRELLIS2_ROOT/service/run_service.sh"
```

### 3.6 Configure and install the systemd unit

systemd does **not** expand environment variables in `User=` or `ExecStart=`. Edit both
to real values before installing:

```bash
sudo cp /path/to/HTX-3D/services/trellis2/trellis2-service.service \
        /etc/systemd/system/
sudo editor /etc/systemd/system/trellis2-service.service   # set User= and ExecStart=
```

Put machine-specific overrides in an environment file the unit already reads:

```bash
sudo tee /etc/default/trellis2-service >/dev/null <<'EOF'
TRELLIS2_ROOT=/opt/trellis2
TRELLIS2_CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh
TRELLIS2_CONDA_ENV=trellis2
TRELLIS2_SERVICE_PORT=8710
HF_HOME=/opt/hf-cache
EOF
```

Authenticate for the gated DINOv3 download — **never commit a token**:

```bash
sudo -u <service-user> HF_HOME=/opt/hf-cache huggingface-cli login
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now trellis2-service
systemctl is-active trellis2-service     # -> active
curl -s localhost:8710/health            # -> {"status":"ok","model_loaded":false}
```

`model_loaded:false` immediately after start is correct — the 4B model lazy-loads on the
first `/generate` so the service starts instantly.

### 3.7 Point HTX-3D at the service

The container reaches the host via `TRELLIS2_SERVICE_URL` (see `docker/docker-compose.yml`
and `backend/app/config.py`). Default is `http://host.docker.internal:8710`.

---

## 4 · Environment variables

| Variable | Default | Controls |
|---|---|---|
| `TRELLIS2_ROOT` | `/opt/trellis2` | Where upstream TRELLIS.2 is cloned. Sets `PYTHONPATH` and the working directory |
| `TRELLIS2_CONDA_SH` | `$HOME/miniconda3/etc/profile.d/conda.sh` | Conda bootstrap script |
| `TRELLIS2_CONDA_ENV` | `trellis2` | Conda environment name |
| `TRELLIS2_SERVICE_PORT` | `8710` | Port the service binds (`0.0.0.0`) |
| `TRELLIS2_EXTDIR` | `/tmp/extensions` | Scratch directory for extension builds |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit root |
| `TORCH_CUDA_ARCH_LIST` | `12.0` | Target arch. `12.0` = sm_120 (Blackwell) |
| `MAX_JOBS` | `$(nproc)` | Build parallelism. Was hardcoded to `16` on the dev machine |
| `ATTN_BACKEND` | `flash_attn` | Dense attention. **FA2. See §7 — `flash_attn_3` is wrong on sm_120** |
| `SPARSE_ATTN_BACKEND` | `flash_attn` | Sparse attention |
| `SPARSE_CONV_BACKEND` | `flex_gemm` | Sparse convolution |
| `HF_HOME` | `$HOME/.cache/huggingface` | HuggingFace cache and token location |
| `HF_TOKEN` | **none** | HF auth for gated DINOv3. Deliberately undefaulted — see below |
| `CUMESH_REF` | *(empty)* | Optional commit/tag pin for CuMesh. Empty = upstream branch tip |
| `FLEXGEMM_REF` | *(empty)* | Optional commit/tag pin for FlexGEMM. Empty = upstream branch tip |

`HF_TOKEN` has **no default by design**. The scripts read it from the environment, or
fall back to `$HF_HOME/token` as written by `huggingface-cli login`. No credential is
stored anywhere in this repository. A misconfigured deployment fails at authentication
rather than silently running unauthenticated and producing a confusing download error.

---

## 5 · Verified configuration

Read from the development machine on 2026-08-13. These are the values under which the
3.79 evaluation result was produced.

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 31.4 GB) |
| NVIDIA driver | 590.48.01 |
| CUDA toolkit | 12.8, V12.8.61 |
| Python | 3.12.13 |
| torch | 2.10.0+cu128 (arch list includes `sm_120`) |
| transformers | 5.13.1 |
| flash_attn | 2.8.3.post1 (PyPI wheel) |
| nvdiffrast | 0.4.0 |
| o_voxel | 0.0.1 |
| cumesh | 0.0.1 |
| flex_gemm | 1.0.0 |
| Upstream TRELLIS.2 | `75fbf018` + `patches/image_feature_extractor.patch` |

---

## 6 · Target hardware note — timings do not transfer

HTX's machine is an **RTX PRO 6000 Blackwell Max-Q** (GB202, sm_120, 96 GB).

Same architecture as the RTX 5090, so **the extension build is unchanged** —
`TORCH_CUDA_ARCH_LIST=12.0` remains correct and no code changes are required. The far
larger 96 GB of VRAM comfortably exceeds the ~6.6 GB peak observed here.

**However, that card is power-limited to 300 W against the RTX 5090's 575 W.** Every
generation time recorded in this project — including the `gen_s` figures in §8 and all
timings in the evaluation reports — was measured on the 575 W card and **is not valid on
HTX's hardware.** They must be re-benchmarked before being quoted in any HTX-facing
document or capacity plan. Expect slower wall-clock; do not assume a particular ratio,
because the relationship between power limit and throughput is not linear.

---

## 7 · Known-unverified

**Nothing in this directory has been executed in its vendored form.** The files were
authored and proven on the development machine, then parameterised for a machine that is
not that one — and the parameterised versions have not themselves been run. The only
component verified as-vendored is the patch, and that is verified by checksum, not by
execution.

Specifically:

1. **The corrected build script's clone path has never run.** Upstream `setup.sh`
   clones CuMesh and FlexGEMM, but the original `build_extensions_blackwell.sh` omitted
   those two clones — it worked only because the extensions were already built in the
   development environment. Those clone steps have been added here (matching the
   upstream URLs at `setup.sh:125,131`) but have never executed.
2. **`run_example_blackwell.sh` has never run on FA2.** In the development environment it
   ran with `ATTN_BACKEND=flash_attn_3`. It is set to `flash_attn` here so the acceptance
   test exercises the same backend as the service — but that combination is untested.
3. **The parameterised `run_service.sh` has never launched the service.** The live
   deployment runs the original hardcoded script; the `TRELLIS2_ROOT` / `TRELLIS2_CONDA_SH`
   / `HF_TOKEN` logic is unexercised.
4. **The systemd unit has never been loaded in this form.** `User=` and `ExecStart=` are
   placeholders; `systemd-analyze verify` flags the `ExecStart` path as non-existent, which
   is expected until §3.6 is completed.

**A clean-machine run is the first thing HTX should do.** A successful end-to-end
install — §3 through the §8 acceptance test — is what promotes this directory from
*corrected* to *verified*. Until that run happens, treat every step here as reviewed but
unproven.

### Reproducibility caveat — unpinned CUDA extensions

Upstream clones **CuMesh** and **FlexGEMM** from the default branch tip with no commit
pin. The commits used for the benchmarked build are **not recoverable**: the source trees
under `/tmp/extensions` were cleared, pip recorded only a local `file://` path with no VCS
metadata, and neither package exposes a `__version__`. A build today may therefore differ
from the one that scored 3.79.

Once a clean-machine run establishes a known-good build, pin it:

```bash
export CUMESH_REF=<commit>
export FLEXGEMM_REF=<commit>
```

and record those hashes here.

### Attention backend

`flash_attn_3` (3.0.0) is installed in the development environment and was tried, but
found wrong on Blackwell sm_120. `flash_attn` (FA2, 2.8.3.post1) is the verified backend
and the one the 3.79 benchmark ran through. Both scripts default to FA2; `ATTN_BACKEND`
is overridable if HTX wants to re-test FA3 on their card.

---

## 8 · Acceptance test

Confirms the CUDA extension build is genuinely correct. This matters more than it
sounds: a mis-built extension on Blackwell does not usually crash — it produces garbage
geometry or an absurd allocation request. A structural check catches that.

**Worked example: SCDF Red Rhino** (a Home Team asset; the largest output in the
reference set, so it exercises the export path hardest).

```bash
# 1. Direct run, bypassing the service — tests the extension build in isolation
TRELLIS2_ROOT=/opt/trellis2 bash run_example_blackwell.sh

# 2. Through the service
curl -s -X POST localhost:8710/generate \
  -F "image=@SCDF_Red_Rhino.jpg" \
  -F "seed=0" -F "texture_size=1024" -F "decimation_target=200000" \
  -o /tmp/red_rhino.glb -D /tmp/headers.txt
```

**Pass criteria — structural, not numeric:**

- HTTP 200, and `Content-Type: model/gltf-binary`.
- The GLB loads without error in any glTF viewer and shows a recognisable vehicle.
- File size is of the expected order — **tens of MB**, not kilobytes.
- Face count is in the **millions**, not thousands.
- Response headers `X-Gen-Seconds` and `X-Peak-VRAM-GB` are present and plausible.
- `build_extensions_blackwell.sh`'s import check reported `OK` for all six modules.

**Reference values from the RTX 5090 — context, NOT thresholds.** Do not fail a
deployment for missing these; see §6 on timings and note that cross-GPU geometric
determinism has not been established.

| Object | Input | gen_s | Peak VRAM | Verts | Faces | GLB |
|---|---|---:|---:|---:|---:|---:|
| SCDF Red Rhino | 1024×678 RGB | 27.0 | 6.64 GB | 5,233,772 | 10,612,622 | 44.1 MB |
| SCDF Fire Engine | 1379×894 RGB | 41.4 | 5.79 GB | 3,814,413 | 7,729,034 | 41.8 MB |
| Nissan Sylphy | 1536×1016 RGBA | 17.6 | 4.97 GB | 3,154,644 | 6,435,988 | 38.5 MB |
| Generator | 1600×1200 RGB | 26.0 | 5.97 GB | 4,142,696 | 8,302,382 | 39.2 MB |
| Hyundai i40 | 1536×752 RGB | 62.9 | 3.96 GB | 1,937,218 | 3,882,068 | 36.3 MB |
| Bronco ATTC ambulance | 2691×1787 RGBA | 17.8 | 5.12 GB | 3,428,526 | 7,006,714 | 42.4 MB |

Source: `results.json` from the development machine's TRELLIS.2 evaluation run (six
objects, all `status: OK`). The GLBs, input cutouts and turntable renders for these six
were left on the development machine under `TRELLIS.2/htx_eval/` (258 MB) and are not
committed here.

### Note on output scale

This service returns geometry in a **normalised** axis-aligned bounding box —
`[[-0.5,-0.5,-0.5], [0.5,0.5,0.5]]`, fixed in `trellis2_service.py`. It performs **no**
metric scaling. Real-world dimensions are applied downstream by HTX-3D's `auto_scale`
service (`backend/app/services/auto_scale.py`). An acceptance test for *this* component
should therefore never assert physical dimensions in metres — that would be testing a
different part of the system.

---

## 9 · Licensing

| Component | Licence | Status |
|---|---|---|
| Upstream TRELLIS.2 (code) | **MIT**, Copyright (c) Microsoft Corporation | Read from `LICENSE` at the pinned commit. Permits use, modification and redistribution with the notice retained |
| `patches/image_feature_extractor.patch` | Derivative of an MIT-licensed file | Keep upstream's `LICENSE` alongside any redistribution of patched source |
| HTX-authored files in this directory | Same terms as the parent HTX-3D repository | — |

**Requires confirmation before deployment:**

- **`microsoft/TRELLIS.2-4B` model weights.** The code being MIT does **not** mean the
  weights are. The weights carry their own terms and were not verified during this work.
  Confirm before any operational or redistributed use.
- **DINOv3 image conditioner.** A gated HuggingFace model — terms are accepted per
  account. Whoever operates the service must accept them under an appropriate account,
  and that acceptance is not transferable by copying a token.
- **UniDepth is CC BY-NC — non-commercial.** UniDepth is not used by this service, but
  it is a dependency of HTX-3D's `auto_scale` feature, which is what supplies real-world
  scale. **If HTX intends any commercial or non-research deployment of auto-scaling, this
  licence is a blocker and needs legal review or a replacement depth backbone.** Flagged
  here because it is the most consequential licensing constraint in the wider system.
