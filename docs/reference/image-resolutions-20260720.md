# Dependency resolutions — working image, 20 July 2026

> **Status: historical record.** For what is pinned now and why — including the measured
> drift and the difference between benchmarked and merely-observed pins — see
> [`dependency-pins.md`](dependency-pins.md). This file remains the raw source those pins
> were recovered from.

**What this is.** The Docker image built on 2026-07-20 is the only build of this stack
ever demonstrated to serve the application. Several of its dependencies are installed
from **unpinned** git tips, so a rebuild today can resolve to different commits and
behave differently. This file records what that working image actually resolved to.

**Why it exists.** The image itself is gone. It was untagged when a subsequent build
moved `docker-htx-3d:latest`, and the daemon dropped it — `docker image inspect` on its
ID now returns *"No such image"*. Its filesystem survives only inside the running
container. This file is therefore the only durable record of the combination that worked.

Captured from `docker exec htx-3d pip freeze` on 2026-08-14, while that container was
still running the 20 July image (244 packages total).

> The container's `/app/app` code was newer than the image, applied via `docker cp`. That
> affects application code only, not the dependency resolutions below.

## Unpinned sources — these are what can drift

The Dockerfile clones these without a commit pin, so a fresh build takes whatever the
default branch points at that day.

| Package | Resolved commit |
|---|---|
| `pytorch3d` | `b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881` |
| `moge` | `07444410f1e33f402353b99d6ccd26bd31e469e8` |
| `pipeline` | `866f059d2a05cde05e4a52211ec5051fd5f276d6` |

To reproduce this image's behaviour, pin those three:

```
pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881
moge       @ git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8
```

## Unpinned AND unrecoverable

These were `pip install`ed from local clones under `/tmp/extensions`, which the Dockerfile
deletes at the end of the build (`RUN rm -rf /tmp/extensions`). Pip recorded only a
`file://` path, so **the commits they were built from cannot be recovered** — not from pip
metadata, not from the image, not from the container.

| Package | Recorded as | Cloned from (Dockerfile) |
|---|---|---|
| `nvdiffrast` | `file:///tmp/extensions/nvdiffrast` | `NVlabs/nvdiffrast` (default branch) |
| `diffoctreerast` | `file:///tmp/extensions/diffoctreerast` | `JeffreyXiang/diffoctreerast` (default branch) |
| `diff-gaussian-rasterization` | `file:///tmp/extensions/mip-splatting/submodules/...` | `autonomousvision/mip-splatting` (default branch) |
| `custom-rasterizer` | `file:///app/engines/hunyuan/.../custom_rasterizer` | Vendored in this repo — **not** a drift risk |

**Fixed on 2026-08-14:** the Dockerfile now echoes `RESOLVED <name> <commit>` for each of
these clones before installing, so every future build records what it used. That closes
the gap going forward but cannot recover the July commits — they remain lost. The three
are now pinned to values observed on 2026-08-14 instead; see
[`dependency-pins.md`](dependency-pins.md) for why those are weaker guarantees than the
`pytorch3d` and `moge` pins.

## Already pinned in the Dockerfile — no drift risk

| Package | Commit / version |
|---|---|
| `unidepth` | `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` |
| `utils3d` | `9a4eb15e4021b67b12c460c7057d642626897ec8` |
| `gsplat` | `1.5.0` |
| `spconv-cu126` | `2.3.8` |
| `kaolin` | `0.18.0` |

## Core stack as resolved

| Package | Version |
|---|---|
| `torch` | `2.7.0+cu128` |
| `torchvision` | `0.22.0+cu128` |
| `transformers` | `4.48.3` |
| `huggingface_hub` | `0.36.2` |
| `numpy` | `2.4.3` |
| `diffusers` | `0.30.0` |
| `trimesh` | `4.11.1` |
| `open3d` | `0.19.0` |
| `onnxruntime-gpu` | `1.24.1` |
| `torchmetrics` | `1.9.0` |
| `torchdiffeq` | `0.2.5` |

The full 244-package snapshot is at `~/htx-container-pipfreeze-20260814-1446.txt` on the
development machine — outside the repo, so copy it in if it should survive the handover.

## Note on `transformers`

This container runs `transformers 4.48.3`, pinned by the Dockerfile to `<4.50`. That is
unrelated to the `transformers 5.13.1` co-pin in `services/trellis2/README.md`: TRELLIS.2
runs in a separate host-side conda environment, not in this container. The two are
independent and must not be reconciled.
