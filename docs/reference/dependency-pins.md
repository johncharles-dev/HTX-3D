# Dependency pins — why they exist and what each one is worth

**Short version.** Five of this image's dependencies were cloned from unpinned git tips.
Two of them drifted within 25 days. All five are now pinned, but **they are not all
equally trustworthy**, and the difference matters when interpreting evaluation results.

Companion document: [`image-resolutions-20260720.md`](image-resolutions-20260720.md) —
the raw record recovered from the 20 July image.

## Why this was done

The 20 July 2026 image produced the project's headline evaluation results: the **3.79
combined manual score** for TRELLIS.2 and all **207 Batch 2 generations**. Those numbers
describe a specific dependency stack. With unpinned tips, a rebuild resolves to whatever
upstream points at that day — so without pins, the reported numbers would describe a build
that no longer exists and cannot be reconstructed.

## Observed drift — 20 July vs 14 August 2026

Measured by a `--no-cache` build on 2026-08-14 (log: `~/htx-rebuild-nocache-20260814-1523.log`
on the development machine).

| Dependency | 20 July 2026 | 14 August 2026 | Drifted? |
|---|---|---|---|
| `pytorch3d` | `b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881` | `3143b3baf8ef8b1023ed76f225af59e2e8a71e06` | **Yes** |
| `moge` | `07444410f1e33f402353b99d6ccd26bd31e469e8` | `925b8ed835a7a9cdb7578ba15c658a0afc969030` | **Yes** |
| `pipeline` | `866f059d2a05cde05e4a52211ec5051fd5f276d6` | `866f059d2a05cde05e4a52211ec5051fd5f276d6` | No |
| `unidepth` | `8d8cfe4c…` | `8d8cfe4c…` | No — already pinned |
| `utils3d` | `9a4eb15e…` | `9a4eb15e…` | No — already pinned |

Two dependencies moved in under a month. Neither move was announced, and neither would
have been visible to anyone rebuilding the image.

## The five pins, and what each is based on

**The basis column is the important one.** Two pins reproduce the benchmarked stack. Three
only reproduce a build we were able to observe — they are *not* known to match what
produced the evaluation numbers.

| Dependency | Pinned commit | Basis | Reproduces the benchmarked stack? |
|---|---|---|---|
| `pytorch3d` | `b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881` | Recovered from the 20 July image's pip metadata | **Yes** |
| `moge` | `07444410f1e33f402353b99d6ccd26bd31e469e8` | Recovered from the 20 July image's pip metadata | **Yes** |
| `nvdiffrast` | `253ac4fcea7de5f396371124af597e6cc957bfae` | Observed in the 2026-08-14 build | **No — unknown** |
| `diffoctreerast` | `b09c20b84ec3aace4729e6e18a613112320eca3a` | Observed in the 2026-08-14 build | **No — unknown** |
| `mip-splatting` | `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9` | Observed in the 2026-08-14 build | **No — unknown** |

### Why three of them can only be "observed"

`nvdiffrast`, `diffoctreerast` and `mip-splatting` are `git clone`d and then installed
from a **local path**. Pip therefore records only `file:///tmp/extensions/...` with no
commit, and the Dockerfile deletes `/tmp/extensions` at the end of the build. The commits
the 20 July image used are gone — not recorded in the image, the container, or any log.

The values pinned above are simply what the tips resolved to on 2026-08-14, captured
because the Dockerfile now echoes `RESOLVED <name> <commit>` before installing. That fixes
the problem going forward; it cannot recover the past.

**Do not describe these three as the benchmarked commits.** If the evaluation is ever
re-run, re-record them from the build log and update this table.

## Verification

The pinned build (`docker-htx-3d:pinned-20260814`, image `28c11635a03d`) confirmed:

- All five pins resolved to exactly the specified commits.
- The July `pytorch3d` and `moge` commits **still build** against the rest of the current
  stack — this was the main risk, and it did not materialise.
- `pytorch3d 0.7.9` and `moge 2.0.0` compiled without error.

Every future build records its own resolutions via the `RESOLVED` echo lines, so drift is
detectable from the build log alone.

### `diffoctreerast` submodules

The clone uses `--recurse-submodules`, which resolves submodules for the **branch tip**,
not for a commit checked out afterwards. The pin is therefore followed by an explicit
`git submodule update --init --recursive`. Without it the pin would silently produce
mismatched submodules — pinned parent, drifting children.

## The gradio inconsistency

The image ships **gradio 6.24.0 in a knowingly broken state**:

```
gradio 6.24.0 requires huggingface-hub<2.0,>=1.16.0,
but you have huggingface-hub 0.36.2 which is incompatible.
```

**Mechanism.** Step 9 installs MoGe, which pulls `huggingface_hub 1.27.0` and drags in
gradio as a transitive. That version of `huggingface_hub` breaks `tokenizers 0.21.4` and
`transformers 4.48.3`, so step 10 downgrades it to `0.36.2`. That downgrade fixes
transformers and leaves gradio unsatisfiable.

**Correction to an earlier assessment.** This was initially thought to be new — a
consequence of MoGe's drift. **It is not.** Pinning MoGe back to the 20 July commit still
installs gradio 6.24.0 with the same conflict, which means the 20 July image almost
certainly shipped the same broken gradio. It is long-standing, not a regression.

**Impact.** Nothing in this backend imports gradio; it is inert. It is left in place
deliberately rather than patched around, because removing it means changing what MoGe
pulls, and that is a larger change than this work warranted. It is recorded here so nobody
mistakes it for a new fault or a symptom of the pinning.

## Image tags on the development machine

| Tag | Image | What it is |
|---|---|---|
| `docker-htx-3d:latest` | `28c11635a03d` | Same as `pinned-20260814` — the correct default |
| `docker-htx-3d:pinned-20260814` | `28c11635a03d` | All five pins applied. **Use this one** |
| `docker-htx-3d:drifted-20260814` | `b0cedfb2a9e5` | Unpinned `--no-cache` build. Kept as drift evidence; never use |
| `docker-htx-3d:cached-20260814` | `d2a48726077e` | 20 July layers + newer app code, built from cache |

> The 20 July image itself **no longer exists**. It was untagged when a later build moved
> `docker-htx-3d:latest`, and the daemon dropped it; `docker commit` on the running
> container then failed with a missing content digest. Its filesystem survives only inside
> the container that is still running from it. **Tag an image before any build that could
> move its tag.**
