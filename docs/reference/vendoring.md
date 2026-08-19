# Vendored engines — provenance and governing licences

`backend/engines/` contains four third-party engines copied into this repository rather
than pulled as dependencies. They are **not** git submodules and carry no upstream git
history, so this file is the record of where each came from and what actually governs it.

**Read this before redistributing the repository or the model weights.** Two of the four
are governed by custom licences with conditions that ordinary open-source licences do not
have, and one upstream file states its licence incorrectly.

## Summary

| Engine | Upstream | Governing licence | Licence file in tree |
|---|---|---|---|
| `trellis` | `microsoft/TRELLIS` | MIT (Microsoft Corporation) | `trellis/LICENSE` |
| `hunyuan` | Tencent Hunyuan3D-2.1, via a personal fork — see below | Tencent Hunyuan 3D 2.1 Community License | `hunyuan/hy3dshape/LICENSE`, `hunyuan/hy3dpaint/LICENSE`, `hunyuan/hy3dshape/NOTICE` |
| `sam3` | `facebookresearch/sam3` @ `ce25da6` | **SAM License** (Meta, 19 Nov 2025) | `sam3/LICENSE` |
| `sam3d_objects` | `facebookresearch/sam-3d-objects` @ `81a8237` | **SAM License** (Meta, 19 Nov 2025) | `sam3d_objects/LICENSE` |

The `sam3` and `sam3d_objects` licence files were **missing** from the vendored trees and
were added on 2026-08-19, copied byte-identically from the upstream clones. Before that,
the repository shipped Meta-licensed code with no licence text at all.

## Known discrepancy — `sam3/pyproject.toml` wrongly declares MIT

`backend/engines/sam3/pyproject.toml` contains:

```toml
license = {file = "LICENSE"}
...
"License :: OSI Approved :: MIT License",
```

**That MIT classifier is wrong.** SAM 3 is governed by Meta's custom SAM License, which is
not MIT and not OSI-approved. `sam3/LICENSE` — the file the same `pyproject.toml` points at
— is the authoritative text and contradicts the classifier directly.

This is **upstream's own error**, present in `facebookresearch/sam3` itself. It is
**preserved deliberately** so the vendored tree remains a faithful, unmodified copy of
upstream, which matters for provenance. It is recorded here rather than corrected in place.

Do not rely on that classifier. Anyone auditing this repository who reads only the
`pyproject.toml` will reach the wrong conclusion about SAM 3's licence.

## SAM License — applies to `sam3`, `sam3d_objects`, and their weights

The SAM License is unusual in that it covers the weights as well as the code:

> "'SAM Materials' means, collectively, Documentation and the models, software and
> algorithms, including machine-learning model code, **trained model weights**,
> inference-enabling code, training-enabling code, fine-tuning enabling code, and other
> elements of the foregoing distributed by Meta and made available under this Agreement."

So the same terms govern the `facebook/sam3`, `facebook/sam3.1` and
`facebook/sam-3d-objects` weights, however they are transferred — including on physical
media.

**Redistribution is permitted, conditionally:**

> "You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited
> license … to use, reproduce, distribute, copy, create derivative works of, and make
> modifications to the SAM Materials."

> "If you distribute or make the SAM Materials, or any derivative works thereof, available
> to a third party, you may only do so under the terms of this Agreement and **you shall
> provide a copy of this Agreement** with any such SAM Materials."

### Trade Controls and ITAR — read this clause

> "You are not the target of Trade Controls and your use of SAM Materials must comply with
> Trade Controls. You agree not to use, or permit others to use, SAM Materials for any
> activities subject to the International Traffic in Arms Regulations (ITAR) or end uses
> prohibited by Trade Controls, **including those related to military or warfare purposes,
> nuclear industries or applications, espionage**, or the development or use of guns or
> illegal weapons."

HTX is a home-affairs agency — police, civil defence, immigration — rather than a military
one, so the ordinary reading is that this is satisfied. But the terms "military or warfare
purposes" and "espionage" are broad, and this clause deserves a deliberate read by whoever
owns deployment, not a silent assumption. It is the one licence term in this project where
the identity of the recipient materially affects the answer.

The two SAM License files (`sam3/LICENSE`, `sam3d_objects/LICENSE`) differ only in
whitespace; they are substantively the same agreement.

## Hunyuan — territory restriction, and fork provenance

The vendored `hunyuan` tree came from a **personal fork**,
`git@github.com:johncharles-dev/Hunyuan3D_Blackwell.git` @ `ac5332a` ("Fix CUDA OOM: always
swap between shape/texture stages"), not from Tencent's repository directly. The fork
carries Blackwell/sm_120 adaptations. Its licence files are identical to the ones in that
clone.

The Tencent Hunyuan 3D 2.1 Community License is **territory-limited**:

> "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH
> KOREA AND IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW."

> "'Territory' shall mean the worldwide territory, excluding the territory of the European
> Union, United Kingdom and South Korea."

> "You must not use, reproduce, modify, distribute, or display the Tencent Hunyuan 3D 2.1
> Works, Output or results of the Tencent Hunyuan 3D 2.1 Works **outside the Territory**.
> Any such use outside the Territory is unlicensed and unauthorized under this Agreement."

**Singapore is inside the Territory**, so HTX's use is covered. Neither the model nor its
output may be used or distributed in the EU, UK or South Korea. Redistribution requires
passing on the agreement (§3(a)), and a scale trigger applies above 1 million monthly
active users (§4).

## A note on scope

Everything above concerns the **engine source code** vendored into this repository. The
model **weights** are separate artefacts with their own terms — although, as noted, the SAM
License and the Tencent licence each cover both. For the full weights audit, including
models whose licence could not be determined from local files, see the manifest that
accompanies the transferred media and
[`dependency-pins.md`](dependency-pins.md) for the Python dependency stack.

**UniDepth is CC BY-NC 4.0 — non-commercial.** It is not a vendored engine (it is a pip
dependency of the auto-scaling feature), but it is the most restrictive licence in the
project and is recorded here so it is not missed.
