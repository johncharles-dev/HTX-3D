#!/usr/bin/env python3
"""Model weight fetcher for the HTX-3D stack.

Covers every model family the stack needs, including the gated Meta repositories and the
one checkpoint that is not on HuggingFace at all.

Two destinations, because the code looks in two places:

  --output DIR   TRELLIS weights and the RealESRGAN checkpoint, laid out the way the
                 engines expect. Defaults to ./weights, matching WEIGHTS_HOST_DIR.
  HF cache       Everything loaded via from_pretrained() lands in the HuggingFace cache
                 ($HF_HOME, default ~/.cache/huggingface). Mounted into the container by
                 docker-compose.yml as HF_CACHE_DIR.

Usage:
    python scripts/download_models.py                  # everything that is not gated
    python scripts/download_models.py --model all      # including gated (needs a token)
    python scripts/download_models.py --model trellis_image
    python scripts/download_models.py --check          # report what is present, fetch nothing
    python scripts/download_models.py --list           # show every family and its source

Every repo ID below was read from this repository's own source or from a populated cache.
Nothing here is guessed. Families whose source could not be established are listed in
UNVERIFIED at the bottom and are deliberately NOT fetched.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry. "evidence" records where the repo ID came from — keep it accurate.
# ---------------------------------------------------------------------------

HF_MODELS = {
    "trellis_image": {
        "repo": "JeffreyXiang/TRELLIS-image-large",
        "dest": "weights",          # snapshot into --output/<name>
        "gated": False,
        "needed_by": "TRELLIS image-to-3D (required)",
        "evidence": "backend/app/config.py:87",
    },
    "trellis_text": {
        "repo": "JeffreyXiang/TRELLIS-text-large",
        "dest": "weights",
        "gated": False,
        "needed_by": "TRELLIS text-to-3D (optional)",
        "evidence": "backend/app/config.py:88",
    },
    "hunyuan": {
        "repo": "tencent/Hunyuan3D-2.1",
        "dest": "hf_cache",
        "gated": False,
        "needed_by": "Hunyuan3D shape + PBR texture engine",
        "evidence": "backend/app/services/hunyuan.py:73",
    },
    "unidepth": {
        "repo": "lpiccinelli/unidepth-v2-vits14",
        "dest": "hf_cache",
        "gated": False,
        "needed_by": "auto-scale metric depth (CC BY-NC — non-commercial)",
        "evidence": "backend/app/services/auto_scale.py:59",
    },
    "clip": {
        "repo": "openai/clip-vit-base-patch32",
        "dest": "hf_cache",
        "gated": False,
        "needed_by": "auto-scale class prior classifier",
        "evidence": "backend/app/services/class_priors.py:104",
    },
    "trellis2": {
        "repo": "microsoft/TRELLIS.2-4B",
        "dest": "hf_cache",
        "gated": False,
        "needed_by": "TRELLIS.2 host service (not the container — see services/trellis2/)",
        "evidence": "services/trellis2/trellis2_service.py:41",
    },
    # --- gated: require an accepted licence on the fetching account ---------
    "sam3": {
        "repo": "facebook/sam3",
        "dest": "hf_cache",
        "gated": True,
        "needed_by": "SAM 3 interactive segmentation",
        "evidence": "backend/engines/sam3/sam3/model_builder.py:668",
    },
    "sam3_1": {
        "repo": "facebook/sam3.1",
        "dest": "hf_cache",
        "gated": True,
        "needed_by": "SAM 3.1 segmentation variant",
        "evidence": "backend/engines/sam3/sam3/model_builder.py:664",
    },
    "sam3d": {
        "repo": "facebook/sam-3d-objects",
        "dest": "hf_cache",
        "gated": True,
        "needed_by": "SAM 3D Objects engine",
        "evidence": "populated HuggingFace cache",
    },
    "dinov3": {
        "repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dest": "hf_cache",
        "gated": True,
        "needed_by": "TRELLIS.2 image conditioner (DinoV3FeatureExtractor)",
        "evidence": "populated cache + services/trellis2/patches/image_feature_extractor.patch",
    },
}

# Not on HuggingFace — a direct release asset.
DIRECT_DOWNLOADS = {
    "realesrgan": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "rel_path": "backend/engines/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth",
        "needed_by": "Hunyuan3D texture upscaling",
        "evidence": "backend/engines/hunyuan/hy3dpaint/README.md:9",
        "note": "Gitignored (67 MB). Must exist on the BUILD HOST before 'docker compose "
                "build', because the Dockerfile COPYs the engine tree into the image.",
    },
}

# Sources that could not be established from this repository. Listed, never fetched.
UNVERIFIED = {
    "microsoft/TRELLIS-image-large":
        "Present in the cache with 2 files, but the code loads JeffreyXiang/TRELLIS-image-large. "
        "Probably an incidental fetch; role unclear.",
    "facebook/dinov2-giant":
        "Present in the cache, but SAM 3D loads DINOv2 via torch.hub "
        "(facebookresearch/dinov2, dinov2_vitl14_reg) rather than from HuggingFace.",
    "openai/clip-vit-large-patch14":
        "Present in the cache; the code references only clip-vit-base-patch32. "
        "Pulled by some dependency — which one is not established.",
    "briaai/RMBG-2.0":
        "Present in the cache; no reference found anywhere in this repository.",
    "Ruicheng/moge-vitl":
        "Fetched indirectly by the 'moge' package at runtime, not by this repository.",
    "u2net.onnx":
        "rembg downloads it automatically to ~/.u2net on first use. For an air-gapped "
        "install, seed it from "
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
}


# ---------------------------------------------------------------------------

def human(nbytes: int) -> str:
    if nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.1f} KB"
    if nbytes < 1024 ** 3:
        return f"{nbytes / 1024 ** 2:.1f} MB"
    return f"{nbytes / 1024 ** 3:.1f} GB"


def dir_size(path: str) -> str:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return human(total)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def hf_token() -> str:
    """Resolve a HuggingFace token without printing or storing it."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    hf_home = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface")
    token_file = os.path.join(hf_home, "token")
    if os.path.isfile(token_file):
        try:
            return open(token_file).read().strip()
        except OSError:
            return ""
    return ""


def gated_preflight(selected: dict) -> bool:
    """Report exactly which gated repos need terms accepted, before any request is made.

    Returns True if it is safe to proceed with the gated entries.
    """
    gated = {k: v for k, v in selected.items() if v.get("gated")}
    if not gated:
        return True

    print("\n" + "=" * 74)
    print("GATED REPOSITORIES")
    print("=" * 74)
    print("These require a HuggingFace account that has accepted each model's terms.")
    print("Access is granted per account and does NOT transfer with copied files.\n")
    for key, m in gated.items():
        print(f"  {m['repo']}")
        print(f"      accept terms at: https://huggingface.co/{m['repo']}")
        print(f"      needed by:       {m['needed_by']}")
    print()

    if not hf_token():
        print("  NO TOKEN FOUND.")
        print("  Authenticate first, then re-run:")
        print("      huggingface-cli login")
        print("  or set HF_TOKEN in the environment.")
        print("  (A token is read from $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN, or")
        print("   $HF_HOME/token — it is never printed or copied by this script.)")
        print("\n  Skipping gated downloads.\n")
        return False

    print("  Token found. Proceeding.\n")
    print("  If a download still fails with 401/403, the token is valid but the terms")
    print("  for that specific repo have not been accepted on this account.\n")
    return True


def fetch_hf(repo_id: str, dest_kind: str, output_dir: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [FAIL] huggingface_hub is not installed:  pip install huggingface_hub")
        return False

    name = repo_id.split("/")[-1]
    kwargs = {"repo_id": repo_id}
    if dest_kind == "weights":
        local_dir = os.path.join(output_dir, name)
        if os.path.isdir(local_dir) and any(Path(local_dir).iterdir()):
            print(f"  [SKIP] {repo_id} -> already at {local_dir} ({dir_size(local_dir)})")
            return True
        kwargs["local_dir"] = local_dir
        target_desc = local_dir
    else:
        target_desc = os.environ.get("HF_HOME", "~/.cache/huggingface")

    print(f"  [GET ] {repo_id} -> {target_desc}")
    try:
        path = snapshot_download(**kwargs)
        print(f"  [OK  ] {repo_id} ({dir_size(path)})")
        return True
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            print(f"  [GATE] {repo_id}: access denied.")
            print(f"         Accept the terms at https://huggingface.co/{repo_id}")
            print(f"         using the same account your token belongs to.")
        else:
            print(f"  [FAIL] {repo_id}: {type(e).__name__}: {msg[:200]}")
        return False


def fetch_direct(key: str, spec: dict) -> bool:
    target = repo_root() / spec["rel_path"]
    if target.is_file():
        print(f"  [SKIP] {key} -> already at {target} ({human(target.stat().st_size)})")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [GET ] {key} -> {target}")
    print(f"         {spec['url']}")
    try:
        urllib.request.urlretrieve(spec["url"], target)
        print(f"  [OK  ] {key} ({human(target.stat().st_size)})")
        return True
    except Exception as e:
        if target.exists():
            target.unlink()          # never leave a truncated checkpoint behind
        print(f"  [FAIL] {key}: {type(e).__name__}: {str(e)[:200]}")
        return False


def report_presence(output_dir: str) -> None:
    print("\nPresence check (nothing downloaded):\n")
    hf_home = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface")
    hub = Path(hf_home) / "hub"
    for key, m in HF_MODELS.items():
        if m["dest"] == "weights":
            p = Path(output_dir) / m["repo"].split("/")[-1]
            ok = p.is_dir() and any(p.iterdir())
        else:
            p = hub / ("models--" + m["repo"].replace("/", "--"))
            ok = p.is_dir()
        flag = "present" if ok else "MISSING"
        gate = " (gated)" if m["gated"] else ""
        print(f"  {flag:8s} {key:15s} {m['repo']}{gate}")
    for key, spec in DIRECT_DOWNLOADS.items():
        p = repo_root() / spec["rel_path"]
        print(f"  {'present' if p.is_file() else 'MISSING':8s} {key:15s} {spec['rel_path']}")
    print()


def show_list() -> None:
    print("\nModel families, with where each repo ID was established from:\n")
    for key, m in HF_MODELS.items():
        print(f"  {key:15s} {m['repo']}{'  [GATED]' if m['gated'] else ''}")
        print(f"                  {m['needed_by']}")
        print(f"                  source: {m['evidence']}")
    for key, spec in DIRECT_DOWNLOADS.items():
        print(f"  {key:15s} {spec['url']}")
        print(f"                  {spec['needed_by']}")
        print(f"                  source: {spec['evidence']}")
    print("\nPresent in the development cache but NOT fetched — source unestablished:\n")
    for repo, why in UNVERIFIED.items():
        print(f"  {repo}")
        print(f"      {why}")
    print()


def main() -> int:
    choices = list(HF_MODELS) + list(DIRECT_DOWNLOADS) + ["all", "ungated"]
    ap = argparse.ArgumentParser(
        description="Download model weights for the HTX-3D stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="ungated", choices=choices,
                    help="'ungated' (default) skips gated repos; 'all' includes them")
    ap.add_argument("--output", default="./weights",
                    help="directory for TRELLIS weights (default: ./weights)")
    ap.add_argument("--check", action="store_true", help="report what is present, fetch nothing")
    ap.add_argument("--list", action="store_true", help="list every family and its source")
    args = ap.parse_args()

    output_dir = os.path.abspath(args.output)

    if args.list:
        show_list()
        return 0
    if args.check:
        report_presence(output_dir)
        return 0

    os.makedirs(output_dir, exist_ok=True)
    print(f"Weights directory: {output_dir}")
    print(f"HuggingFace cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

    if args.model == "all":
        hf_sel = dict(HF_MODELS)
        direct_sel = dict(DIRECT_DOWNLOADS)
    elif args.model == "ungated":
        hf_sel = {k: v for k, v in HF_MODELS.items() if not v["gated"]}
        direct_sel = dict(DIRECT_DOWNLOADS)
        print("\nMode: ungated only. Use --model all for the gated Meta repositories.")
    elif args.model in DIRECT_DOWNLOADS:
        hf_sel, direct_sel = {}, {args.model: DIRECT_DOWNLOADS[args.model]}
    else:
        hf_sel, direct_sel = {args.model: HF_MODELS[args.model]}, {}

    do_gated = gated_preflight(hf_sel)
    if not do_gated:
        hf_sel = {k: v for k, v in hf_sel.items() if not v["gated"]}

    failures = []
    if hf_sel:
        print("HuggingFace repositories:")
        for key, m in hf_sel.items():
            if not fetch_hf(m["repo"], m["dest"], output_dir):
                failures.append(m["repo"])
    if direct_sel:
        print("\nDirect downloads:")
        for key, spec in direct_sel.items():
            if not fetch_direct(key, spec):
                failures.append(key)
            if spec.get("note"):
                print(f"         NOTE: {spec['note']}")

    print("\n" + "=" * 74)
    if failures:
        print(f"Completed with {len(failures)} failure(s): {', '.join(failures)}")
    else:
        print("All selected models present.")
    print("=" * 74)
    print("\nPoint the stack at these locations via docker/.env "
          "(copy docker/.env.example):")
    print(f"    WEIGHTS_HOST_DIR={output_dir}")
    print(f"    HF_CACHE_DIR={os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))}")
    print("\nThe remaining cache variables (TORCH_CACHE_DIR, HY3DGEN_CACHE_DIR,")
    print("U2NET_CACHE_DIR) default to the standard per-user locations and only need")
    print("setting if your caches live elsewhere. See docker/.env.example.")
    if UNVERIFIED:
        print(f"\n{len(UNVERIFIED)} cached repositories were NOT fetched because their role "
              "could not be\nestablished from this repository. Run --list to see them.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
