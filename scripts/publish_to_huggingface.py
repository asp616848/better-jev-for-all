"""
Publish ekVachan's checkpoints and the Space demo to Hugging Face.

Written without HF/GPU access (this session only authors code, never runs
it -- see PRD.md's "written without GPU access" convention). Meant to be
run once, by hand, on the GPU box where `checkpoints/` actually exists,
after `huggingface-cli login` (or HF_TOKEN set) with write access to the
target account.

Usage:
    uv run python3 -u -m scripts.publish_to_huggingface \
        --model-repo abhi6168/ekvachan-decoder \
        --space-repo abhi6168/ekvachan

Does two independent things (each individually re-runnable/idempotent via
--skip-model / --skip-space):
  1. Uploads checkpoints/ekvachan-decoder-qwen-{benchcorpus,vision}/ (each
     with its manifest.json) to a HF *model* repo, preserving the two
     top-level subfolder names RoutingDecoderModel(checkpoints_dir=...)
     expects, so `snapshot_download(repo_id=..., local_dir="checkpoints")`
     in hf_space/app.py reproduces the exact layout serve/inference.py
     already knows how to load -- no repacking needed on either end.
  2. Uploads hf_space/ (app.py, requirements.txt, README.md) PLUS the
     actual source subtrees app.py imports (serve/, benchmarks/common/,
     training/decoder_lora_lib.py, eval/ -- see VENDOR_PATHS below) to a
     HF *Space* repo, sdk=gradio, deliberately **no Docker** (explicit
     cost/complexity requirement for this deployment): a plain Gradio SDK
     Space just needs its source files present and a requirements.txt: no
     Dockerfile, and critically, no `git clone` of this repo happening at
     Space build/request time -- this script performs that "get the code
     over there" step itself, once, from the actual reviewed source on
     disk, not from a URL the Space would fetch on its own later. This
     does NOT create the Space or set its hardware to T4-small/ZeroGPU --
     that one-time step still has to be done by hand in the HF web UI
     (Settings -> Hardware), since the API for it depends on account
     tier/quota/billing this script has no way to verify blind.

Real verification still owed after running this (this script only
authors + performs the upload calls; check by hand, don't assume):
  - The Space actually builds and serves a request end to end.
  - snapshot_download's local layout really does match what
    RoutingDecoderModel expects -- if it doesn't, RoutingDecoderModel's own
    FileNotFoundError on missing manifest.json will say so clearly at
    Space startup; read the build logs, don't guess.
  - bf16 actually runs correctly (not just slowly) on whatever GPU the
    Space ends up assigned -- untested on anything but this project's own
    training-server GPU (see PRD.md's HF Space deployment section for why
    this is flagged, not assumed fine).
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT_CHECKPOINT_NAME = "ekvachan-decoder-qwen-benchcorpus"
VISION_CHECKPOINT_NAME = "ekvachan-decoder-qwen-vision"

# The exact, minimal set of source files hf_space/app.py's import chain
# needs (traced by hand: serve/inference.py -> benchmarks.common.backends/
# .schema -> training.decoder_lora_lib -> eval.metrics -- nothing in this
# list pulls in vizdoom/datasets/accelerate/scikit-learn, so this stays a
# small, fast-installing Space rather than the full project dependency
# set). Copied as real directories (preserving package structure) into the
# Space repo so `from serve.inference import RoutingDecoderModel` resolves
# exactly as it does when this project is run locally -- no packaging
# tricks, no pip-installing this repo as a library.
VENDOR_PATHS = [
    "serve/__init__.py",
    "serve/inference.py",
    "benchmarks/__init__.py",
    "benchmarks/common/__init__.py",
    "benchmarks/common/backends.py",
    "benchmarks/common/schema.py",
    "training/__init__.py",
    "training/decoder_lora_lib.py",
    "eval/__init__.py",
    "eval/metrics.py",
]


def upload_model(model_repo: str, checkpoints_dir: Path, dry_run: bool) -> None:
    from huggingface_hub import HfApi

    for name in (TEXT_CHECKPOINT_NAME, VISION_CHECKPOINT_NAME):
        manifest = checkpoints_dir / name / "manifest.json"
        if not manifest.exists():
            print(f"FATAL: {manifest} does not exist -- refusing to upload a partial/wrong checkpoints_dir.")
            sys.exit(1)

    api = HfApi()
    print(f"Creating (or reusing) model repo {model_repo!r} ...")
    if not dry_run:
        api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)

    model_card = REPO_ROOT / "hf_model" / "README.md"
    print(f"Uploading {checkpoints_dir} -> {model_repo} (this can take a while -- LoRA weights only, "
          f"not the base model, but still real file transfer) ...")
    if not dry_run:
        api.upload_folder(
            repo_id=model_repo,
            repo_type="model",
            folder_path=str(checkpoints_dir),
            allow_patterns=[f"{TEXT_CHECKPOINT_NAME}/*", f"{VISION_CHECKPOINT_NAME}/*"],
        )
        if model_card.exists():
            api.upload_file(
                repo_id=model_repo,
                repo_type="model",
                path_or_fileobj=str(model_card),
                path_in_repo="README.md",
            )
    print(f"Done: https://huggingface.co/{model_repo}")


def _stage_space_contents(staging_dir: Path) -> None:
    """Copies hf_space/'s own files plus every VENDOR_PATHS entry into
    staging_dir, preserving relative package paths, so the result is a
    single self-contained tree to upload -- app.py sitting right next to
    the real serve/, benchmarks/, training/, eval/ directories it imports
    from, exactly like this project's own repo root."""
    shutil.copytree(REPO_ROOT / "hf_space", staging_dir, dirs_exist_ok=True)
    for rel_path in VENDOR_PATHS:
        src = REPO_ROOT / rel_path
        if not src.exists():
            print(f"FATAL: expected source file {src} does not exist -- VENDOR_PATHS is out of date "
                  f"with the real import chain (see this script's own docstring).", file=sys.stderr)
            sys.exit(1)
        dst = staging_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def upload_space(space_repo: str, dry_run: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Creating (or reusing) Space repo {space_repo!r} (sdk=gradio, no Docker) ...")
    if not dry_run:
        api.create_repo(repo_id=space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            staging_dir = Path(tmp) / "space"
            _stage_space_contents(staging_dir)
            print(f"Uploading {staging_dir} (hf_space/ + vendored source: {', '.join(VENDOR_PATHS)}) "
                  f"-> {space_repo} ...")
            api.upload_folder(
                repo_id=space_repo,
                repo_type="space",
                folder_path=str(staging_dir),
            )
    print(f"Done: https://huggingface.co/spaces/{space_repo}")
    print("REMAINING MANUAL STEPS (this script cannot do these):")
    print("  1. Open the Space's Settings tab and pick its Hardware -- ZeroGPU if available on this "
          "account, otherwise T4-small (the cheapest GPU tier, $0.40/hr at last check -- verify the "
          "current rate in the UI, it can change).")
    print("  2. If on paid/dedicated hardware (not ZeroGPU): set the shortest available "
          "'sleep after inactivity' option in Settings, and manually Pause the Space from Settings "
          "whenever it isn't actively being demoed -- that's the only way to fully stop billing.")
    print("  3. Send one real test request and actually read the response before calling this done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-repo", default="abhi6168/ekvachan-decoder")
    parser.add_argument("--space-repo", default="abhi6168/ekvachan")
    parser.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-space", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen, upload nothing")
    args = parser.parse_args()

    if not args.skip_model:
        upload_model(args.model_repo, args.checkpoints_dir, args.dry_run)
    if not args.skip_space:
        upload_space(args.space_repo, args.dry_run)


if __name__ == "__main__":
    main()
