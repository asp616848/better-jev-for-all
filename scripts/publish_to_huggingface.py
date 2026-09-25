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
  2. Uploads hf_space/ (app.py, Dockerfile, README.md) to a HF *Space*
     repo (sdk=docker). This does NOT create the Space or set its hardware
     to ZeroGPU -- that one-time step still has to be done by hand in the
     HF web UI (Settings -> Hardware), since the API for it depends on
     account tier/quota this script has no way to verify blind.

Real verification still owed after running this (this script only
authors + performs the upload calls; check by hand, don't assume):
  - The Space actually builds and serves a request end to end.
  - snapshot_download's local layout really does match what
    RoutingDecoderModel expects -- if it doesn't, RoutingDecoderModel's own
    FileNotFoundError on missing manifest.json will say so clearly at
    Space startup; read the build logs, don't guess.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT_CHECKPOINT_NAME = "ekvachan-decoder-qwen-benchcorpus"
VISION_CHECKPOINT_NAME = "ekvachan-decoder-qwen-vision"


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


def upload_space(space_repo: str, dry_run: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Creating (or reusing) Space repo {space_repo!r} (sdk=docker) ...")
    if not dry_run:
        api.create_repo(repo_id=space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            repo_id=space_repo,
            repo_type="space",
            folder_path=str(REPO_ROOT / "hf_space"),
        )
    print(f"Done: https://huggingface.co/spaces/{space_repo}")
    print("REMAINING MANUAL STEP: open the Space's Settings tab and set Hardware to ZeroGPU -- "
          "this script cannot do that part.")


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
