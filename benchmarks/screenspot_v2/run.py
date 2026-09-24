"""
CLI entry point for the ScreenSpot-v2 grounding-as-choice harness. See
benchmarks/screenspot_v2/README.md for what this does and does not prove, and
loader.py for the real item counts and the reframing logic.

Usage:
  # Real dataset, against the vision-trained checkpoint (PRD.md 5.2b/13a.11 --
  # checkpoints/ekvachan-decoder-qwen-vision, once training finishes):
  python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema \\
      --checkpoint-dir checkpoints/ekvachan-decoder-qwen-vision

  # Vision harness wiring self-test -- synthetic fixture, no checkpoint,
  # no torch/transformers/peft, no better-jev-bench clone needed:
  python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema-mock --selftest

  # Real dataset, filter-only proof (0 items need a model loaded to count
  # them -- see benchmarks/common/harness.py's docstring):
  python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema-mock
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.harness import make_arg_parser, run_harness
from benchmarks.screenspot_v2 import loader


def main() -> dict:
    parser = make_arg_parser("benchmarks.screenspot_v2.run")
    parser.add_argument(
        "--bench-repo-dir", default=None,
        help="path to a better-jev-bench clone (with `bjb build --datasets screenspot_v2` already "
             "run) whose data/images/screenspot_v2/ cache holds the real screenshots. Default: "
             "loader.DEFAULT_BENCH_REPO_DIR (a `bench-repo` sibling of this repo). Unused with "
             "--selftest, which ships its own two tiny synthetic images.",
    )
    args = parser.parse_args()

    if args.backend not in (
        "decoder-vision-multischema", "decoder-vision-multischema-mock",
    ) and args.backend != "mock":
        print(
            f"NOTE: --backend {args.backend!r} cannot use ScreenSpot-v2's images (every real item "
            "here carries one) -- it will attempt every item text-only and score accordingly. Use "
            "decoder-vision-multischema[-mock] to actually use the screenshot.",
            file=sys.stderr,
        )

    if args.selftest:
        from benchmarks.screenspot_v2.fixtures.selftest import SELFTEST_PATH, load_selftest_items

        items, stats = load_selftest_items()
        provenance = {
            "source": "synthetic fixture authored in this repo for harness self-test, NOT real "
                      "ScreenSpot-v2 data",
            "path": str(SELFTEST_PATH.relative_to(REPO_ROOT)),
            "reframing_stats": stats,
        }
    else:
        items, stats = loader.load_heldout(bench_repo_dir=args.bench_repo_dir)
        provenance = {
            "source_repo": "https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2",
            "license": "Apache-2.0 (verified 2026-09-24 via HF Hub API cardData['license'], "
                      "independently of better-jev-bench's own verification in "
                      "datasets/screenspot_v2/manifest.toml; obligation: attribution, see NOTICE.md)",
            "vendored_via": "better-jev-bench's `bjb export --slice heldout --datasets "
                            "screenspot_v2` (better-jev-bench is eval-only for this dataset -- "
                            "`--slice public` emits 0 records, verified by running it -- see "
                            "loader.py's module docstring)",
            "vendored_file": "vendored/heldout.jsonl",
            "n_exported_records": stats["n_records_in"],
            "n_reframed_items": stats["n_items_out"],
            "n_excluded_pool_too_small": stats["n_excluded_pool_too_small"],
            "pool_size_histogram": stats["pool_size_histogram"],
            "note": (
                "858 real exported items (not the manifest's headline 898 = 40 public + 858 "
                "heldout -- the 40 public rows are structurally unreachable via `bjb export "
                "--slice public` for an eval_only dataset, and `--slice heldout` only ever reads "
                "the 858-row heldout file, not both -- see loader.py). Reframed from "
                f"{stats['n_images']} distinct images into {stats['n_items_out']} genuine N-way "
                "grounding-as-choice items (N = 2 to 6, real per-image annotated-element pools, "
                "no invented distractors); 0 items excluded for too-small a pool on this real "
                "export. See benchmarks/screenspot_v2/README.md.",
            ),
            "images": "referenced from better-jev-bench's content-addressed cache, not vendored "
                     "into this repo -- see README.md's 'Vendoring: images vs. metadata'.",
        }

    return run_harness(benchmark="screenspot_v2", args=args, items=items, dataset_provenance=provenance)


if __name__ == "__main__":
    main()
