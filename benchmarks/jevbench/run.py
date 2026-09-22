"""
CLI entry point for the JevBench harness. See benchmarks/jevbench/README.md
for what this does and does not prove.

Usage:
  # Real dataset, in-process (needs torch/transformers + a real checkpoint at
  # checkpoints/ekvachan-base-run2, per serve/inference.py's load_default()):
  python -m benchmarks.jevbench.run

  # Real dataset, against a running serve/server.py:
  python -m benchmarks.jevbench.run --backend http --http-endpoint http://localhost:8000

  # Harness wiring self-test -- synthetic fixture, no checkpoint needed:
  python -m benchmarks.jevbench.run --backend mock --selftest
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.harness import make_arg_parser, run_harness
from benchmarks.jevbench import loader


def main() -> dict:
    parser = make_arg_parser("benchmarks.jevbench.run")
    args = parser.parse_args()

    if args.selftest:
        from benchmarks.jevbench.fixtures.selftest import SELFTEST_PATH, load_selftest_items

        items = load_selftest_items()
        provenance = {
            "source": "synthetic fixture authored in this repo for harness self-test, NOT real "
                      "JevBench data",
            "path": str(SELFTEST_PATH.relative_to(REPO_ROOT)),
            "n_items": len(items),
        }
    else:
        items = loader.load_public()
        provenance = {
            "source_repo": "https://github.com/fstandhartinger/jevbench",
            "license": "MIT (repo-level license verified via GitHub API 2026-09-22; every vendored "
                      "record also carries its own provenance.license == \"MIT\" field)",
            "vendored_files": sorted(p.name for p in loader.DATA_DIR.glob("*.jsonl")),
            "n_public_items": len(items),
            "n_total_incl_heldout": 534,
            "note": "231 of JevBench v1.2's 534 total decisions are public (original 72, easy 48, "
                    "hard 111); the remaining 303 (heldout 24, router 78, judge 68, easy-heldout 24, "
                    "109 more hard items) are deliberately unpublished by upstream and cannot be "
                    "vendored here -- see benchmarks/jevbench/README.md.",
        }

    return run_harness(benchmark="jevbench", args=args, items=items, dataset_provenance=provenance)


if __name__ == "__main__":
    main()
