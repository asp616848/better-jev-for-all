"""
CLI entry point for the jabr-v2 harness. See benchmarks/jabr_v2/README.md for
what this does and does not prove, and for how jabr-v2 was actually found
(PRD.md's "no confirmed public repo" for jabr-v2 turned out to be wrong --
see that README for the correction and the exact evidence).

Usage:
  # Real dataset (v1 + v2, 947 cases), in-process (needs torch/transformers +
  # a real checkpoint at checkpoints/ekvachan-base-run2):
  python -m benchmarks.jabr_v2.run

  # Real dataset, against a running serve/server.py:
  python -m benchmarks.jabr_v2.run --backend http --http-endpoint http://localhost:8000

  # Harness wiring self-test -- synthetic fixture, no checkpoint needed:
  python -m benchmarks.jabr_v2.run --backend mock --selftest

  # Real dataset, against a decoder/multischema checkpoint (PRD.md 13a.5 --
  # e.g. training/train_decoder_lora_wideschema.py's output dir, once the
  # wide-schema run lands):
  python -m benchmarks.jabr_v2.run --backend decoder-multischema \\
      --checkpoint-dir checkpoints/ekvachan-decoder-qwen-wideschema

  # Multischema harness wiring self-test -- no checkpoint needed:
  python -m benchmarks.jabr_v2.run --backend decoder-multischema-mock
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.harness import make_arg_parser, run_harness
from benchmarks.jabr_v2 import loader


def main() -> dict:
    parser = make_arg_parser("benchmarks.jabr_v2.run")
    args = parser.parse_args()

    if args.selftest:
        from benchmarks.jabr_v2.fixtures.selftest import SELFTEST_PATH, load_selftest_items

        items = load_selftest_items()
        provenance = {
            "source": "synthetic fixture authored in this repo for harness self-test, NOT real "
                      "jabr-v2 data",
            "path": str(SELFTEST_PATH.relative_to(REPO_ROOT)),
            "n_items": len(items),
        }
    else:
        items = loader.load_public()
        provenance = {
            "source_repo": "https://github.com/jabr/classifier-benchmark",
            "license": "CC0 1.0 Universal (public domain) -- verified by reading LICENSE at the "
                      "repo root 2026-09-22; GitHub's API license-detector mislabels it 'other' "
                      "because of a short prose preamble before the standard CC0 legal text.",
            "vendored_files": sorted(p.name for p in loader.DATA_DIR.glob("*.toml")),
            "n_items": len(items),
            "suites": {"v1": "8 tasks / 78 cases, upstream-locked (cases/hashes.json)",
                       "v2": "49 tasks / 866 cases as actually vendored 2026-09-22 -- 3 fewer than "
                             "the 869 the repo's own README/results/v1v2-summary.md report, because "
                             "v2.toml is upstream 'under review' / unlocked (no hash pin yet) and has "
                             "apparently drifted since that summary was generated: commit_intent has "
                             "15 cases here vs 16 published, grammar_issue 21 vs 22, "
                             "fair_housing_violation 18 vs 19. See benchmarks/jabr_v2/README.md."},
            "correction": "PRD.md Section 8.1 was written assuming jabr-v2 had no confirmed "
                          "public repo. It does: github.com/jabr/classifier-benchmark, linked "
                          "directly from Von's own README as 'the jabr v2 benchmark'.",
        }

    return run_harness(benchmark="jabr_v2", args=args, items=items, dataset_provenance=provenance)


if __name__ == "__main__":
    main()
