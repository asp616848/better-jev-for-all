"""
Shared run loop: filter -> (maybe) call a backend on the supported items ->
score with eval/metrics.py -> write a PRD.md Section 8.2 evidence bundle.
benchmarks/jevbench/run.py and benchmarks/jabr_v2/run.py are thin CLI
wrappers around run_harness() below -- loading the dataset and describing its
provenance is the only genuinely benchmark-specific part.

Deliberately structured so that filtering happens *before* any backend is
built: benchmarks/common/schema.FIXED_CHECKPOINT_LABELS is enough to know
which items are even answerable, and that's dataset-only work with no torch/
transformers/checkpoint dependency. A backend (which may need all of those)
is only constructed if there's at least one supported item to actually run.
That is what lets a real run against the real vendored datasets execute -- and
produce a real, honest 0-supported-items evidence bundle -- in an environment
with no ML stack installed at all, which is exactly the environment this was
built and tested in. See benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.common import backends
from benchmarks.common.evidence import write_bundle
from benchmarks.common.items import Item
from benchmarks.common.schema import FIXED_CHECKPOINT_LABELS
from benchmarks.common.schema_filter import filter_supported

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def build_backend(args: argparse.Namespace):
    if args.backend == "in_process":
        ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else None
        return backends.InProcessBackend(ckpt)
    if args.backend == "http":
        return backends.HTTPBackend(args.http_endpoint)
    if args.backend == "mock":
        return backends.MockBackend()
    raise ValueError(f"unknown backend {args.backend!r}")


def make_arg_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument(
        "--backend", choices=["in_process", "http", "mock"], default="in_process",
        help="in_process imports serve.inference.EncoderChoiceModel directly (needs torch/"
             "transformers and a real checkpoint on disk); http calls a running serve/server.py "
             "over the wire (needs no local ML stack, but a server must already be up); mock is "
             "a non-trained wiring self-test, only meaningful together with --selftest.",
    )
    p.add_argument("--http-endpoint", default="http://127.0.0.1:8000")
    p.add_argument("--checkpoint-dir", default=None, help="override serve.inference's default checkpoint path")
    p.add_argument(
        "--selftest", action="store_true",
        help="run against benchmarks/<name>/fixtures/selftest.* (synthetic, authored in this repo) "
             "instead of the real vendored dataset under benchmarks/<name>/vendored/",
    )
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument(
        "--seed", type=int, default=0,
        help="recorded in the evidence manifest for PRD.md 8.2's sake. This harness has no "
             "stochastic component today (a fixed checkpoint's deterministic forward pass, "
             "argmax) -- kept for the evidence-bundle schema and any future resampling/CI step.",
    )
    return p


def run_harness(*, benchmark: str, args: argparse.Namespace, items: list[Item], dataset_provenance: dict) -> dict:
    filt = filter_supported(items, FIXED_CHECKPOINT_LABELS)
    label_to_idx = {l: i for i, l in enumerate(FIXED_CHECKPOINT_LABELS)}

    raw_rows: list[dict] = []
    probs_rows: list[list[float]] = []
    label_idx_rows: list[int] = []

    if filt.supported:
        backend = build_backend(args)
        if list(backend.labels) != FIXED_CHECKPOINT_LABELS:
            raise RuntimeError(
                f"{args.backend} backend reports labels={backend.labels!r}, which disagrees with "
                f"benchmarks.common.schema.FIXED_CHECKPOINT_LABELS={FIXED_CHECKPOINT_LABELS!r} used "
                "to filter this run -- refusing to score against a schema mismatch."
            )
        for item in filt.supported:
            out = backend.predict_choice(item.state, item.options)
            correct = out["choice"] == item.expected
            raw_rows.append({
                "item_id": item.item_id,
                "item_sha256": item.sha256(),
                "source_split": item.source_split,
                "state": item.state,
                "options": item.options,
                "expected": item.expected,
                "prediction": out["choice"],
                "probabilities": out["probabilities"],
                "confidence": out["confidence"],
                "correct": correct,
                "latency_ms": out.get("latency_ms"),
            })
            probs_rows.append([out["probabilities"][l] for l in FIXED_CHECKPOINT_LABELS])
            label_idx_rows.append(label_to_idx[item.expected])
        model_info = {"instantiated": True, **backend.describe()}
        backend_name = backend.name
    else:
        model_info = {
            "instantiated": False,
            "backend_requested": args.backend,
            "reason": "0 items in this run matched the checkpoint's fixed schema -- no model "
                      "needed to be loaded to know that (see benchmarks/common/schema.py).",
        }
        backend_name = args.backend

    if probs_rows:
        import numpy as np
        from eval.metrics import full_report
        report = full_report(np.array(probs_rows), np.array(label_idx_rows), len(FIXED_CHECKPOINT_LABELS))
    else:
        report = {"n": 0, "accuracy": None, "brier": None, "ece": None}

    results_summary = {
        "n_items_considered": len(items),
        "n_supported_by_fixed_schema": len(filt.supported),
        "n_unsupported": len(filt.unsupported),
        **report,
        "is_complete_benchmark_score": (
            not args.selftest and len(items) > 0 and len(filt.supported) == len(items)
        ),
        "note": (
            "n_supported_by_fixed_schema is how many items this checkpoint could legitimately "
            "attempt (a 'choice' item whose options are exactly the checkpoint's fixed 3-way "
            "schema). is_complete_benchmark_score is true only when every real (non-selftest) "
            "item was both supported and attempted -- as of this checkpoint that has never "
            "happened (PRD.md Section 10a); read benchmarks/README.md and the per-benchmark "
            "README before treating this file as a benchmark ranking."
        ),
    }

    run_id = f"{benchmark}-{'selftest-' if args.selftest else ''}{backend_name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest_path, raw_path = write_bundle(
        out_dir=Path(args.out_dir),
        run_id=run_id,
        benchmark=benchmark,
        seed=args.seed,
        dataset_provenance=dataset_provenance,
        schema_filter_summary={
            "checkpoint_labels": FIXED_CHECKPOINT_LABELS,
            "unsupported_reasons": dict(filt.unsupported_reasons),
        },
        model_info=model_info,
        results_summary=results_summary,
        raw_rows=raw_rows,
        extra={"harness_selftest": bool(args.selftest)},
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {raw_path}")
    print(json.dumps(results_summary, indent=2, default=str))
    return results_summary
