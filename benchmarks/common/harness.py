"""
Shared run loop: filter -> (maybe) call a backend on the supported items ->
score with eval/metrics.py -> write a PRD.md Section 8.2 evidence bundle.
benchmarks/jevbench/run.py, benchmarks/jabr_v2/run.py, and
benchmarks/screenspot_v2/run.py are thin CLI wrappers around run_harness()
below -- loading the dataset and describing its provenance is the only
genuinely benchmark-specific part.

Deliberately structured so that filtering happens *before* any backend is
built: for the fixed-schema path (`--backend in_process|http|mock`),
benchmarks/common/schema.FIXED_CHECKPOINT_LABELS is enough to know which
items are even answerable; for either multischema path (`--backend
decoder-multischema[-mock]` or `--backend decoder-vision-multischema[-mock]`),
benchmarks/common/schema.DECODER_MULTISCHEMA_MAX_OPTIONS (or --max-options)
is enough. All are dataset-only checks with no torch/transformers/peft/
checkpoint dependency. A backend (which may need all of those, plus pillow/
torchvision for the vision arm) is only constructed if there's at least one
supported item to actually run. That is what lets a real run against the
real vendored datasets execute -- and produce a real, honest evidence bundle
even when 0 items are supported -- in an environment with no ML stack
installed at all, which is exactly the environment this was built and tested
in. See benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.common import backends
from benchmarks.common.evidence import write_bundle
from benchmarks.common.items import Item
from benchmarks.common.schema import DECODER_MULTISCHEMA_MAX_OPTIONS, FIXED_CHECKPOINT_LABELS
from benchmarks.common.schema_filter import filter_supported, filter_supported_multischema

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Backends that answer ANY choice question with 2..max_options options,
# rather than one hardcoded label set -- see benchmarks/common/backends.py
# and PRD.md Section 14 Q4 / 13a.5. The vision arm (PRD.md 5.2b/13a.11)
# answers the exact same 2..max_options schema, just optionally with an image
# attached to the item -- `filter_supported_multischema` needs no changes at
# all for it (an item's option *count* is what's being filtered on, and a
# `choice` item with an image is still, structurally, a `choice` item).
MULTISCHEMA_BACKENDS = {
    "decoder-multischema", "decoder-multischema-mock",
    "decoder-vision-multischema", "decoder-vision-multischema-mock",
}
VISION_BACKENDS = {"decoder-vision-multischema", "decoder-vision-multischema-mock"}


def build_backend(args: argparse.Namespace):
    if args.backend == "in_process":
        ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else None
        return backends.InProcessBackend(ckpt)
    if args.backend == "http":
        return backends.HTTPBackend(args.http_endpoint)
    if args.backend == "mock":
        return backends.MockBackend()
    if args.backend == "random":
        return backends.RandomChoiceBackend(seed=args.seed)
    if args.backend == "decoder-multischema":
        if not args.checkpoint_dir:
            raise ValueError(
                "--backend decoder-multischema requires --checkpoint-dir -- there is no default "
                "here the way in_process has serve.inference.load_default(), because the wide-"
                "schema checkpoint isn't vendored in this repo (checkpoints/ is gitignored; see "
                "PRD.md Section 12 and 13a.5's 'in progress' checkpoint)."
            )
        return backends.DecoderMultischemaBackend(
            Path(args.checkpoint_dir),
            base_model=args.base_model,
            apply_temperature=args.decoder_apply_temperature,
        )
    if args.backend == "decoder-multischema-mock":
        return backends.DecoderMultischemaMockBackend(max_options=args.max_options or DECODER_MULTISCHEMA_MAX_OPTIONS)
    if args.backend == "decoder-vision-multischema":
        if not args.checkpoint_dir:
            raise ValueError(
                "--backend decoder-vision-multischema requires --checkpoint-dir -- point it at "
                "checkpoints/ekvachan-decoder-qwen-vision (PRD.md 13a.11) or an equivalent "
                "train_decoder_lora_general.py output dir trained on vision-bearing data. "
                "checkpoints/ is gitignored, so there is no default the way in_process has one."
            )
        return backends.DecoderVisionMultischemaBackend(
            Path(args.checkpoint_dir),
            base_model=args.base_model,
            apply_temperature=args.decoder_apply_temperature,
            max_pixels=args.max_pixels,
        )
    if args.backend == "decoder-vision-multischema-mock":
        return backends.DecoderVisionMultischemaMockBackend(
            max_options=args.max_options or DECODER_MULTISCHEMA_MAX_OPTIONS
        )
    raise ValueError(f"unknown backend {args.backend!r}")


def make_arg_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument(
        "--backend",
        choices=[
            "in_process", "http", "mock", "random",
            "decoder-multischema", "decoder-multischema-mock",
            "decoder-vision-multischema", "decoder-vision-multischema-mock",
        ],
        default="in_process",
        help="in_process imports serve.inference.EncoderChoiceModel directly (needs torch/"
             "transformers and a real fixed-schema checkpoint on disk); http calls a running "
             "serve/server.py over the wire (needs no local ML stack, but a server must already be "
             "up); mock is a non-trained fixed-schema wiring self-test, only meaningful together "
             "with --selftest. random is a uniform-random baseline over whatever options are "
             "offered -- unlike mock, a real publishable control (PRD.md 8.1d Finding 2), seeded by "
             "--seed. decoder-multischema loads a text-only LoRA adapter checkpoint from "
             "train_decoder_lora_wideschema.py/_benchcorpus.py/_general.py (needs torch/transformers/"
             "peft and --checkpoint-dir pointing at one -- see PRD.md 13a.5); "
             "decoder-multischema-mock is that path's non-trained wiring self-test. "
             "decoder-vision-multischema loads a vision-bearing checkpoint from "
             "train_decoder_lora_general.py (needs torch/transformers/peft/pillow/torchvision and "
             "--checkpoint-dir pointing at one -- see PRD.md 5.2b/13a.11); it can answer both "
             "text-only and image-bearing choice items. decoder-vision-multischema-mock is that "
             "path's non-trained wiring self-test (it never looks at the image).",
    )
    p.add_argument("--http-endpoint", default="http://127.0.0.1:8000")
    p.add_argument("--checkpoint-dir", default=None,
                    help="for in_process: override serve.inference's default checkpoint path. For "
                         "decoder-multischema/decoder-vision-multischema: REQUIRED, path to a "
                         "training/train_decoder_lora_*.py output dir containing manifest.json + "
                         "adapter/.")
    p.add_argument("--base-model", default=None,
                    help="decoder-multischema/decoder-vision-multischema only: override the base "
                         "model id instead of using the one recorded in the checkpoint's "
                         "manifest.json (e.g. to point at a locally-cached copy of the same model).")
    p.add_argument("--max-options", type=int, default=None,
                    help="decoder-multischema[-mock]/decoder-vision-multischema[-mock] only: "
                         "override the option-count cap used both to filter items before any "
                         "backend is built and (for the real backends) to cross-check against the "
                         "loaded checkpoint's own manifest['max_options']. Default: "
                         "benchmarks.common.schema.DECODER_MULTISCHEMA_MAX_OPTIONS (588, PRD 5.1b).")
    p.add_argument("--max-pixels", type=int, default=None,
                    help="decoder-vision-multischema only: AutoProcessor cap on image area, passed "
                         "straight through to it. Default: the checkpoint manifest's own recorded "
                         "max_pixels (PRD.md 5.2b: 256*28*28 caps any screenshot at 180 image "
                         "tokens, the training-time default once vision rows are present).")
    p.add_argument("--decoder-apply-temperature", action="store_true",
                    help="decoder-multischema/decoder-vision-multischema only: apply the checkpoint "
                         "manifest's fitted temperature instead of raw probabilities. Off by "
                         "default -- PRD.md 13a.2/13a.5 found this architecture's temperature-"
                         "scaling procedure makes ECE/Brier worse, not better; 'use raw for now' is "
                         "that section's own conclusion.")
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
    multischema = args.backend in MULTISCHEMA_BACKENDS
    is_vision_backend = args.backend in VISION_BACKENDS
    max_options = args.max_options or DECODER_MULTISCHEMA_MAX_OPTIONS

    if multischema:
        filt = filter_supported_multischema(items, max_options)
    else:
        filt = filter_supported(items, FIXED_CHECKPOINT_LABELS)
        label_to_idx = {l: i for i, l in enumerate(FIXED_CHECKPOINT_LABELS)}

    raw_rows: list[dict] = []
    probs_rows: list[list[float]] = []
    label_idx_rows: list[int] = []

    if filt.supported:
        backend = build_backend(args)
        if multischema:
            if backend.max_options != max_options:
                raise RuntimeError(
                    f"{args.backend} backend reports max_options={backend.max_options!r}, which "
                    f"disagrees with max_options={max_options!r} used to filter this run -- "
                    "refusing to score against a schema-cap mismatch."
                )
        else:
            if list(backend.labels) != FIXED_CHECKPOINT_LABELS:
                raise RuntimeError(
                    f"{args.backend} backend reports labels={backend.labels!r}, which disagrees with "
                    f"benchmarks.common.schema.FIXED_CHECKPOINT_LABELS={FIXED_CHECKPOINT_LABELS!r} used "
                    "to filter this run -- refusing to score against a schema mismatch."
                )
        n_with_image = 0
        for item in filt.supported:
            image_path = item.image_path
            if image_path is not None:
                n_with_image += 1
                if not is_vision_backend:
                    # Not an error: a non-vision backend simply can't use the image, same as
                    # `instructions` being unused by the fixed-schema backends. Still recorded
                    # (n_with_image below) so a run against the wrong backend is visibly, not
                    # silently, leaving information on the table.
                    pass
            if multischema:
                out = backend.predict_choice(
                    item.state, item.options, instructions=item.instructions, image_path=image_path
                )
            else:
                out = backend.predict_choice(item.state, item.options, image_path=image_path)
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
                "image_path": image_path,
            })
            if multischema:
                n = len(item.options)
                padded = [out["probabilities"].get(opt, 0.0) for opt in item.options]
                padded.extend([0.0] * (max_options - n))
                probs_rows.append(padded)
                label_idx_rows.append(item.options.index(item.expected))
            else:
                probs_rows.append([out["probabilities"][l] for l in FIXED_CHECKPOINT_LABELS])
                label_idx_rows.append(label_to_idx[item.expected])
        model_info = {"instantiated": True, **backend.describe()}
        backend_name = backend.name
    else:
        n_with_image = 0
        model_info = {
            "instantiated": False,
            "backend_requested": args.backend,
            "reason": (
                "0 items in this run had between 2 and max_options options -- no model needed to "
                "be loaded to know that (see benchmarks/common/schema.py)."
                if multischema else
                "0 items in this run matched the checkpoint's fixed schema -- no model "
                "needed to be loaded to know that (see benchmarks/common/schema.py)."
            ),
        }
        backend_name = args.backend

    n_columns = max_options if multischema else len(FIXED_CHECKPOINT_LABELS)
    if probs_rows:
        import numpy as np
        from eval.metrics import full_report
        report = full_report(np.array(probs_rows), np.array(label_idx_rows), n_columns)
    else:
        report = {"n": 0, "accuracy": None, "brier": None, "ece": None}

    n_supported_key = "n_supported_by_multischema_cap" if multischema else "n_supported_by_fixed_schema"
    results_summary = {
        "n_items_considered": len(items),
        n_supported_key: len(filt.supported),
        "n_unsupported": len(filt.unsupported),
        "n_supported_with_image": n_with_image,
        **report,
        "is_complete_benchmark_score": (
            not args.selftest and len(items) > 0 and len(filt.supported) == len(items)
        ),
        "note": (
            (
                f"{n_supported_key} is how many items a decoder/multischema model could "
                f"legitimately attempt (a 'choice' item with between 2 and max_options={max_options} "
                "options, whose gold label is one of them). n_supported_with_image is how many of "
                "those items also carry a real screenshot (PRD.md 5.2b/13a.11) -- only "
                "decoder-vision-multischema[-mock] backends actually use it; every other backend "
                "here ignores an item's image the same way it ignores `instructions` if it doesn't "
                "need it. is_complete_benchmark_score is true only when every real (non-selftest) "
                "item was both supported and attempted. See PRD.md Section 14 Q4 / 13a.5 and "
                "benchmarks/README.md before treating this file as a benchmark ranking."
            ) if multischema else (
                "n_supported_by_fixed_schema is how many items this checkpoint could legitimately "
                "attempt (a 'choice' item whose options are exactly the checkpoint's fixed 3-way "
                "schema). is_complete_benchmark_score is true only when every real (non-selftest) "
                "item was both supported and attempted -- as of this checkpoint that has never "
                "happened (PRD.md Section 10a); read benchmarks/README.md and the per-benchmark "
                "README before treating this file as a benchmark ranking."
            )
        ),
    }

    schema_filter_summary = (
        {"max_options": max_options, "unsupported_reasons": dict(filt.unsupported_reasons)}
        if multischema else
        {"checkpoint_labels": FIXED_CHECKPOINT_LABELS, "unsupported_reasons": dict(filt.unsupported_reasons)}
    )

    run_id = f"{benchmark}-{'selftest-' if args.selftest else ''}{backend_name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest_path, raw_path = write_bundle(
        out_dir=Path(args.out_dir),
        run_id=run_id,
        benchmark=benchmark,
        seed=args.seed,
        dataset_provenance=dataset_provenance,
        schema_filter_summary=schema_filter_summary,
        model_info=model_info,
        results_summary=results_summary,
        raw_rows=raw_rows,
        extra={"harness_selftest": bool(args.selftest)},
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {raw_path}")
    print(json.dumps(results_summary, indent=2, default=str))
    return results_summary
