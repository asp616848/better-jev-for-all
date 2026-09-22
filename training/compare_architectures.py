"""
Compare the two Phase 1 training arms (PRD.md Section 5.1/3.1a) using
whatever `manifest.json` files actually exist under the given checkpoint
directories.

Field names below were read directly from both scripts' manifest-writing
code near the end of `main()` (`training/train_encoder.py`,
`training/train_decoder_lora.py`) rather than assumed. The two schemas are
close but NOT identical:

  Shared by both:
    base_model, seed, epochs, train_size, calib_size, test_size,
    temperature, raw_report {n, accuracy, brier, ece},
    calibrated_report {n, accuracy, brier, ece}, timestamp_utc,
    total_train_seconds

  Encoder-only (train_encoder.py):
    batch_size, lr, brier_lambda, weight_sha256
    -- and no explicit "architecture" key at all, since that script only
    ever implements one architecture. This script labels it
    "encoder-classification-head" by inference from that absence, not from
    any field the script writes.

  Decoder-only (train_decoder_lora.py):
    architecture ("decoder-lora-restricted-logit"), lora_r, lora_alpha,
    skipped_oom_batches
    -- no batch_size/lr/brier_lambda/weight_sha256 (the LoRA/AdamW
    hyperparameters that would map to those aren't recorded in the
    manifest as written today).

Both scripts route their accuracy/Brier/ECE numbers through the same
eval/metrics.py, so those specific fields are directly, fairly comparable
to each other -- they are still each arm's own held-out NLI test split,
not a shared external benchmark (jabr-v2/JevBench). See PRD.md Section
13a.4 and Section 14 Q4, which this script's output feeds once the decoder
run finishes.

Usage:
    python -m training.compare_architectures \\
        --checkpoints checkpoints/ekvachan-base-run2 checkpoints/ekvachan-decoder-qwen-run1

A checkpoint directory with no manifest.json yet (run still in progress,
not started, or a typo'd path) is handled gracefully: it's reported and
excluded from the table, not treated as a fatal error.
"""

import argparse
import json
from pathlib import Path

# Fields written by both training scripts' manifests, in display order.
SHARED_FIELD_ROWS = [
    ("base_model", lambda m: m.get("base_model")),
    ("train_size", lambda m: m.get("train_size")),
    ("calib_size", lambda m: m.get("calib_size")),
    ("test_size", lambda m: m.get("test_size")),
    ("epochs", lambda m: m.get("epochs")),
    ("temperature", lambda m: m.get("temperature")),
    ("accuracy (raw)", lambda m: m.get("raw_report", {}).get("accuracy")),
    ("accuracy (calibrated)", lambda m: m.get("calibrated_report", {}).get("accuracy")),
    ("ECE (raw)", lambda m: m.get("raw_report", {}).get("ece")),
    ("ECE (calibrated)", lambda m: m.get("calibrated_report", {}).get("ece")),
    ("Brier (raw)", lambda m: m.get("raw_report", {}).get("brier")),
    ("Brier (calibrated)", lambda m: m.get("calibrated_report", {}).get("brier")),
    ("timestamp_utc", lambda m: m.get("timestamp_utc")),
]

# Fields present in only one script's manifest -- reported per-checkpoint
# below the shared table rather than padded into it as blanks.
ARCH_SPECIFIC_KEYS = [
    "batch_size", "lr", "brier_lambda", "weight_sha256",
    "lora_r", "lora_alpha", "skipped_oom_batches",
]


def load_manifest(checkpoint_dir: Path) -> dict | None:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def architecture_label(manifest: dict) -> str:
    # train_decoder_lora.py writes an explicit "architecture" field.
    # train_encoder.py doesn't (see module docstring) -- inferred, not read.
    return manifest.get("architecture", "encoder-classification-head")


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def seconds_to_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser(
        description="Compare ekVachan training-arm manifest.json files side by side "
                    "(PRD.md Section 14 Q4: encoder vs. decoder-LoRA architecture decision)."
    )
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoint directories, each expected to contain a manifest.json, "
             "e.g. --checkpoints checkpoints/ekvachan-base-run2 checkpoints/ekvachan-decoder-qwen-run1",
    )
    args = ap.parse_args()

    entries = [(Path(p).name, Path(p), load_manifest(Path(p))) for p in args.checkpoints]

    print(f"Comparing {len(entries)} checkpoint(s):\n")

    available = []
    for name, checkpoint_dir, manifest in entries:
        if manifest is None:
            print(
                f"  [{name}] NOT YET AVAILABLE -- no manifest.json at "
                f"{checkpoint_dir / 'manifest.json'}. The run may still be in progress, may not "
                f"have started, or the path may be wrong. Skipping from the comparison table below."
            )
        else:
            available.append((name, checkpoint_dir, manifest))

    if not available:
        print(
            "\nNo manifests available yet -- nothing to compare. Re-run this script once at "
            "least one training run has completed and written a manifest.json."
        )
        return

    print()

    columns = [name for name, _, _ in available]
    col_width = max(12, max(len(c) for c in columns) + 2)
    label_width = 24

    def print_row(label: str, values: list[str]) -> None:
        cells = [label.ljust(label_width)] + [v.ljust(col_width) for v in values]
        print(" | ".join(cells))

    print_row("field", columns)
    print("-" * (label_width + len(columns) * (col_width + 3)))

    print_row("architecture", [architecture_label(m) for _, _, m in available])
    for field_label, getter in SHARED_FIELD_ROWS:
        print_row(field_label, [fmt(getter(m)) for _, _, m in available])
    print_row(
        "total_train_time",
        [
            seconds_to_hms(m["total_train_seconds"]) if "total_train_seconds" in m else "-"
            for _, _, m in available
        ],
    )

    print()

    for name, _, manifest in available:
        extras = {k: manifest[k] for k in ARCH_SPECIFIC_KEYS if k in manifest}
        if not extras:
            continue
        if "weight_sha256" in extras:
            extras["weight_sha256"] = extras["weight_sha256"][:16] + "..."
        extras_str = ", ".join(f"{k}={v}" for k, v in extras.items())
        print(f"[{name}] fields not shared with the other manifest schema: {extras_str}")

    print(
        "\nNote: accuracy/ECE/Brier above are each measured on that arm's own held-out test split "
        "from training/data.py's NLI slice. Both training scripts route through the same "
        "eval/metrics.py, so the two arms' numbers are directly comparable to each other -- but "
        "neither is a result on a shared external benchmark (jabr-v2/ViZDoom/JevBench). "
        "See PRD.md Section 13a.4 and Section 14 Q4."
    )


if __name__ == "__main__":
    main()
