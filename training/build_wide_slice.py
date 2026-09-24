"""
PRD 5.1b implementation checklist item 4: rebuild the data slice with
genuinely-wide rows, added alongside (not instead of) the existing
26-option subsets -- built on top of `data/processed/vision_slice`
(13a.11/13a.14's frozen text+vision mix) unchanged, per the same
fork-don't-edit policy this project has applied at every step.

New rows: `data/processed/bench_raw_wide/public` (`bjb export --max-options
588 --max-per-task 2500` against banking77/ledgar/cuad/massive/go_emotions),
filtered to genuinely-wide rows only (len(options) > 26) -- the export
included each dataset's other, already-narrow tasks too (cuad/clause_present
2-way, massive/scenario 18-way), which are dropped here since they'd
duplicate rows already in the training mix.

**Deliberate deviation from PRD 5.1b's own GPU-time-estimate table, stated
here rather than silently applied**: that table lists all 6 wide tasks
including clinc150 as training rows. This script excludes clinc150. PRD
5.1b's own "Eval design" section, two paragraphs later in the same PRD
section, requires "CLINC150 zero-shot-schema >= 95.5%" as an *unchanged*
regression instrument from the 13a.6 lineage -- and every script in that
lineage (build_wideschema_slice.py through build_benchcorpus_slice.py)
deliberately excludes CLINC150 from training for exactly this reason: it
stops being a zero-shot check the moment it's trained on. The GPU-time
table's inclusion of clinc150 as a training row contradicts the eval
design's own requirement two paragraphs later; this script resolves that
contradiction in favor of preserving the regression instrument, consistent
with unbroken project precedent, rather than silently doing what a
GPU-hours estimate table implied. 5 wide tasks instead of 6, 12,500 new
rows instead of 15,000, ~93m less GPU time than 5.1b's table estimated.

Split: 90% of the new wide rows go to train, 10% to eval_id (so the
eval_id_raw_by_question_type / by_source reporting `train_decoder_lora_
general.py` already does gets real per-width-bucket signal without a new
script). None go to eval_ood -- that split stays CLINC150-only, unchanged.

Re-runs the token-budget assertion over every row, old and new (13a.11's
own lesson: an assertion that only checks new rows is the one that misses
a stale-budget bug on the *old* rows -- see PRD 13a.11's own postmortem).
"""

import json
import random
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk

from training.build_vision_slice import VISION_MAX_PIXELS, _assert_within_token_budget

# The real trained checkpoint (checkpoints/ekvachan-decoder-qwen-vision) used
# max_length=1280, not build_vision_slice.VISION_MAX_LENGTH=768 -- confirmed
# against its own manifest.json. 1280 is the real production budget this
# continue-train run will use; budget-check against that, not the module default.
VISION_MAX_LENGTH = 1280

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
EXCLUDED_WIDE_TASK = "bjb:clinc150/intent"  # kept exclusively as the untrained zero-shot regression check


def main():
    rng = random.Random(SEED)

    wide_raw = load_from_disk(str(REPO_ROOT / "data/processed/bench_raw_wide/public"))
    wide_records = [r for r in wide_raw if len(r["options"]) > 26]
    assert not any(r["source"] == EXCLUDED_WIDE_TASK for r in wide_records), (
        "clinc150 leaked into the wide export -- it must never be trained on (see module docstring)"
    )
    for r in wide_records:
        r["images"] = []
    print(f"wide rows kept (>26 options, clinc150 excluded): {len(wide_records)}")
    by_source = {}
    for r in wide_records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print("by source:", by_source)

    rng.shuffle(wide_records)
    n_eval = int(0.10 * len(wide_records))
    wide_eval = wide_records[:n_eval]
    wide_train = wide_records[n_eval:]
    print(f"wide split: {len(wide_train)} train, {len(wide_eval)} eval_id")

    base_train = load_from_disk(str(REPO_ROOT / "data/processed/vision_slice/train"))
    base_eval_id = load_from_disk(str(REPO_ROOT / "data/processed/vision_slice/eval_id"))
    base_eval_ood = load_from_disk(str(REPO_ROOT / "data/processed/vision_slice/eval_ood"))

    def to_dataset(records, like):
        from datasets import Dataset
        return Dataset.from_list(records, features=like.features)

    train_ds = concatenate_datasets([base_train, to_dataset(wide_train, base_train)]).shuffle(seed=SEED)
    eval_id_ds = concatenate_datasets([base_eval_id, to_dataset(wide_eval, base_eval_id)]).shuffle(seed=SEED)
    eval_ood_ds = base_eval_ood  # unchanged -- CLINC150-only, stays untrained

    out_dir = REPO_ROOT / "data/processed/wide_slice"
    train_ds.save_to_disk(str(out_dir / "train"))
    eval_id_ds.save_to_disk(str(out_dir / "eval_id"))
    eval_ood_ds.save_to_disk(str(out_dir / "eval_ood"))

    print(f"train: {len(train_ds)} | eval_id: {len(eval_id_ds)} | eval_ood: {len(eval_ood_ds)}")

    # checklist item 4's own required safety check -- every row, old and new.
    base_model = "Qwen/Qwen3.5-4B"
    for name, ds in [("train", train_ds), ("eval_id", eval_id_ds), ("eval_ood", eval_ood_ds)]:
        text_records = [dict(r) for r in ds if not r["images"]]
        vision_records = [dict(r) for r in ds if r["images"]]
        print(f"asserting token budget for {name}: {len(text_records)} text + {len(vision_records)} vision rows...")
        _assert_within_token_budget(vision_records, text_records, base_model, VISION_MAX_LENGTH, VISION_MAX_PIXELS)

    stats = {
        "n_wide_rows_added": len(wide_records),
        "wide_rows_by_source": by_source,
        "excluded_task": EXCLUDED_WIDE_TASK,
        "exclusion_reason": "preserved as the untrained zero-shot regression check per 13a.6 lineage",
        "wide_train": len(wide_train),
        "wide_eval_id": len(wide_eval),
        "train_total": len(train_ds),
        "eval_id_total": len(eval_id_ds),
        "eval_ood_total": len(eval_ood_ds),
        "base_slice": "data/processed/vision_slice (13a.11/13a.14, unchanged pass-through)",
        "wide_source": "data/processed/bench_raw_wide/public (bjb export --max-options 588 --max-per-task 2500)",
        "seed": SEED,
    }
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
