"""
PRD 5.1b's continue-train fallback: "a continue-train from checkpoints/
ekvachan-decoder-qwen-vision on 15,000 wide + ~15,000 replay rows lands in
~4h30m instead [of the full 93,000-row retrain] -- defensible precisely
because <=26-option prompts are byte-identical, so the wide rows are
genuinely new capability rather than a contradicting relabel."

Builds a small dedicated train split from the already-validated
data/processed/wide_slice (every row already passed the token-budget
assertion there -- not re-run here, would be redundant): all 11,250 wide
train rows (5 tasks, clinc150 excluded -- see build_wide_slice.py) plus a
random 15,000-row replay sample of the original (narrow) rows, so the
continue-train run sees new capability without catastrophically forgetting
the existing one. eval_id/eval_ood are reused from wide_slice unchanged
(evaluation is cheap; no need to shrink it).
"""

import random
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_REPLAY = 15_000


def main():
    rng = random.Random(SEED)
    train_ds = load_from_disk(str(REPO_ROOT / "data/processed/wide_slice/train"))

    is_wide = [len(o) > 26 for o in train_ds["options"]]
    wide_idx = [i for i, w in enumerate(is_wide) if w]
    narrow_idx = [i for i, w in enumerate(is_wide) if not w]
    print(f"wide rows: {len(wide_idx)}, narrow rows available for replay: {len(narrow_idx)}")

    replay_idx = rng.sample(narrow_idx, min(N_REPLAY, len(narrow_idx)))
    keep_idx = sorted(wide_idx + replay_idx)
    continue_train_ds = train_ds.select(keep_idx).shuffle(seed=SEED)
    print(f"continue-train set: {len(continue_train_ds)} rows ({len(wide_idx)} wide + {len(replay_idx)} replay)")

    out_dir = REPO_ROOT / "data/processed/wide_slice_continue"
    continue_train_ds.save_to_disk(str(out_dir / "train"))

    # eval_id/eval_ood: symlink-equivalent copy (small, cheap) so the training
    # script's --data-dir contract (a dir with train/eval_id/eval_ood) holds
    # without re-deriving already-validated eval data.
    for split in ("eval_id", "eval_ood"):
        src = load_from_disk(str(REPO_ROOT / "data/processed/wide_slice" / split))
        src.save_to_disk(str(out_dir / split))
        print(f"{split}: {len(src)} rows (unchanged copy from wide_slice)")


if __name__ == "__main__":
    main()
