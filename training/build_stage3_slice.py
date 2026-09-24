"""
Stage 3 (user's original 3-stage plan: "train with more data... focus on
improving the base model much more"): PRD 13a.18 found go_emotions/emotion
(28-way) the one clearly weak wide task at 60.6% accuracy, with only 2,500
of its 45,270 available items used in training. This script adds 3,500 new
go_emotions rows (bjb export --max-per-task 6000, seed=42 -- assumed to be
a stable superset of the earlier --max-per-task 2500 export at the same
seed, so the first 2,500 lines are skipped as probable duplicates of
already-trained rows; worst case a few overlaps slip through, which is
redundancy, not a correctness problem) plus a replay sample from the
already-proven wide_slice_continue_capped/train (itself already <=768-token
capped, so not re-checked) to avoid forgetting, for a second continue-train
pass from checkpoints/ekvachan-decoder-qwen-wide (not the original vision
checkpoint -- this continues from the best checkpoint so far).
"""

import json
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

from training.build_vision_slice import _assert_within_token_budget

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
VISION_MAX_LENGTH = 850  # matches the v6 run that produced ekvachan-decoder-qwen-wide
VISION_MAX_PIXELS = 200704
N_REPLAY = 10_000
SKIP_N = 2500  # assumed-duplicate rows from the earlier --max-per-task 2500 export


def main():
    rng = random.Random(SEED)

    goemo_path = Path("/data/interns/studentiotlab/ekvachan/bench-repo/data/processed/bench_raw_goemo_v2/public.jsonl")
    rows = [json.loads(l) for l in goemo_path.read_text().splitlines() if l.strip()]
    print(f"go_emotions v2 export: {len(rows)} total rows")
    new_rows = rows[SKIP_N:]
    print(f"treating rows [{SKIP_N}:] as new: {len(new_rows)} rows")
    for r in new_rows:
        r["images"] = []
        r.setdefault("instructions", r.get("instructions", ""))

    base_model = "Qwen/Qwen3.5-4B"
    print("asserting token budget for new go_emotions rows...")
    _assert_within_token_budget(
        [], new_rows, base_model, VISION_MAX_LENGTH, VISION_MAX_PIXELS,
    )

    rng.shuffle(new_rows)
    n_eval = int(0.10 * len(new_rows))
    new_eval = new_rows[:n_eval]
    new_train = new_rows[n_eval:]
    print(f"new split: {len(new_train)} train, {len(new_eval)} eval_id")

    base_train = load_from_disk(str(REPO_ROOT / "data/processed/wide_slice_continue_capped/train"))
    replay_idx = rng.sample(range(len(base_train)), min(N_REPLAY, len(base_train)))
    replay_ds = base_train.select(sorted(replay_idx))
    print(f"replay: {len(replay_ds)} rows from wide_slice_continue_capped/train")

    new_train_ds = Dataset.from_list(new_train, features=replay_ds.features)
    stage3_train = concatenate_datasets([replay_ds, new_train_ds]).shuffle(seed=SEED)

    base_eval_id = load_from_disk(str(REPO_ROOT / "data/processed/wide_slice_continue_capped/eval_id"))
    new_eval_ds = Dataset.from_list(new_eval, features=base_eval_id.features)
    stage3_eval_id = concatenate_datasets([base_eval_id, new_eval_ds]).shuffle(seed=SEED)

    stage3_eval_ood = load_from_disk(str(REPO_ROOT / "data/processed/wide_slice_continue_capped/eval_ood"))

    out_dir = REPO_ROOT / "data/processed/stage3_slice"
    stage3_train.save_to_disk(str(out_dir / "train"))
    stage3_eval_id.save_to_disk(str(out_dir / "eval_id"))
    stage3_eval_ood.save_to_disk(str(out_dir / "eval_ood"))

    stats = {
        "goemo_v2_total_rows": len(rows),
        "goemo_v2_new_rows_used": len(new_rows),
        "goemo_new_train": len(new_train),
        "goemo_new_eval_id": len(new_eval),
        "replay_rows": len(replay_ds),
        "train_total": len(stage3_train),
        "eval_id_total": len(stage3_eval_id),
        "eval_ood_total": len(stage3_eval_ood),
        "continues_from": "checkpoints/ekvachan-decoder-qwen-wide",
        "seed": SEED,
    }
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
