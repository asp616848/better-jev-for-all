"""
Fork of `build_primitives_slice.py`, built to widen its narrow noul/score data
(one dataset each -- BoolQ, a single 3-level sentiment set) using the real
better-jev-bench corpus (`abhijeet-labgpu:~/ekvachan/bench-repo`,
422,878 items across 8 Tier-A datasets and 11 tasks, all 3 primitives, 4
width strata) -- exactly what 13a.8 named as the next lever for `score`'s
weak spot ("more/wider score data is the right next lever, not a design
change") and what the owner's 2026-09-23 resequencing put first in the
roadmap (better-jev-bench, then retrain).

**Why fork again, not edit `build_primitives_slice.py` in place**: same
policy this project has applied at every step (`train_decoder_lora.py` ->
`_multischema` -> `_wideschema` -> `_primitives` -> this). That script's
output is the exact data behind the real, published 13a.8 numbers (choice
92.02%, noul 88.80%, score 75.40%). Editing it in place would make an
already-published result silently unreproducible from its own script.

**Where the data actually comes from**: `bjb export` (the bench repo's own
CLI) was already run for real on 2026-09-23, producing
`data/processed/bench_raw/{public,heldout}` -- 55,000 / 4,400 records in
ekVachan's exact eight-key record shape, independently verified against
`build_primitives_slice.py`'s own `_validate()` (see
`bench-repo/scripts/check_ekvachan_compat.py`, 5,500/5,500 records passed).
This script's job is narrower: combine that export with the existing
NLI/DBpedia-14 continuity anchor (so `choice` accuracy stays comparable
across runs) and keep CLINC150 held out entirely, exactly as 13a.6/13a.8
already did.

**One real, deliberate exclusion, not an oversight**: `bjb:clinc150/intent`
rows are dropped from both the bench public and held-out pools before use.
better-jev-bench's own CLINC150 loader is a real, independently-licensed
pull of the same underlying dataset this project has used as its zero-shot-
schema regression check since 13a.6 -- training on it here would silently
break that continuity (the check would no longer be zero-shot). CLINC150
stays sourced exclusively from `build_clinc_records()` below (copied
unchanged from `build_wideschema_slice.py`), held out of training entirely,
exactly as every prior run in this lineage has done.

Everything else -- NLI/DBpedia loading, validation, the record shape -- is
carried over unchanged from `build_primitives_slice.py`.

Run: `uv run python3 -m training.build_benchcorpus_slice`
"""

import collections
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_OPTIONS = 26

CLINC_MIN_OPTIONS = 15
CLINC_MAX_OPTIONS = 26

SEED = 42

N_TRAIN_NLI = 6_000
N_TRAIN_DBPEDIA = 6_000
N_EVAL_ID_NLI = 375
N_EVAL_ID_DBPEDIA = 375
N_EVAL_OOD_CLINC = 3_000

DBPEDIA_INSTRUCTIONS = (
    "Given the title and content of an encyclopedia article, choose which "
    "of the following category options the article best belongs to."
)
CLINC_INSTRUCTIONS = (
    "Given a user's spoken request, transcribed as text, choose which of "
    "the following intent options it expresses. Only choose \"oos\" if it "
    "is one of the listed options and none of the other options match."
)

BENCH_RAW_DIR = REPO_ROOT / "data" / "processed" / "bench_raw"
CLINC_SOURCE_TAG = "bjb:clinc150/intent"  # the exclusion this script's docstring explains


def _sample_wide_subset(rng: random.Random, all_labels: list[str], true_name: str, min_n: int, max_n: int) -> tuple[list[str], int]:
    n_options = rng.randint(min_n, min(max_n, len(all_labels)))
    distractor_pool = [name for name in all_labels if name != true_name]
    distractors = rng.sample(distractor_pool, n_options - 1)
    options = distractors + [true_name]
    rng.shuffle(options)
    return options, options.index(true_name)


def build_nli_records(split_dir: Path, n: int, rng: random.Random) -> list[dict]:
    ds = load_from_disk(str(split_dir))
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    ds = ds.select(idx)
    out = []
    for row in ds:
        out.append({
            "state": row["state"], "question_key": row["question_key"], "question_type": row["question_type"],
            "instructions": row["instructions"], "options": row["options"], "label": row["label"],
            "label_idx": row["label_idx"], "source": row["source"],
        })
    return out


def build_dbpedia_records(split: str, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset("fancyzhx/dbpedia_14", split=split)
    all_labels = ds.features["label"].names
    assert len(all_labels) <= MAX_OPTIONS
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    ds = ds.select(idx)
    out = []
    for row in ds:
        true_name = all_labels[row["label"]]
        options = list(all_labels)
        rng.shuffle(options)
        label_idx = options.index(true_name)
        title = (row["title"] or "").strip()
        content = (row["content"] or "").strip()[:600]
        out.append({
            "state": f"Title: {title}\nContent: {content}", "question_key": "category", "question_type": "choice",
            "instructions": DBPEDIA_INSTRUCTIONS, "options": options, "label": true_name,
            "label_idx": label_idx, "source": f"dbpedia_14_fullwidth_{split}",
        })
    return out


def build_clinc_records(n: int, rng: random.Random) -> list[dict]:
    """Unchanged from build_wideschema_slice.py/build_primitives_slice.py -- CLINC150
    stays sourced exclusively here, held out of training entirely. See module docstring."""
    pool = []
    for split in ("train", "validation", "test"):
        pool.append(load_dataset("clinc/clinc_oos", "small", split=split))
    all_labels = pool[0].features["intent"].names
    assert len(all_labels) == 151
    combined_len = sum(len(p) for p in pool)
    idx = rng.sample(range(combined_len), min(n, combined_len))
    out = []
    for flat_idx in idx:
        for p in pool:
            if flat_idx < len(p):
                row = p[flat_idx]
                break
            flat_idx -= len(p)
        true_name = all_labels[row["intent"]]
        options, label_idx = _sample_wide_subset(rng, all_labels, true_name, CLINC_MIN_OPTIONS, CLINC_MAX_OPTIONS)
        out.append({
            "state": (row["text"] or "").strip(), "question_key": "intent", "question_type": "choice",
            "instructions": CLINC_INSTRUCTIONS, "options": options, "label": true_name,
            "label_idx": label_idx, "source": "clinc150_oos_small_wide",
        })
    return out


def load_bench_records(slice_name: str) -> list[dict]:
    """Reads the already-exported `bjb export --format hf` output, drops the
    CLINC150 rows (see module docstring), and returns plain dicts in the same
    shape every other loader here produces."""
    ds = load_from_disk(str(BENCH_RAW_DIR / slice_name))
    out = []
    for row in ds:
        if row["source"] == CLINC_SOURCE_TAG:
            continue
        out.append({
            "state": row["state"], "question_key": row["question_key"], "question_type": row["question_type"],
            "instructions": row["instructions"], "options": row["options"], "label": row["label"],
            "label_idx": row["label_idx"], "source": row["source"],
        })
    return out


def _validate(records: list[dict], name: str) -> None:
    for r in records:
        n = len(r["options"])
        assert 2 <= n <= MAX_OPTIONS, f"{name}: option count {n} out of [2, {MAX_OPTIONS}]"
        assert 0 <= r["label_idx"] < n, f"{name}: label_idx {r['label_idx']} out of range for {n} options"
        assert r["options"][r["label_idx"]] == r["label"], f"{name}: label_idx doesn't point at label text"
        assert len(set(r["options"])) == n, f"{name}: duplicate option text in {r['options']}"
        assert r["question_type"] in ("choice", "noul", "score"), f"{name}: unexpected question_type {r['question_type']!r}"


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    nli_dir = REPO_ROOT / "data" / "processed" / "nli_slice"
    print("loading NLI train sample (continuity anchor)...")
    train_nli = build_nli_records(nli_dir / "train", N_TRAIN_NLI, rng)
    print("loading DBpedia-14 train sample (continuity anchor)...")
    train_dbpedia = build_dbpedia_records("train", N_TRAIN_DBPEDIA, rng)
    print("loading better-jev-bench public export (11 tasks, CLINC150 excluded)...")
    train_bench = load_bench_records("public")
    train_records = train_nli + train_dbpedia + train_bench
    rng.shuffle(train_records)

    print("loading eval_id (in-distribution held-out) samples...")
    eval_id_nli = build_nli_records(nli_dir / "eval", N_EVAL_ID_NLI, rng)
    eval_id_dbpedia = build_dbpedia_records("test", N_EVAL_ID_DBPEDIA, rng)
    eval_id_bench = load_bench_records("heldout")
    eval_id_records = eval_id_nli + eval_id_dbpedia + eval_id_bench
    rng.shuffle(eval_id_records)

    print("loading CLINC150 (out-of-distribution, zero-shot-schema, unchanged regression check)...")
    eval_ood_records = build_clinc_records(N_EVAL_OOD_CLINC, rng)
    rng.shuffle(eval_ood_records)

    for name, records in [("train", train_records), ("eval_id", eval_id_records), ("eval_ood", eval_ood_records)]:
        _validate(records, name)

    Dataset.from_list(train_records).save_to_disk(str(out_dir / "train"))
    Dataset.from_list(eval_id_records).save_to_disk(str(out_dir / "eval_id"))
    Dataset.from_list(eval_ood_records).save_to_disk(str(out_dir / "eval_ood"))

    def by_type_and_source(records):
        return {
            "by_question_type": dict(collections.Counter(r["question_type"] for r in records)),
            "n_sources": len(set(r["source"] for r in records)),
        }

    stats = {
        "train": {"n": len(train_records), **by_type_and_source(train_records)},
        "eval_id": {"n": len(eval_id_records), **by_type_and_source(eval_id_records)},
        "eval_ood": {
            "n": len(eval_ood_records), **by_type_and_source(eval_ood_records),
            "note": "unchanged CLINC150 wide zero-shot-schema choice eval, reused across 13a.6/13a.8/this run as a regression check.",
        },
        "bench_source": "bjb export --slice public/heldout --tiers A --max-options 26 --max-per-task 5000/400 --seed 42, "
                         "run 2026-09-23 against abhijeet-labgpu:~/ekvachan/bench-repo (422,878-item corpus)",
        "clinc_excluded_from_bench_pool": True,
        "max_options_cap": MAX_OPTIONS,
        "seed": SEED,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    stats = build(REPO_ROOT / "data" / "processed" / "benchcorpus_slice")
    print(json.dumps(stats, indent=2))
