"""
Builds a small, real, mixed-schema training/eval corpus to test the open
question PRD.md Section 14 Q4 raised on 2026-09-23: the decoder's
restricted-logit read is architecturally NOT fixed-width (unlike the
encoder's classification head), but it has only ever been *trained* on one
fixed 3-class schema (entailment/neutral/contradiction). This script builds
the data needed to test whether that architectural slack is real.

Three real sources, all loaded live via `datasets.load_dataset` (verified
working on this server before being wired in -- see below), normalized into
the exact `state`/`question_key`/`question_type`/`instructions`/`options`/
`label`/`label_idx`/`source` record shape `training/data.py` already uses,
so the output is a drop-in dataset for the width-aware training script
(`training/train_decoder_lora_multischema.py`):

1. **The existing NLI slice** (`data/processed/nli_slice`, built by
   `training/data.py` -- ANLI+WANLI+MultiNLI+SNLI). A modest sample, not
   the full 1.2M. Fixed 3-way schema, already the training arm's baseline.

2. **DBpedia-14** (`fancyzhx/dbpedia_14`). Verified license: CC-BY-SA 3.0 +
   GFDL (better-jev-bench_PRD.md Section 4.1's "Tier A-share-alike" --
   freely usable, but carries attribution/copyleft obligations that must
   propagate to anything derived from this corpus; this is a research
   experiment, not a redistribution, but that obligation is noted here for
   whoever redistributes the output later). 14-way single-label ontology
   classification, clean schema, verified to load directly via
   `load_dataset("fancyzhx/dbpedia_14", ...)` on 2026-09-23 before being
   used here (not assumed from memory).

3. **CLINC150** (`clinc/clinc_oos`, "small" config). Verified license:
   CC-BY-3.0 (Tier A, same PRD section). 151-way intent classification
   (150 real intents + a genuine "oos" / none-of-the-above class) --
   **held out entirely**: zero examples from this source ever appear in
   `train/`. It exists only in `eval_ood/`, so it is the actual test of
   "does the mechanism generalize to an option set it never trained on,"
   not "did it memorize this option set." Verified loadable on 2026-09-23.

**A dataset that was tried and deliberately dropped**: `PolyAI/banking77`
(the task's other suggested Tier-A candidate) failed to load under the
`datasets>=3.0` pinned in this repo's `pyproject.toml` --
`RuntimeError: Dataset scripts are no longer supported, but found
banking77.py`. A working mirror exists (`legacy-datasets/banking77`,
verified loadable), but this script doesn't use it: DBpedia-14 (in-training
schema) + CLINC150 (held-out zero-shot schema) already satisfy the task's
requirement of "2 real datasets, one held out entirely for the
generalization test," and BANKING77's 77-way schema would need the exact
same random-N-way-subset treatment as DBpedia-14 below for no added
experimental signal. Not used, but the load failure is real and reported
here rather than silently working around it.

**The width cap, and why**: every dataset above natively exceeds this
experiment's option-count cap of 10 (NLI is the only one that doesn't).
Rather than extend `LETTERS` past the alphabet (a real design decision
the task asked not to stumble into) or pick a single fixed-width slice
that would just be "one more fixed schema," each DBpedia-14/CLINC150
record here presents a *random* N-way subset (4-8 options, always
including the true label) drawn from that dataset's full label space. This
is deliberate: it means even within one dataset, the model sees a
different option *count* and a different option *set* almost every
example -- a harder and more honest generalization test than one static
N-way slice would be, and it keeps every record within the token-verified
single-token-letter budget (A-J were confirmed as 10 distinct single
tokens in Qwen3.5-4B's tokenizer on this server before this was written).

Run: `uv run python3 -m training.build_multischema_slice`
"""

import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_OPTIONS = 10  # keep in sync with training/train_decoder_lora_multischema.py's MAX_OPTIONS
MIN_OPTIONS_SAMPLED = 4  # floor for the random-subset schemas (DBpedia-14, CLINC150)
MAX_OPTIONS_SAMPLED = 8  # ceiling for the random-subset schemas -- leaves headroom under MAX_OPTIONS

SEED = 42

# Target sizes -- modest by design (this is a hypothesis test, not a corpus
# build). ~30k total, proportioned across three roles: in-training (NLI +
# DBpedia-14), in-distribution held-out (same two schemas, disjoint
# examples), and out-of-distribution held-out (CLINC150, disjoint dataset).
N_TRAIN_NLI = 12_000
N_TRAIN_DBPEDIA = 12_000
N_EVAL_ID_NLI = 1_500
N_EVAL_ID_DBPEDIA = 1_500
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


def _sample_option_subset(rng: random.Random, all_labels: list[str], true_name: str) -> tuple[list[str], int]:
    """Random 4-8-way subset of `all_labels` that always contains `true_name`.

    Returns (options, label_idx). The subset -- and its order -- is fixed at
    build time; the *presentation* order (which letter shows which option)
    is re-shuffled per-example at training/eval read time by
    `shuffled_letter_order` in the training script, exactly like the
    original 3-way script already does for LABELS. Building a canonical,
    unshuffled option list here and letting the training script own the
    position-debiasing shuffle keeps the same separation of concerns
    `training/data.py` already uses (it stores `options` in a fixed order
    too).
    """
    n_options = rng.randint(MIN_OPTIONS_SAMPLED, min(MAX_OPTIONS_SAMPLED, len(all_labels)))
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
        out.append(
            {
                "state": row["state"],
                "question_key": row["question_key"],
                "question_type": row["question_type"],
                "instructions": row["instructions"],
                "options": row["options"],
                "label": row["label"],
                "label_idx": row["label_idx"],
                "source": row["source"],
                "schema": "nli",
            }
        )
    return out


def build_dbpedia_records(split: str, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset("fancyzhx/dbpedia_14", split=split)
    all_labels = ds.features["label"].names
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    ds = ds.select(idx)
    out = []
    for row in ds:
        true_name = all_labels[row["label"]]
        options, label_idx = _sample_option_subset(rng, all_labels, true_name)
        title = (row["title"] or "").strip()
        content = (row["content"] or "").strip()[:600]  # cap content length, this is a smoke/hypothesis test not a corpus
        out.append(
            {
                "state": f"Title: {title}\nContent: {content}",
                "question_key": "category",
                "question_type": "choice",
                "instructions": DBPEDIA_INSTRUCTIONS,
                "options": options,
                "label": true_name,
                "label_idx": label_idx,
                "source": f"dbpedia_14_{split}",
                "schema": "dbpedia_14",
            }
        )
    return out


def build_clinc_records(n: int, rng: random.Random) -> list[dict]:
    pool = []
    for split in ("train", "validation", "test"):
        pool.append(load_dataset("clinc/clinc_oos", "small", split=split))
    all_labels = pool[0].features["intent"].names
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
        options, label_idx = _sample_option_subset(rng, all_labels, true_name)
        out.append(
            {
                "state": (row["text"] or "").strip(),
                "question_key": "intent",
                "question_type": "choice",
                "instructions": CLINC_INSTRUCTIONS,
                "options": options,
                "label": true_name,
                "label_idx": label_idx,
                "source": "clinc150_oos_small",
                "schema": "clinc150",
            }
        )
    return out


def _validate(records: list[dict], name: str) -> None:
    for r in records:
        n = len(r["options"])
        assert 2 <= n <= MAX_OPTIONS, f"{name}: option count {n} out of [2, {MAX_OPTIONS}]"
        assert 0 <= r["label_idx"] < n, f"{name}: label_idx {r['label_idx']} out of range for {n} options"
        assert r["options"][r["label_idx"]] == r["label"], f"{name}: label_idx doesn't point at label text"
        assert len(set(r["options"])) == n, f"{name}: duplicate option text in {r['options']}"


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    stats = {}

    nli_dir = REPO_ROOT / "data" / "processed" / "nli_slice"
    print("loading NLI train sample...")
    train_nli = build_nli_records(nli_dir / "train", N_TRAIN_NLI, rng)
    print("loading DBpedia-14 train sample...")
    train_dbpedia = build_dbpedia_records("train", N_TRAIN_DBPEDIA, rng)
    train_records = train_nli + train_dbpedia
    rng.shuffle(train_records)

    print("loading NLI eval (in-distribution held-out) sample...")
    eval_id_nli = build_nli_records(nli_dir / "eval", N_EVAL_ID_NLI, rng)
    print("loading DBpedia-14 test split (in-distribution held-out) sample...")
    eval_id_dbpedia = build_dbpedia_records("test", N_EVAL_ID_DBPEDIA, rng)  # DBpedia's own test split -> disjoint from train sample by construction
    eval_id_records = eval_id_nli + eval_id_dbpedia
    rng.shuffle(eval_id_records)

    print("loading CLINC150 (out-of-distribution, zero-shot-schema held-out) sample...")
    eval_ood_records = build_clinc_records(N_EVAL_OOD_CLINC, rng)
    rng.shuffle(eval_ood_records)

    for name, records in [("train", train_records), ("eval_id", eval_id_records), ("eval_ood", eval_ood_records)]:
        _validate(records, name)

    for r in train_records + eval_id_records + eval_ood_records:
        r.pop("schema", None)  # internal bookkeeping only, not part of training/data.py's record shape -- dropped before writing

    Dataset.from_list(train_records).save_to_disk(str(out_dir / "train"))
    Dataset.from_list(eval_id_records).save_to_disk(str(out_dir / "eval_id"))
    Dataset.from_list(eval_ood_records).save_to_disk(str(out_dir / "eval_ood"))

    import collections

    def option_count_hist(records):
        return dict(collections.Counter(len(r["options"]) for r in records))

    stats = {
        "train": {
            "n": len(train_records),
            "by_source": dict(collections.Counter(r["source"] for r in train_records)),
            "option_count_hist": option_count_hist(train_records),
        },
        "eval_id": {
            "n": len(eval_id_records),
            "by_source": dict(collections.Counter(r["source"] for r in eval_id_records)),
            "option_count_hist": option_count_hist(eval_id_records),
        },
        "eval_ood": {
            "n": len(eval_ood_records),
            "by_source": dict(collections.Counter(r["source"] for r in eval_ood_records)),
            "option_count_hist": option_count_hist(eval_ood_records),
            "note": "zero examples from this source appear in train/ -- this is the generalization test, not held-out-examples-from-a-seen-schema",
        },
        "max_options_cap": MAX_OPTIONS,
        "seed": SEED,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    stats = build(REPO_ROOT / "data" / "processed" / "multischema_slice")
    print(json.dumps(stats, indent=2))
