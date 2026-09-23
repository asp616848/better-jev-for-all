"""
First data builder to introduce the `noul` and `score` primitives into
training, alongside `choice` -- following the same fork discipline every
prior data builder in this project has used (`build_multischema_slice.py` ->
`build_wideschema_slice.py`, now -> this). PRD.md 13a.6/13a.7 and STATUS.md's
review log (2026-09-23) both record why this is next: the 92/231 and 557/944
JevBench/jabr-v2 items this project still can't attempt are not a
schema-width problem (widest real `choice` option set in either dataset is
6, comfortably inside the 26-letter budget) -- they are 100% `noul`/`score`
items. No amount of further `choice` work touches them.

**Why this is expected to work with (almost) no new mechanism**: the
restricted-logit read this project already trained and proved generalizes by
option count and by domain (13a.6: 96.10% zero-shot on a wide, unseen
schema) doesn't know or care what a "choice" is semantically -- it reads
whichever single-uppercase-letter token comes out on top, masked to the
row's own option count. A `noul` proposition is representable as a 2-option
lettered question ("A) Yes / B) No"); a `score` on an N-level ordered scale
is representable as an N-option lettered question over the level
descriptions, in scale order. Training data for both was the actual gap, not
model architecture -- this script closes it.

**One real, deliberate mechanism change this data requires, not just new
data** -- documented here because it's genuinely load-bearing, not a detail:
`score` options are ordinal (PRD 1.2: "can land between levels --
probability-weighted"), so unlike `choice`/`noul` they must NOT go through
the per-example `shuffled_letter_order` position-randomization the training
script applies to every other row. A `score` row's options are written here
in fixed, canonical ascending-scale order (index 0 = lowest level) and the
paired training script (`train_decoder_lora_primitives.py`) must present
them in that same fixed order every time, so the model can learn that letter
position corresponds to scale position -- exactly the property that makes a
softmax's neighboring-letter probability mass meaningful as "landed between
levels," rather than noise from an arbitrary per-example shuffle. `choice`
and `noul` rows are unaffected: both still get the existing per-example
shuffle, since neither has an ordinal structure a fixed letter position
could exploit correctly.

**Data sources for the two new primitives, license-checked directly against
the HF Hub API on this server on 2026-09-23, not assumed:**

1. **`noul` <- BoolQ** (`google/boolq`, CC-BY-SA-3.0, Tier A-share-alike).
   {question, passage, answer(bool)} triples -- a genuine yes/no proposition
   grounded in a passage, which is exactly `noul`'s shape (PRD 1.2: a single
   proposition, float 0-1 answer). Train split (9,427 rows) used for
   training; validation split (3,270 rows, disjoint) used for eval_id --
   same disjoint-split pattern `build_dbpedia_records`'s train/test already
   established.

2. **`score` <- Sp1786/multiclass-sentiment-analysis-dataset** (Apache-2.0).
   3-level ordinal sentiment (negative=0 < neutral=1 < positive=2, confirmed
   directly by sampling 2,000 rows and checking every (label, sentiment)
   pair maps consistently -- not assumed from the dataset name). This is
   deliberately a narrow (3-level) first pass, not a wide one -- mirrors this
   project's own established pattern (13a.5's narrow multischema smoke test
   before 13a.6's wide follow-up): prove the primitive works at all before
   spending a second run widening it to a 5-10 level scale. A wide-scale
   `score` follow-up, once a suitable licensed 5+ level dataset is found, is
   the natural next step after this one -- not attempted here.

`choice` continuity: NLI (existing 3-way slice) and full-width DBpedia-14
carried over unchanged from `build_wideschema_slice.py` (same instructions,
same full-14-category presentation, same license notes) so this run is also
a regression check that mixing in new primitives doesn't quietly hurt
`choice` accuracy. CLINC150's wide (15-26-way) zero-shot-schema eval is
reused as-is for the same reason -- if 13a.6's 96.10% number holds up again
here, `choice` generalization survived the primitive mix; if it doesn't,
that's a real finding to report, not something to hide by dropping the
comparison.

**What this pass does NOT establish, stated plainly**: `noul`'s eventual
served output is a float `P(true)`, and `score`'s is a probability-weighted
scale position -- neither of those primitive-specific output formats is
built yet (that's `serve/` + `benchmarks/` integration work, tracked
separately in STATUS.md). This run reuses the existing `eval/metrics.py`
accuracy/Brier/ECE machinery (argmax against `label_idx`) as a first proxy,
exactly as 13a.5/13a.6 did for `choice` before any of that was wired into a
server. A `noul`/`score`-specific eval (float calibration for the former,
scale-distance for the latter) is real follow-up work, not done here.

Run: `uv run python3 -m training.build_primitives_slice`
"""

import collections
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_OPTIONS = 26  # unchanged ceiling, inherited from build_wideschema_slice.py

CLINC_MIN_OPTIONS = 15
CLINC_MAX_OPTIONS = 26

SEED = 42

# Four-way mix, same total order of magnitude (24k) as build_wideschema_slice.py's
# two-way mix, so "adding two new primitives" is the only real variable versus that
# run -- not also a scale change.
N_TRAIN_NLI = 6_000
N_TRAIN_DBPEDIA = 6_000
N_TRAIN_BOOLQ = 6_000
N_TRAIN_SENTIMENT = 6_000
N_EVAL_ID_NLI = 375
N_EVAL_ID_DBPEDIA = 375
N_EVAL_ID_BOOLQ = 375
N_EVAL_ID_SENTIMENT = 375
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
BOOLQ_INSTRUCTIONS = (
    "Given the passage above and a yes/no question about it, choose "
    "whether the correct answer is \"Yes\" or \"No\"."
)
SENTIMENT_INSTRUCTIONS = (
    "Given the text above, choose the option that best describes its "
    "overall sentiment, from most negative to most positive."
)
SENTIMENT_LEVELS = ["negative", "neutral", "positive"]  # fixed ascending scale order -- see module docstring


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


def build_boolq_records(split: str, n: int, rng: random.Random) -> list[dict]:
    """`noul` primitive: a yes/no proposition grounded in a passage. See module
    docstring for the CC-BY-SA-3.0 license verification and split choice."""
    ds = load_dataset("google/boolq", split=split)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    ds = ds.select(idx)
    out = []
    for row in ds:
        options = ["Yes", "No"]  # shuffled per-example at train/eval time by the training script, same as choice
        label = "Yes" if row["answer"] else "No"
        label_idx = options.index(label)
        passage = (row["passage"] or "").strip()
        question = (row["question"] or "").strip()
        out.append({
            "state": f"Passage: {passage}\n\nQuestion: {question}", "question_key": "answer", "question_type": "noul",
            "instructions": BOOLQ_INSTRUCTIONS, "options": options, "label": label,
            "label_idx": label_idx, "source": f"boolq_{split}",
        })
    return out


def build_sentiment_records(split: str, n: int, rng: random.Random) -> list[dict]:
    """`score` primitive: a 3-level ordinal scale, narrow first pass -- see module
    docstring for why width isn't widened in this same run, and for why options
    are NOT shuffled (fixed ascending order, load-bearing for ordinal semantics)."""
    ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset", split=split)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    ds = ds.select(idx)
    out = []
    for row in ds:
        label_idx = int(row["label"])
        assert 0 <= label_idx < len(SENTIMENT_LEVELS), f"unexpected sentiment label {row['label']}"
        assert SENTIMENT_LEVELS[label_idx] == row["sentiment"], "label/sentiment mapping drifted from what was verified"
        text = (row["text"] or "").strip()
        out.append({
            "state": text, "question_key": "sentiment", "question_type": "score",
            "instructions": SENTIMENT_INSTRUCTIONS, "options": list(SENTIMENT_LEVELS), "label": row["sentiment"],
            "label_idx": label_idx, "source": f"sp1786_sentiment_{split}",
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
        if r["question_type"] == "score":
            assert r["options"] == SENTIMENT_LEVELS, f"{name}: score row options are not in fixed canonical scale order"


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    nli_dir = REPO_ROOT / "data" / "processed" / "nli_slice"
    print("loading NLI train sample...")
    train_nli = build_nli_records(nli_dir / "train", N_TRAIN_NLI, rng)
    print("loading DBpedia-14 train sample (full-width)...")
    train_dbpedia = build_dbpedia_records("train", N_TRAIN_DBPEDIA, rng)
    print("loading BoolQ train sample (noul)...")
    train_boolq = build_boolq_records("train", N_TRAIN_BOOLQ, rng)
    print("loading Sp1786 sentiment train sample (score)...")
    train_sentiment = build_sentiment_records("train", N_TRAIN_SENTIMENT, rng)
    train_records = train_nli + train_dbpedia + train_boolq + train_sentiment
    rng.shuffle(train_records)

    print("loading eval_id (in-distribution held-out) samples...")
    eval_id_nli = build_nli_records(nli_dir / "eval", N_EVAL_ID_NLI, rng)
    eval_id_dbpedia = build_dbpedia_records("test", N_EVAL_ID_DBPEDIA, rng)
    eval_id_boolq = build_boolq_records("validation", N_EVAL_ID_BOOLQ, rng)  # disjoint from train split
    eval_id_sentiment = build_sentiment_records("validation", N_EVAL_ID_SENTIMENT, rng)  # disjoint from train split
    eval_id_records = eval_id_nli + eval_id_dbpedia + eval_id_boolq + eval_id_sentiment
    rng.shuffle(eval_id_records)

    print("loading CLINC150 (out-of-distribution, zero-shot-schema, WIDE 15-26 way, choice-regression-check) sample...")
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
            "by_source": dict(collections.Counter(r["source"] for r in records)),
        }

    stats = {
        "train": {"n": len(train_records), **by_type_and_source(train_records)},
        "eval_id": {"n": len(eval_id_records), **by_type_and_source(eval_id_records)},
        "eval_ood": {
            "n": len(eval_ood_records), **by_type_and_source(eval_ood_records),
            "note": "unchanged from build_wideschema_slice.py -- CLINC150 wide zero-shot-schema choice eval, "
                    "reused here as a regression check that mixing in noul/score doesn't hurt choice generalization.",
        },
        "max_options_cap": MAX_OPTIONS,
        "sentiment_levels_fixed_order": SENTIMENT_LEVELS,
        "seed": SEED,
        "note": "noul (BoolQ) and score (Sp1786 sentiment, 3-level) are new primitives in this data slice -- "
                "see module docstring for what this run does and does not establish.",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    stats = build(REPO_ROOT / "data" / "processed" / "primitives_slice")
    print(json.dumps(stats, indent=2))
