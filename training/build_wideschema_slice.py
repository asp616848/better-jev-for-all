"""
Fork of `build_multischema_slice.py`, built to answer the honest caveat
PRD.md Section 13a.5 (2026-09-23) states about its own result: the
multischema run's 98.5%-accuracy zero-shot-schema finding used random
4-8-way subsets (always containing the true label) for BOTH its DBpedia-14
training schema AND its CLINC150 held-out schema -- a substantially easier
task than genuine wide-schema classification, and DBpedia's own
in-distribution eval hit 99.3% for the same reason. This script builds the
harder, honest follow-up 13a.5 explicitly calls "the natural next
experiment": genuine full-width DBpedia-14 (no subsetting -- every example
sees all 14 real categories) and a genuinely wider CLINC150 held-out eval
(15-26 options, not 4-8), without pretending either one hides the mechanism's
real structural ceiling.

**Why a new fork, not editing `build_multischema_slice.py` in place**: the
same reasoning `train_decoder_lora_multischema.py`'s own docstring already
gives for why IT forked from `train_decoder_lora.py` applies again one level
up. `build_multischema_slice.py`'s output is the exact data backing PRD.md
Section 13a.5's published, reviewed numbers (98.50% accuracy / 0.0339 raw ECE
on 3,000 zero-shot-schema CLINC150 examples). Editing that script in place
to change what DBpedia-14/CLINC150 records look like would make that
already-published result silently unreproducible from its own generating
script. Forking again keeps 13a.5's script and data frozen and isolates the
new, harder task design so a bug here can't retroactively touch the trusted
13a.5 artifact. Same policy, applied consistently, one experiment later.

**The three real, load-bearing changes from `build_multischema_slice.py`**:

1. **`MAX_OPTIONS`: 10 -> 26.** Independently re-verified on this server on
   2026-09-23 (not just carried over from the multischema script's own
   verification of A-J): all 26 uppercase letters A-Z encode to exactly one,
   mutually distinct token under Qwen3.5-4B's tokenizer (ids 32-57,
   contiguous). Digits 0-9 are also single tokens but are NOT needed or used
   here -- 26 letters is already the full budget this mechanism can support
   without inventing a multi-token identifier scheme, which the task
   deliberately does not attempt (see point 3).

2. **DBpedia-14 is now genuinely full-width: every example presents all 14
   real categories, never a subset.** `build_dbpedia_records` below no longer
   samples distractors -- it takes the dataset's full 14-label space, shuffles
   presentation order (letter position gets re-randomized again downstream by
   `shuffled_letter_order` in the training script regardless, so this is
   belt-and-suspenders, not load-bearing), and that's the whole option list.
   This directly tests true 14-way classification (chance rate ~7.1%), not
   "pick the right one out of 4-8 where a coin flip already gets you closer."
   DBpedia-14 has exactly 14 real classes -- comfortably inside the 26-letter
   budget, so this is the one schema in this experiment tested at its true,
   full width with zero subsetting.

3. **CLINC150's held-out eval is now wide (15-26 options), not merely
   modest (4-8), but is STILL NOT the genuine 151-way task (150 intents +
   `oos`).** 151 exceeds the 26-letter budget this restricted-logit
   mechanism can address with single uppercase-letter identifiers -- that is
   a structural ceiling of THIS mechanism, not a bug or a corner cut here.
   Reaching true 151-way would need a different identifier scheme (multi-
   token option labels) or a different mechanism entirely (PRD.md Section
   5.1a's cross-attention decision head, which encodes option text instead
   of reading a single output token). Neither is attempted here -- inventing
   a non-single-token identifier hack to dodge the 26-letter cap was
   explicitly out of scope for this task, so this script states the limit
   plainly instead: **15-26-way CLINC150 subsets (chance rate ~3.8-6.7%
   depending on width) are meaningfully harder than the previous 4-8-way
   design, but "much wider" is not "full width," and anyone reading this
   later should not conflate the two.**

Everything else -- the three real data sources, their licenses, the
BANKING77 drop, NLI staying fixed 3-way, the record shape
(`state`/`question_key`/`question_type`/`instructions`/`options`/`label`/
`label_idx`/`source`), and the DBpedia-14/CLINC150 license notes -- is
carried over unchanged from `build_multischema_slice.py`, which already got
those right:

1. **NLI** (`data/processed/nli_slice`, ANLI+WANLI+MultiNLI+SNLI). Fixed
   3-way schema, unchanged from before -- kept in the mix at the same role
   (a third, already-validated schema, not the object of this experiment).

2. **DBpedia-14** (`fancyzhx/dbpedia_14`, CC-BY-SA 3.0 + GFDL, Tier
   A-share-alike per `better-jev-bench_PRD.md` Section 4.1 -- attribution/
   copyleft obligations noted here for whoever redistributes derived output
   later; this is a research experiment, not a redistribution).

3. **CLINC150** (`clinc/clinc_oos`, "small" config, CC-BY-3.0, Tier A).
   Held out entirely: zero examples from this source ever appear in
   `train/`. Re-verified on this server on 2026-09-23: `intent` has 151
   names including `oos` (at index 42); combined train+validation+test pool
   is 16,200 examples, comfortably enough for a 3,000-example wide-subset
   sample without replacement.

**Scale, and why it matches the previous run's order of magnitude**:
`build_multischema_slice.py` used 24,000 train examples (12k NLI + 12k
DBpedia-14) and this script keeps that same split and total, so the harder
task is the only real variable changing between the two runs -- a fair,
like-for-like comparison rather than a confound between "harder task" and
"different scale." Full-width DBpedia-14 needs no more examples than the
subset version did to be a genuine test: DBpedia-14's train split has
560,000 rows across 14 roughly-balanced classes (~40k/class), so a 12k
random sample still gives ~850 examples/class, plenty to learn a genuine
14-way discrimination. Eval sizes are unchanged too (1,500 NLI + 1,500
DBpedia-14 in-distribution, 3,000 CLINC150 zero-shot-schema) so 13a.5's
n=3,000-on-the-held-out-schema statistical power carries over exactly.

Run: `uv run python3 -m training.build_wideschema_slice`
"""

import collections
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_OPTIONS = 26  # A-Z, all independently re-verified single-token on this server 2026-09-23 -- see docstring point 1

# CLINC150's wide-subset range: as wide as the 26-letter budget allows,
# still short of its true 151-way schema (150 intents + oos) -- see
# docstring point 3 for why 151 is structurally out of reach for this
# mechanism, not a corner cut in this script.
CLINC_MIN_OPTIONS = 15
CLINC_MAX_OPTIONS = 26

SEED = 42

# Same order of magnitude as build_multischema_slice.py's 24k train / 3k+3k
# eval split -- see docstring's scale note for why this is a deliberate,
# like-for-like choice, not an arbitrary carry-over.
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


def _sample_wide_subset(
    rng: random.Random, all_labels: list[str], true_name: str, min_n: int, max_n: int
) -> tuple[list[str], int]:
    """Random [min_n, max_n]-way subset of `all_labels` that always contains `true_name`.

    Generalized version of `build_multischema_slice.py`'s
    `_sample_option_subset` with the width range as parameters instead of
    hardcoded module constants, so the same function serves CLINC150's wide
    (15-26) range here. Same contract otherwise: the subset -- and its
    build-time order -- is fixed here; per-example letter-position shuffling
    is owned by the training script's `shuffled_letter_order`, exactly as
    before.
    """
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
    """Genuinely full-width: every record's `options` is all 14 real DBpedia-14
    categories, never a subset. This is the actual hardening over
    `build_multischema_slice.py`'s random-4-8-way version -- see docstring
    point 2."""
    ds = load_dataset("fancyzhx/dbpedia_14", split=split)
    all_labels = ds.features["label"].names
    assert len(all_labels) <= MAX_OPTIONS, (
        f"DBpedia-14 has {len(all_labels)} labels, which exceeds MAX_OPTIONS={MAX_OPTIONS} "
        "-- full-width presentation is no longer safe under the letter budget."
    )
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    ds = ds.select(idx)
    out = []
    for row in ds:
        true_name = all_labels[row["label"]]
        options = list(all_labels)
        rng.shuffle(options)  # canonical build-time order only; letter position is re-shuffled again at train/eval read time
        label_idx = options.index(true_name)
        title = (row["title"] or "").strip()
        content = (row["content"] or "").strip()[:600]  # cap content length, this is a hypothesis test not a corpus
        out.append(
            {
                "state": f"Title: {title}\nContent: {content}",
                "question_key": "category",
                "question_type": "choice",
                "instructions": DBPEDIA_INSTRUCTIONS,
                "options": options,
                "label": true_name,
                "label_idx": label_idx,
                "source": f"dbpedia_14_fullwidth_{split}",
                "schema": "dbpedia_14_fullwidth",
            }
        )
    return out


def build_clinc_records(n: int, rng: random.Random) -> list[dict]:
    """Wide (15-26 way) subsets of CLINC150's full 151-label space, always
    containing the true label. Wider than `build_multischema_slice.py`'s
    4-8-way version, but still not the genuine 151-way task -- see
    docstring point 3 for why that's a structural limit of this mechanism,
    stated plainly rather than worked around."""
    pool = []
    for split in ("train", "validation", "test"):
        pool.append(load_dataset("clinc/clinc_oos", "small", split=split))
    all_labels = pool[0].features["intent"].names
    assert len(all_labels) == 151, f"expected CLINC150's 150 intents + oos = 151 labels, got {len(all_labels)}"
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
        out.append(
            {
                "state": (row["text"] or "").strip(),
                "question_key": "intent",
                "question_type": "choice",
                "instructions": CLINC_INSTRUCTIONS,
                "options": options,
                "label": true_name,
                "label_idx": label_idx,
                "source": "clinc150_oos_small_wide",
                "schema": "clinc150_wide",
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

    nli_dir = REPO_ROOT / "data" / "processed" / "nli_slice"
    print("loading NLI train sample...")
    train_nli = build_nli_records(nli_dir / "train", N_TRAIN_NLI, rng)
    print("loading DBpedia-14 train sample (full-width, all 14 categories)...")
    train_dbpedia = build_dbpedia_records("train", N_TRAIN_DBPEDIA, rng)
    train_records = train_nli + train_dbpedia
    rng.shuffle(train_records)

    print("loading NLI eval (in-distribution held-out) sample...")
    eval_id_nli = build_nli_records(nli_dir / "eval", N_EVAL_ID_NLI, rng)
    print("loading DBpedia-14 test split (in-distribution held-out, full-width) sample...")
    eval_id_dbpedia = build_dbpedia_records("test", N_EVAL_ID_DBPEDIA, rng)  # DBpedia's own test split -> disjoint from train sample by construction
    eval_id_records = eval_id_nli + eval_id_dbpedia
    rng.shuffle(eval_id_records)

    print("loading CLINC150 (out-of-distribution, zero-shot-schema, WIDE 15-26 way) sample...")
    eval_ood_records = build_clinc_records(N_EVAL_OOD_CLINC, rng)
    rng.shuffle(eval_ood_records)

    for name, records in [("train", train_records), ("eval_id", eval_id_records), ("eval_ood", eval_ood_records)]:
        _validate(records, name)

    for r in train_records + eval_id_records + eval_ood_records:
        r.pop("schema", None)  # internal bookkeeping only, not part of training/data.py's record shape -- dropped before writing

    Dataset.from_list(train_records).save_to_disk(str(out_dir / "train"))
    Dataset.from_list(eval_id_records).save_to_disk(str(out_dir / "eval_id"))
    Dataset.from_list(eval_ood_records).save_to_disk(str(out_dir / "eval_ood"))

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
            "note": "zero examples from this source appear in train/ -- this is the generalization test, not held-out-examples-from-a-seen-schema. "
                    "Options are wide (15-26 way) subsets of CLINC150's full 151-label space, NOT the full 151-way task -- see module docstring point 3.",
        },
        "max_options_cap": MAX_OPTIONS,
        "clinc_option_range": [CLINC_MIN_OPTIONS, CLINC_MAX_OPTIONS],
        "dbpedia_fullwidth": True,
        "seed": SEED,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    stats = build(REPO_ROOT / "data" / "processed" / "wideschema_slice")
    print(json.dumps(stats, indent=2))
