# benchmarks/

Standing benchmark harnesses for PRD.md Section 8: **JevBench** (`jevbench/`)
and **jabr-v2** (`jabr_v2/`). Both are real, runnable, tested harnesses. Both
currently report the same honest headline number: **0 real items answerable
by the current checkpoint.** That is not a placeholder or a bug — read on.

## The one fact that shapes everything in here

`ekvachan-base` (the only trained checkpoint that exists — PRD.md Section
13a.1) has a classifier head fixed to exactly one 3-way schema:
`options == ["entailment", "neutral", "contradiction"]` over a
`Premise: ...\nHypothesis: ...` state. `serve/inference.py`'s
`ChoiceUnsupportedError` and PRD.md Section 10a spell this out; nothing here
tries to work around it.

Both JevBench and jabr-v2 are real, task-diverse decision benchmarks —
support routing, refund policy, severity ratings, intent classification,
content moderation, and dozens more — and **every single real item in both
of them uses its own domain-specific label set.** Verified by actually
downloading and parsing every public record in both datasets (`benchmarks/
jevbench/vendored/`, `benchmarks/jabr_v2/vendored/`): zero items in either dataset
use `["entailment", "neutral", "contradiction"]` as their option set. So the
honest number today is **0 of 231 public JevBench items** and **0 of 944
vendored jabr-v2 items** are things this checkpoint could legitimately
attempt — not "0 correct," but 0 *attemptable*, which is a different and more
important fact (PRD.md Section 8.1b: scoring "declined to guess" the same as
"answered wrong" is exactly backwards for a project whose pitch is
calibration).

**This is not a benchmark failure being hidden — it's the harness working
correctly.** Run `python -m benchmarks.jevbench.run` yourself: it filters
every real item through `benchmarks/common/schema_filter.py`, finds none
match, and says so in the evidence bundle rather than forcing an answer onto
the fixed head or silently dropping the unsupported items. See each
subdirectory's README for the exact numbers and how they were produced.

## Layout

```
benchmarks/
  common/            shared by every harness here
    items.py           canonical Item shape + content-hashing
    schema.py          FIXED_CHECKPOINT_LABELS -- the one schema the
                        checkpoint can answer, kept dependency-free (see
                        its docstring for why it's a duplicated literal,
                        not an import)
    schema_filter.py    decides which items are answerable -- the single
                        place that logic is allowed to live
    backends.py         InProcessBackend / HTTPBackend / MockBackend, one
                        interface, three ways to actually answer a question
    evidence.py         writes the PRD.md 8.2 evidence bundle to results/
    harness.py          the shared run loop + CLI argument parser
  jevbench/
    vendored/            vendored JevBench public dataset (MIT) -- see NOTICE.md
    fixtures/            synthetic self-test fixture, NOT real data
    loader.py, run.py
    README.md            what was verified, the real item counts, the finding
  jabr_v2/
    vendored/            vendored jabr-v2 dataset (CC0) -- see NOTICE.md
    fixtures/
    loader.py, run.py
    README.md            what was verified, the real item counts, the finding
```

## Running a harness

```sh
# Real dataset, in-process (needs torch/transformers + a real checkpoint):
python -m benchmarks.jevbench.run
python -m benchmarks.jabr_v2.run

# Real dataset, against a running serve/server.py instance:
python -m benchmarks.jevbench.run --backend http --http-endpoint http://localhost:8000

# Harness wiring self-test -- proves the pipeline works without a checkpoint,
# torch, or transformers installed at all (this is how it was tested while
# building it: see each README's "How this was tested" section):
python -m benchmarks.jevbench.run --backend mock --selftest
python -m benchmarks.jabr_v2.run --backend mock --selftest
```

Every run writes two files to `results/`:
`<run_id>.manifest.json` (small, meant to be committed — provenance, the
schema-filter breakdown, aggregate scores) and `<run_id>.raw` (one JSON line
per attempted item, matching the existing `.gitignore` pattern
`results/*.raw` so it never gets checked in — see PRD.md Section 8.2 and
`benchmarks/common/evidence.py`'s docstring).

## What happens the day a schema-general model exists

PRD.md Section 5.1a's cross-attention decision head (or the decoder-LoRA arm)
would make `options` arbitrary instead of fixed. Nothing in `benchmarks/`
would need to change structurally: swap `serve/inference.py`'s fixed-3-way
contract for one that reports its actual supported schema per request, update
`benchmarks/common/schema.py`'s `FIXED_CHECKPOINT_LABELS` doc comment (or
replace the whole-list-equality check in `schema_filter.py` with whatever the
new model's real constraint is, e.g. "options must be single tokens"), and
every real item in both vendored datasets becomes attemptable in the same run
that currently reports 0. That's the point of building this now, against a
checkpoint that can't clear the bar yet: it's a drop-in run, not a rebuild.
