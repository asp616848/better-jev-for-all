# benchmarks/

Standing benchmark harnesses for PRD.md Section 8: **JevBench** (`jevbench/`)
and **jabr-v2** (`jabr_v2/`). Both are real, runnable, tested harnesses.
Against the only checkpoint that existed when this was first built
(`ekvachan-base`, the fixed-3-class encoder), both reported the same honest
headline number: **0 real items answerable.** That result is unchanged and
still real evidence (PRD.md 13a.1/13a.3) — see "The one fact that shapes
everything in here" below, and each subdirectory's README, for it.

**Update 2026-09-23 (PRD.md Section 14 Q4 / 13a.5):** the decoder (Qwen3.5-4B
LoRA, restricted-logit read) is now the primary architecture, and its
mechanism can answer ANY `choice` question with 2 to `max_options` (26)
options — not one hardcoded label set. Filtering the same two real, vendored
datasets by *that* rule instead (`benchmarks/common/schema_filter.py`'s new
`filter_supported_multischema()`, `benchmarks/common/schema.py`'s
`DECODER_MULTISCHEMA_MAX_OPTIONS`) gives a real, computed, non-zero number
today — **without needing the trained checkpoint at all**, since this is
dataset-only work exactly like the original 0-answerable finding was:

| | total | `choice` (answerable, 2-26 opts) | `noul`/`score` (no model) |
|---|---|---|---|
| JevBench | 231 | **139** | 92 |
| jabr-v2 | 944 | **387** | 557 |

Every real `choice` item in both datasets tops out at 6 options (verified by
actually reading the max option count across both vendored files), so the
26-option cap doesn't reject anything today — it's a real architectural
ceiling (PRD.md 13a.5), just not yet a binding one on these two benchmarks.
See `benchmarks/common/backends.py`'s `DecoderMultischemaBackend` (real,
loads a `train_decoder_lora_wideschema.py`-shaped checkpoint) and
`DecoderMultischemaMockBackend` (non-trained wiring self-test, no torch/
checkpoint needed — same discipline as `MockBackend` below) for how to
actually run these 139/387 items once a checkpoint exists:

```sh
python -m benchmarks.jevbench.run --backend decoder-multischema \
    --checkpoint-dir checkpoints/ekvachan-decoder-qwen-wideschema
python -m benchmarks.jabr_v2.run --backend decoder-multischema \
    --checkpoint-dir checkpoints/ekvachan-decoder-qwen-wideschema
```

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

# Decoder/multischema path (PRD.md 14 Q4 / 13a.5) -- real dataset, against a
# train_decoder_lora_wideschema.py checkpoint once one exists:
python -m benchmarks.jevbench.run --backend decoder-multischema --checkpoint-dir <path>
python -m benchmarks.jabr_v2.run --backend decoder-multischema --checkpoint-dir <path>

# Decoder/multischema wiring self-test -- real vendored items, filtered by
# the 2-26-option rule, answered by a non-trained stub -- no checkpoint,
# torch, transformers, or peft needed:
python -m benchmarks.jevbench.run --backend decoder-multischema-mock
python -m benchmarks.jabr_v2.run --backend decoder-multischema-mock
```

Every run writes two files to `results/`:
`<run_id>.manifest.json` (small, meant to be committed — provenance, the
schema-filter breakdown, aggregate scores) and `<run_id>.raw` (one JSON line
per attempted item, matching the existing `.gitignore` pattern
`results/*.raw` so it never gets checked in — see PRD.md Section 8.2 and
`benchmarks/common/evidence.py`'s docstring).

## What happens the day a schema-general model exists

This section originally predicted, ahead of time, exactly what 2026-09-23's
update above did: PRD.md Section 5.1a's cross-attention decision head or the
decoder-LoRA arm would make `options` arbitrary instead of fixed, nothing in
`benchmarks/` would need to change *structurally*, and every real item in
both vendored datasets would become attemptable in the same run that
currently reports 0 for the fixed-schema path. That's now implemented
(`filter_supported_multischema`, `DecoderMultischemaBackend`, `--backend
decoder-multischema`) rather than hypothetical — it's a drop-in run, not a
rebuild, exactly as predicted, and the 139/231 and 387/944 numbers above are
real today even before a wide-schema checkpoint exists to actually answer
them (that's the whole point of `benchmarks/common/schema_filter.py` being
dataset-only logic with no model dependency).

The one thing still missing is the checkpoint itself: PRD.md 13a.5's
true-wide-schema training run is in progress as of this writing. The moment
it lands (see its manifest for the exact `checkpoints/` path — don't guess
the name), `python -m benchmarks.jevbench.run --backend decoder-multischema
--checkpoint-dir <that path>` (and the jabr-v2 equivalent) is the command
that turns these 139/387 *answerable* counts into real accuracy/Brier/ECE
numbers, the same way `--backend in_process` already does for the fixed-
schema path.
