# benchmarks/

Standing benchmark harnesses for PRD.md Section 8: **JevBench** (`jevbench/`),
**jabr-v2** (`jabr_v2/`), **ScreenSpot-v2** (`screenspot_v2/`, PRD.md
5.2b/13a.11 — the vision analogue of the other two, added 2026-09-24), and
**ViZDoom** (`vizdoom/`, PRD.md 8.1d/13a.13 — added 2026-09-24, the only
*episodic* one). Against the only checkpoint that existed when the first two
were built (`ekvachan-base`, the fixed-3-class encoder), both reported the
same honest headline number: **0 real items answerable.** That result is
unchanged and still real evidence (PRD.md 13a.1/13a.3) — see "The one fact
that shapes everything in here" below, and each subdirectory's README, for it.

**Update 2026-09-24 (PRD.md 8.1d/13a.13): ViZDoom added.** The fourth
standing benchmark, and structurally different from the other three: there is
no frozen dataset, no gold label, and the item sequence is generated live by
a closed-loop `Defend the Center`/`Health Gathering` ViZDoom episode reacting
to the model's own actions — Von's own headline claim (9.00 kills vs. Jev's
5.62) is a ViZDoom claim, run on Von's own published 8-seed-per-scenario
protocol. This required a genuinely new shared module,
`benchmarks/common/episodic.py::run_episodic_harness()`, rather than a branch
through `run_harness()` (see its own docstring). Validated end to end against
the real, headless ViZDoom engine with rubric-oracle/random/mock policies —
no real checkpoint run yet, one command away once the GPU is free (see
`benchmarks/vizdoom/README.md`).

**Update 2026-09-24 (PRD.md 5.2b/13a.11): ScreenSpot-v2 added.** A third
standing benchmark, vendored (as metadata only — see its own README's
"Vendoring: images vs. metadata") via the sibling `better-jev-bench` repo's
`bjb export --slice heldout`. 858 real items (not the upstream manifest's
headline 898 — see `benchmarks/screenspot_v2/README.md` for why), reframed
from ScreenSpot-v2's own binary grounding format into genuine N-way
(2-to-6-option) `choice` items, real distractors only, 0 items excluded. Runs
today with `--backend decoder-vision-multischema-mock` (real data, real image
hash-verification, no torch/checkpoint needed) and against
`checkpoints/ekvachan-decoder-qwen-vision` (PRD.md 13a.11, training as of
this writing) with `--backend decoder-vision-multischema` the moment it
lands.

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
    backends.py         InProcessBackend / HTTPBackend / MockBackend (fixed-
                        schema); DecoderMultischemaBackend[-Mock] (text
                        multischema); DecoderVisionMultischemaBackend[-Mock]
                        (vision multischema, PRD.md 5.2b/13a.11) -- one
                        interface, every way to actually answer a question
    evidence.py         writes the PRD.md 8.2 evidence bundle to results/
    harness.py          the shared static-dataset run loop + CLI argument parser
    episodic.py         the shared CLOSED-LOOP run loop (PRD.md 8.1d) -- for
                        benchmarks with no frozen item list and no gold label
                        (ViZDoom today); reuses build_backend/make_arg_parser/
                        write_bundle/eval.metrics unchanged
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
  screenspot_v2/
    vendored/            vendored metadata only (Apache-2.0) -- see NOTICE.md;
                        images referenced from better-jev-bench's cache, not
                        vendored (see this dir's own README)
    fixtures/            synthetic self-test fixture + two tiny real PNGs
    loader.py, run.py
    README.md            what was verified, the real 858-item count, the
                        grounding-as-choice N-way reframing
  vizdoom/
    vendored/LICENSE     Von's own Apache-2.0 LICENSE text -- see NOTICE.md
                        (no vendored dataset file here; the "vendored" content
                        is a handful of literal protocol constants, attributed
                        inline in env.py/observe.py/rubric.py)
    env.py               DoomEnvironment + DoomSnapshot -- Von's exact config,
                        against the real ViZDoom engine
    observe.py            observe_text() -- byte-faithful to Von's
                        format_doom_state(); observe_frame() is a documented
                        stub (the vision arm, PRD.md 8.1d step 8)
    rubric.py             --rubric von/none framing + rubric_action() (the
                        deterministic oracle/DAgger teacher)
    run.py                CLI: --scenario, --rubric, --policy (model/oracle),
                        --backend, --seeds, --episodes-per-seed, --dagger-out
    README.md             what was verified, the real oracle/random baseline
                        numbers, what the vision arm still needs
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

# Vision multischema path (PRD.md 5.2b/13a.11) -- real ScreenSpot-v2 data
# (858 real N-way grounding items), against checkpoints/ekvachan-decoder-qwen-vision
# once it exists:
python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema \
    --checkpoint-dir checkpoints/ekvachan-decoder-qwen-vision

# Vision multischema wiring self-test -- real 858-item dataset, real image
# hash-verification against better-jev-bench's cache, answered by a
# non-trained stub that never looks at the image -- no checkpoint, torch,
# transformers, or peft needed:
python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema-mock
python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema-mock --selftest

# ViZDoom (PRD.md 8.1d/13a.13) -- episodic, no frozen dataset. Real text-arm
# run against the milestone checkpoint, both scenarios, both rubric
# conditions, all 16 published seeds:
python -m benchmarks.vizdoom.run --backend decoder-multischema \
    --checkpoint-dir checkpoints/ekvachan-decoder-qwen-benchcorpus \
    --scenario both --rubric both

# ViZDoom baselines that need no checkpoint at all -- the rubric-oracle
# (Von's own rule run as a policy) and a uniform-random policy:
python -m benchmarks.vizdoom.run --policy oracle --scenario both --rubric von
python -m benchmarks.vizdoom.run --backend random --scenario both --rubric von \
    --seeds 0,1,2,3,4,5,6,7 --episodes-per-seed 10

# ViZDoom wiring self-test -- real ViZDoom episodes, no torch/transformers/
# peft/checkpoint:
python -m benchmarks.vizdoom.run --backend decoder-multischema-mock --scenario both
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
