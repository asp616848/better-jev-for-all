# ScreenSpot-v2 harness

PRD.md Sections 5.2b / 13a.11 / 8.2. See `benchmarks/README.md` for the
cross-benchmark design; this file covers what's specific to ScreenSpot-v2 --
the vision analogue of what JevBench and jabr-v2 are for text (PRD.md 5.2b's
own framing).

**Status: built and validated with the mock backend; not yet run against a
real checkpoint.** `checkpoints/ekvachan-decoder-qwen-vision` (PRD.md 13a.11)
was still training when this was built -- the real run is one command, listed
below, for the coordinating session once it lands.

## What was verified (2026-09-24)

- **The data is real, and it comes from better-jev-bench, not from
  HuggingFace directly.** ScreenSpot-v2 is built, license-verified, and
  hash-committed in the sibling repo (`better_jev_bench/datasets/
  screenspot_v2.py`, `datasets/screenspot_v2/manifest.toml`). Apache-2.0,
  verified independently here too (HF Hub API `cardData['license']`), not
  just trusted from the sibling repo's own claim.
- **`--slice public` really does emit 0 items -- checked by running it, not
  assumed from reading the code.** ScreenSpot-v2 is `eval_only = true` in
  better-jev-bench's manifest; `better_jev_bench/export.py`'s slice-gating
  logic skips any `eval_only` dataset outright when `--slice public` is
  requested. Ran `bjb export --slice public --datasets screenspot_v2 --out
  <dir>` for real: `<dir>/public.jsonl` has 0 lines, with a printed
  `[export] skip screenspot_v2: dataset.eval_only = true` line.
- **`--slice heldout` gives 858 real items, not the manifest's headline 898.**
  The upstream manifest's `item_count = 898` is `40 public + 858 heldout`.
  `--slice heldout` only ever reads `bench/heldout/<name>.jsonl.gz` -- it does
  not also pull in the 40 public-slice rows, which are structurally
  unreachable through the sanctioned export path for an `eval_only` dataset.
  Ran `bjb export --slice heldout --datasets screenspot_v2 --out <dir>` for
  real: `<dir>/heldout.jsonl` has exactly 858 lines. **This benchmark reports
  858, and says why, rather than rounding up to 898.**
- **The real record shape** (from actually running the export, not from the
  dataset's own README): `{state, question_key, question_type, instructions,
  options, label, label_idx, source, images}` -- better-jev-bench's own
  ekVachan-training-compatible export schema (PRD.md 13a.11's
  `training/adapt_bench_vision_export.py` consumes the exact same shape).
  Each record's own `options` is a real **binary** pair: the true target
  element's descriptor and one real same-screenshot distractor (its cyclic
  neighbour in reading order) -- see `better_jev_bench/datasets/
  screenspot_v2.py`'s own docstring for why it stopped at binary (its export
  schema requires one fixed `option_count` per task; the real per-image
  candidate count varies).

## The grounding-as-choice reframing: real N-way, not the upstream binary

PRD.md 5.2b's original framing: "an item is 'here is the screenshot and an
instruction; which of these N elements is the target,' with distractors
drawn from the same screenshot's other annotated elements." This project's
`DecoderMultischemaBackend`/`DecoderVisionMultischemaBackend` answer any
2-to-26-option `choice` question, so there's no need to settle for
better-jev-bench's own binary export (which exists because *its* export
schema, unlike this project's decoder, does require one fixed width per
task). `benchmarks/screenspot_v2/loader.py`'s `reframe_records_to_items()`
recovers the fuller N-way question: group every exported row by its shared
image, and take the union of every row's own (label, distractor) pair as
that image's real candidate pool.

**Every option string in every reconstructed item is real, verbatim text
that already appears in the real, hash-verified export.** Nothing is
invented; this loader only regroups what's already there.

**Real per-image pool-size distribution**, computed 2026-09-24 over the real
858-record export (not assumed, not estimated -- see `vendored/manifest.json`
for the same numbers committed alongside the data):

| Pool size (N) | Items |
|---|---|
| 2 | 412 |
| 3 | 310 |
| 4 | 92 |
| 5 | 39 |
| 6 | 5 |

**The "too few annotated elements" case, handled explicitly.** An image whose
reconstructed pool has fewer than 2 distinct candidates can't support even a
binary choice. `reframe_records_to_items()` excludes such an image's rows and
*counts* the exclusion in its returned stats rather than dropping them
unremarked or padding the pool with an invented option. **On the real
858-item export this affects 0 items** -- every real image's pool has >= 2
members (better-jev-bench's own `MIN_CANDIDATES_PER_IMAGE = 2` guarantees
this at build time; this loader re-verifies it independently). The synthetic
self-test fixture (`fixtures/selftest.jsonl`) deliberately includes one image
whose pool collapses to a single string, specifically so this exclusion path
is exercised by *something*, since real data never triggers it.

One honest caveat, stated rather than silently assumed: the reconstructed
pool is a *lower bound* on an image's true annotated-element count (see
`loader.py`'s module docstring for exactly which upstream filtering step
could, in principle, drop a candidate from every surviving row's view). This
does not change any item's correctness -- `expected` is always a row's own
real `label`, always present in its own image's pool by construction -- it
only means a currently-zero number of images might offer fewer real options
than they truly have.

## Vendoring: images vs. metadata (the tradeoff, stated rather than picked silently)

`benchmarks/jevbench/` and `benchmarks/jabr_v2/` both vendor their entire
dataset (text) into `vendored/`. ScreenSpot-v2 is different: its 356 distinct
images are ~370MB on disk, sitting in better-jev-bench's own content-
addressed cache (`data/images/screenspot_v2/<sha256[:2]>/<sha256>.<ext>`),
already downloaded, license-verified, and hash-committed there.

**Decision: vendor the small metadata (`vendored/heldout.jsonl`, ~730KB
text), reference the images from better-jev-bench's cache at load time.**

- **For vendoring the images too**: this repo would be fully self-contained
  -- no sibling clone required, no risk of the cache moving or being pruned.
- **Against**: it duplicates ~370MB of bytes better-jev-bench already stores,
  license-verifies, and hash-commits; it would go stale the moment either
  repo's image set changes; and it breaks the precedent PRD.md 13a.11 already
  set for exactly this situation --
  `training/adapt_bench_vision_export.py` references better-jev-bench's
  images directly rather than copying them, because "both repos share a
  server, so `path` is already a real, directly-readable local file, no copy
  needed." Vendoring images here would be a second, inconsistent answer to
  the same question this repo already answered once.
- **The real, stated cost of the decision made**: this benchmark's real
  (non-`--selftest`) path cannot run at all without `better-jev-bench` cloned
  and built (`bjb build --datasets screenspot_v2`) as a sibling of this repo
  -- the exact same requirement the training pipeline already has, not a new
  one. `--bench-repo-dir` overrides the default sibling-path assumption for
  any other layout. `--selftest` never needs this at all -- it ships its own
  two tiny synthetic PNGs.
- **Every image path is re-resolved and re-hashed at load time**, never
  trusted from a stored string -- see `loader.resolve_bench_repo_image()`.
  This is strictly more paranoid than vendoring a path once and trusting it
  forever, and it's what actually catches a stale or corrupted cache instead
  of silently scoring against the wrong screenshot.

## How this was tested

There's no vision-trained checkpoint on disk yet (`checkpoints/
ekvachan-decoder-qwen-vision` was still training when this was built --
PRD.md 13a.11) and this session was explicitly told not to load anything onto
the GPU, so `--backend decoder-vision-multischema` against real weights could
not be exercised here. What *was* actually run, not just written:

1. **The real dataset, real reframing, no backend needed to count it** --
   `python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema-mock`
   loads the real 858-item export, reframes it (real grouping/pool logic,
   real image-path resolution against better-jev-bench's cache, real sha256
   re-verification of every one of the 356 distinct images), and filters it
   through `filter_supported_multischema` -- 858/858 supported (every item is
   a 2-6-option `choice` whose gold label is one of its own options), 0
   excluded for too-small a pool, matching the histogram above exactly.
2. **`--backend decoder-vision-multischema-mock --selftest`** -- the
   synthetic 4-record fixture (`fixtures/selftest.jsonl`, real bjb-export
   record shape, *not* real ScreenSpot-v2 data, two tiny checked-in PNGs, not
   real screenshots) run through the real loader function
   (`reframe_records_to_items`, the exact same function the real path calls),
   the real schema filter, a deterministic non-trained keyword-overlap
   backend that explicitly refuses to fabricate an answer from pixels it
   cannot see, and the real `eval/metrics.py` scoring path. Result: 3 of the
   4 synthetic rows survive reframing as a real N=3 choice item (the fourth
   is the deliberately-degenerate single-candidate image, correctly excluded
   and counted -- proving the exclusion path fires, not just exists) -- see
   the committed manifest for the exact accuracy/Brier/ECE this run produced;
   its number means nothing about grounding ability (the mock never looks at
   the image), only that filter -> call -> score -> evidence bundle runs end
   to end.
3. **Image hash-verification failure path**, checked by hand: pointing the
   resolver at a mismatched sha256 raises `ImageResolutionError` rather than
   silently serving the wrong bytes.

None of this is a ScreenSpot-v2 score. It's a real, working harness with a
real, verified item count and a real, verified reframing -- ready to run for
real the moment there's a checkpoint trained on vision data to answer it.

## Running

```sh
# Real dataset, against the vision checkpoint once it exists (PRD.md 13a.11):
python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema \
    --checkpoint-dir checkpoints/ekvachan-decoder-qwen-vision

# Vision harness wiring self-test -- synthetic fixture, no checkpoint, no
# torch/transformers/peft, no better-jev-bench clone needed:
python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema-mock --selftest

# Real dataset, filter-only proof (no model needed to know the item count):
python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema-mock
```

`--bench-repo-dir <path>` overrides the default assumption that
`better-jev-bench` is cloned as a sibling of this repo (see "Vendoring" above).

## What this harness does *not* do (yet)

- It has not been run against a real vision-trained checkpoint --
  `checkpoints/ekvachan-decoder-qwen-vision` (PRD.md 13a.11) was still
  training when this was built, and this session was explicitly scoped to
  not touch the GPU. See PRD.md and STATUS.md for the exact command to run
  once it lands (same as "Running" above, `--backend
  decoder-vision-multischema`).
- It does not implement a `noul`/`score` vision eval -- PRD.md 5.2b already
  named the honest state of those strata for vision data ("real, but
  narrower than it looks" / "under-populated, and we should say so rather
  than fake it"); ScreenSpot-v2 itself is `choice`-only in its real schema,
  so there's nothing to build here for the other two primitives from this
  dataset specifically.
- It does not attempt the 40 upstream "public"-slice rows in any form -- they
  are real ScreenSpot-v2 items, just structurally unreachable through
  better-jev-bench's sanctioned export path for an `eval_only` dataset (see
  above). Pulling them in some other way would defeat the point of marking
  the dataset eval-only in the first place.
