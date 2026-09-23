# jabr-v2 harness

PRD.md Section 8.1 ("reproduce jabr-v2's exact eval harness/seeds if publicly
available; if not, use the same task categories and disclose the
difference"). See `benchmarks/README.md` for the cross-benchmark design;
this file covers what's specific to jabr-v2.

## Correction to the task brief this was built under

The brief stated PRD.md treats jabr-v2 as having "no confirmed public repo."
**That's checkable, and it's public: [`jabr/classifier-benchmark`]
(https://github.com/jabr/classifier-benchmark).** Von's own README (`wfzyx/
von` — the model PRD.md Section 3.1/G3 benchmarks against) links straight to
this repo's `results/v1v2-summary.md` and calls it "the jabr v2 benchmark."
That file documents exactly the "49-task, 869-case" figure PRD.md Section
3.1's table cites for Von's jabr-v2 macro-accuracy. See `NOTICE.md` for the
full chain of evidence and a real discrepancy found along the way (the
vendored `v2.toml` has 866 cases, not 869 — 3 short, because that suite is
still upstream "under review"/unlocked). PRD.md itself is out of scope for
this task to edit; this correction is recorded here for whoever updates it
next.

## What was verified (2026-09-22)

- **The repo is real, public, and reachable.** `gh api repos/jabr/
  classifier-benchmark` — created 2026-09-19, pushed 2026-09-21,
  `archived: false`, `visibility: "public"`.
- **License is CC0 1.0 Universal (public domain)**, confirmed by reading the
  actual `LICENSE` file (GitHub's API detector mislabels it — see
  `NOTICE.md`).
- **The real data/harness format**: cases live in plain **TOML** files
  (`cases/v1.toml`, `cases/v2.toml`) — an array of `[[task]]` tables, each
  with an `id`, a `type` (`choice`/`noul`/`score`), a `[task.question]`
  block (`instructions`, and `criteria` — a dict of `option: description`
  for `choice`/most `noul` tasks, or a list of level descriptions for
  `score`), and an array of `[[task.cases]]` (`state`, `expected`). The
  harness (`bench/` upstream) is a Python CLI (`python -m bench.run
  --backend von,gliner2,laya --suite v1 ...`) with pluggable local-model
  backends (`bench/backends/`) — you point it at weights on disk, not a
  hosted leaderboard you submit results to.
- **v1 is locked (78 cases, 8 tasks), v2 is unlocked/"under review" (49
  tasks, 866 cases as actually vendored — see the discrepancy note in
  `NOTICE.md`).** `v1` and `v2` combined = 944 items, as vendored here —
  not the 947 the repo's published summary states, for the same reason.

## The finding: 0 of 944 vendored items are answerable today

Checked programmatically across every `task.type` and, for `choice` tasks,
every `task.question.criteria` key set in both files: task label sets are
things like `["billing","tech","sales","account","other"]` (support
routing), `["book_restaurant","call_taxi","get_stock_price",
"search_flights","set_reminder"]` (intent), ordinal level lists for `score`
tasks, and true/false for `noul` tasks (49 + 8 = 57 distinct tasks total).
**Not one task in either suite uses `["entailment","neutral",
"contradiction"]` as its option set** — unsurprising, since every jabr-v2
task is an operational/business decision (support, expenses, on-call
routing, content moderation, compliance, triage, ...), not an NLI-style
premise/hypothesis judgment.

```
$ python -m benchmarks.jabr_v2.run
{
  "n_items_considered": 944,
  "n_supported_by_fixed_schema": 0,
  "n_unsupported": 944,
  ...
  "accuracy": null, "brier": null, "ece": null
}
```

Ran with no checkpoint present and no torch/transformers installed, for the
same reason as the JevBench harness — see `benchmarks/README.md`.

## Update 2026-09-23: 387 of 944 are answerable under the decoder/multischema filter

Same update as `benchmarks/jevbench/README.md`'s (see that file for the full
PRD.md 14 Q4 / 13a.5 context). Filtering these 944 vendored items by the
decoder/multischema rule (2-26 options, `filter_supported_multischema`)
instead of the fixed-schema rule above:

```
$ python -m benchmarks.jabr_v2.run --backend decoder-multischema-mock
{
  "n_items_considered": 944,
  "n_supported_by_multischema_cap": 387,
  "n_unsupported": 557,
  ...
}
```

**387 of 944 — all 387 `choice` items — are now answerable**, up from 0. The
557 unsupported are the 337 `noul` + 220 `score` items (no trained model for
either). None were rejected for exceeding the 26-option cap — the widest
option set across both suites has 6 options. Real, committed manifest:
`results/jabr_v2-decoder_multischema_mock-*` (`--backend
decoder-multischema-mock`, a non-trained stub, real vendored dataset, no
`--selftest`). Once a wide-schema checkpoint exists, `--backend
decoder-multischema --checkpoint-dir <path>` runs these same 387 items for a
real score.

## How this was tested

Same pattern as JevBench (see `benchmarks/jevbench/README.md`'s "How this
was tested" for the full explanation):

1. **Real dataset, real run, default `in_process` backend, no `--selftest`**
   — shown above; produced a real evidence bundle with the honest 0/944
   result, no torch required.
2. **`--backend mock --selftest`** — a 4-item synthetic fixture
   (`fixtures/selftest.toml`, real jabr-v2 TOML task/case shape, explicitly
   marked as synthetic in its own comment header) run through the real TOML
   loader (`loader.load_toml_file`), the real schema filter, a deterministic
   non-trained heuristic, and `eval/metrics.py`. 4/4 passed the schema
   filter, 2/4 scored correct by the mock's heuristic — full pipeline
   exercised end to end.
3. **`HTTPBackend`** — same stdlib-mocked-server test as JevBench's harness;
   this backend's code is shared (`benchmarks/common/backends.py`) between
   both benchmarks, so it was verified once, not twice.
4. **`--backend decoder-multischema-mock`, real dataset and `--selftest`** —
   same pattern as JevBench's update above; this backend's code is also
   shared between both benchmarks. Real 944-item dataset: 387/944 passed
   `filter_supported_multischema`, no crashes
   (`results/jabr_v2-decoder_multischema_mock-*`). Fixture: 4/4 passed
   (`results/jabr_v2-selftest-decoder_multischema_mock-*`).

## Running

```sh
python -m benchmarks.jabr_v2.run                                    # real data, in-process
python -m benchmarks.jabr_v2.run --backend http --http-endpoint http://localhost:8000
python -m benchmarks.jabr_v2.run --backend mock --selftest          # wiring self-test

python -m benchmarks.jabr_v2.run --backend decoder-multischema \    # real data, decoder/multischema
    --checkpoint-dir checkpoints/ekvachan-decoder-qwen-wideschema    # (PRD.md 13a.5, once it lands)
python -m benchmarks.jabr_v2.run --backend decoder-multischema-mock # multischema wiring self-test
```

## What this harness does *not* do (yet)

- It does not reproduce jabr-v2's own reported Von/Jev/GLiNER2/Laya numbers
  or attempt to match its exact micro/macro aggregation — `eval/metrics.py`
  (accuracy/Brier/ECE) is used instead, consistent with every other harness
  in this repo, and documented here as a deliberate choice, not an oversight.
- It does not attempt `noul` or `score` items (337 `noul` + 220 `score` =
  557 of the 944 vendored items, by measured type count; the remaining 387
  are `choice`, and it's those 387 whose option sets were checked against
  the fixed schema and found to match zero of them) — no trained head
  exists for `noul`/`score` at all (PRD.md 10a).
- It has no opinion on which of `v1.toml` / `v2.toml` is "the" jabr-v2 PRD.md
  8.1 meant — `load_public()` runs both by default (`--suites v1` or
  `--suites v2` would need a small loader.py change to expose as a CLI flag,
  not yet added since it changes nothing about the 0-supported result
  either way).
