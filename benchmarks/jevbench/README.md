# JevBench harness

PRD.md Sections 3.1a / 8.1a / 8.2. See `benchmarks/README.md` for the
cross-benchmark design; this file covers what's specific to JevBench.

## What was verified (2026-09-22)

- **The repo is real, public, and active.** `gh api repos/fstandhartinger/
  jevbench` — created 2026-09-19, 77 stargazers, `pushed_at` the same day
  this was written, `archived: false`, `visibility: "public"`.
- **License is MIT.** `license.spdx_id == "MIT"` from the GitHub API, and
  independently confirmed at the per-record level — every one of the 231
  vendored records carries its own `provenance.license: "MIT"` field. See
  `NOTICE.md` for the one stale-looking sentence in upstream's own
  `THIRD-PARTY.md` and why it isn't treated as overriding the per-record
  field.
- **The real data/harness format**, from actually downloading and parsing
  the files (not the README's prose description): JevBench ships as JSONL —
  one JSON object per line — with fields `id`, `family`, `group`, `labels`
  (the exact answer-option list), `expected` (gold label), `provenance`
  (license/source/label-basis), `question` (`type`: `choice`/`noul`/`score`,
  `instructions`, `criteria`), `split`, and `state`. The harness itself
  (`jevbench/` package upstream) is a Python CLI with pluggable *adapters*
  (`jevbench/adapters/*.py`) — HTTP adapters for hosted APIs
  (`typesafe`, `openai_compat`, ...) and in-process adapters for locally
  loaded open weights (`laya_local`, `local_openjev`, ...). It is not a
  static leaderboard you submit a CSV to; you run their CLI (or, as here,
  something that reads their frozen dataset the same way) against your own
  model and it produces per-item, per-axis results.
- **534 total items, 231 of them public.** `datasets/manifest.json`
  (vendored to `vendored/manifest.json`) lists 7 splits: `original` (72),
  `heldout` (24), `router` (78), `judge` (68), `easy` (48), `easy-heldout`
  (24), `hard` (111 public + 109 held out = 220). Only `original`, `easy`,
  and `hard` are published as files (`datasets/public/*.jsonl`) — the other
  four splits are deliberately unpublished, by upstream's own explicit
  anti-gaming design (PRD.md 8.1a: "held-out items committed before any run
  starts"). 72 + 48 + 111 = **231 public items**, which is exactly what
  `benchmarks/jevbench/vendored/` vendors and `loader.load_public()` loads.

## The finding: 0 of 231 public items are answerable today

Checked programmatically (every record's `question.type` and `labels`
field, across all 231 public records): **139 are `choice`, 74 are `noul`,
18 are `score` — and among the 139 `choice` items, the label sets used are
things like `["billing_question","cancel_order","change_address",
"report_damage","track_order"]`, `["cancel","change_address","other",
"refund","status"]`, `["high","low","medium","urgent"]`, and 60+ other
domain-specific sets. Not one of the 231 public items uses
`["entailment","neutral","contradiction"]`.**

Running the harness confirms this directly:

```
$ python -m benchmarks.jevbench.run
{
  "n_items_considered": 231,
  "n_supported_by_fixed_schema": 0,
  "n_unsupported": 231,
  ...
  "accuracy": null, "brier": null, "ece": null
}
```

This ran with **no checkpoint present and no torch/transformers installed**
in the environment it was built in — see `benchmarks/common/harness.py`'s
docstring for why filtering doesn't need a live model, and
`benchmarks/README.md` for what that means for reproducing this exact
result yourself.

## How this was tested

There's no checkpoint on disk in this environment (`checkpoints/` is
gitignored — PRD.md Section 12 — and lives on the training server this task
had no access to) and no `torch`/`transformers` installed, so the
`in_process`/`http` backends against real weights could not be exercised
here. What *was* actually run, not just written:

1. **The real dataset, real run, `--backend in_process` (the default), no
   `--selftest`** — shown above. Produced a real evidence bundle
   (`results/jevbench-in_process-<timestamp>.manifest.json`) with the honest
   0/231 result, without needing torch at all (confirms the filter-before-
   backend design in `benchmarks/common/harness.py` works as intended).
2. **`--backend mock --selftest`** — a synthetic 4-item fixture
   (`fixtures/selftest.jsonl`, real JevBench record shape, *not* real
   JevBench data, clearly marked as such in its own `provenance` field and
   in the run's evidence manifest) run through the real loader
   (`loader.load_file`), the real schema filter, a deterministic non-trained
   heuristic backend, and the real `eval/metrics.py` scoring path. Result:
   4/4 items passed the schema filter (by construction — the fixture uses
   the checkpoint's exact schema), 2/4 scored correct by the mock's dumb
   heuristic (accuracy 0.5, ECE 0.4625) — proving the whole pipeline
   (filter → call → score → evidence bundle) actually runs end to end.
3. **`HTTPBackend` against a real (stdlib, mocked-response) HTTP server** —
   confirmed the request shape (`{"state":..., "questions":{"decision":
   {"type":"choice","options":[...]}}}`) and response parsing
   (`results.decision.{choice,probabilities,confidence}`) both work, without
   needing FastAPI installed to prove it.

None of this is a JevBench score. It's a real, working harness that reports
a real, honest 0 — and is ready to run for real (`--backend in_process`,
real checkpoint, no `--selftest`) the moment there's a model that can answer
more than one fixed 3-way schema.

## Running

```sh
python -m benchmarks.jevbench.run                                    # real data, in-process
python -m benchmarks.jevbench.run --backend http --http-endpoint http://localhost:8000
python -m benchmarks.jevbench.run --backend mock --selftest          # wiring self-test
```

## What this harness does *not* do (yet)

- It does not implement JevBench's own four-axis score (Intelligence /
  Calibration / Speed / Cost, geometric mean, `jevbench/composite_v13.py`
  upstream) — that requires a Cost basis (a $/1k-decisions tariff or a
  labelled "no tariff" reason) and upstream's exact Speed-scoring curve,
  neither of which is meaningful to compute over 0 answered items. Accuracy/
  Brier/ECE come from `eval/metrics.py` (the same module `training/
  train_encoder.py` uses), not from JevBench's own scoring code — that's a
  deliberate scope cut, documented here rather than silently approximated.
- It does not run the 303 held-out items (`heldout`, `router`, `judge`,
  `easy-heldout`, 109 more `hard`) — they are not public anywhere and cannot
  be vendored. A real JevBench submission (per upstream's own process,
  `docs/v1.2-additions*.md`) would need those to be run against upstream's
  own infrastructure, not this repo's harness.
- It does not attempt `noul` or `score` items in any form (probability
  readout, majority-vote, anything) — the checkpoint has no trained head for
  either (PRD.md 10a), so they are correctly counted as unsupported rather
  than approximated.
