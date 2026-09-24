# Attribution

`benchmarks/vizdoom/` reproduces the ViZDoom `Defend the Center` and
`Health Gathering` benchmark protocol published by
[`wfzyx/von`](https://github.com/wfzyx/von) (`benchmarks/doom_eval.py`,
274 lines, and `benchmarks/run_doom_benchmark.py`, 116 lines, fetched from
`master` via the GitHub API on 2026-09-24), so that this project's own
`ekvachan-decoder-*` checkpoints can be compared to Von's own published
9.00 kills / 12.11s survival numbers on a like-for-like basis (PRD.md 8.1 /
8.1d).

- **License: Apache-2.0.** Verified via `gh api repos/wfzyx/von/license` ->
  `spdx_id: "Apache-2.0"`, backed by the repo's own `LICENSE.md` (fetched and
  vendored below), not just the API's summary field.
- **What is reused, and how.** Nothing from Von's repo is copy-pasted or
  vendored as a file. This project's `env.py`, `observe.py` and `rubric.py`
  are independent reimplementations (see each file's own docstring for why
  reimplementing rather than copying was the deliberate choice: it forces
  every design decision -- button order, buffer indexing, bucket boundaries
  -- to be re-verified against the real, running ViZDoom engine rather than
  trusted from reading Von's source once). What genuinely *is* taken
  verbatim, because doing so is the entire point of a like-for-like
  reproduction (PRD.md 7.4), are the following **protocol constants** --
  data, not code:
  - The two scenarios' action sets and ViZDoom button mapping
    (`env.py::SCENARIO_ACTIONS`, `BUTTON_MAP`).
  - The published 8-seed lists per scenario
    (`env.py::DEFEND_SEEDS`, `HEALTH_SEEDS`) -- reproducing "the same 8
    shared seeds Von used" (PRD.md 8.1 item 2) requires using these exact
    integers, not arbitrary ones.
  - `tics=4` (frameskip) and the 300-step-per-episode cap.
  - The `MONSTERS`/`ITEMS`/`PRETTY_NAMES` actor-name classification sets
    used to decide what the text observation describes and how it names it.
  - The five position buckets, four size buckets, and four depth buckets
    (`observe.py::format_position/format_range/format_depth`) and the
    sentence templates built from them (`observe_text`) -- Von's own
    `format_doom_state()`.
  - The `--rubric von` instructions/criteria strings, verbatim
    (`rubric.py::_DEFEND_INSTRUCTIONS`, `_DEFEND_CRITERIA`,
    `_HEALTH_INSTRUCTIONS`, `_HEALTH_CRITERIA`) -- Von's own
    `get_doom_question()`.
- **Copyright**: the reproduced text/data above originates from the `wfzyx/von`
  project's own committed source.
- **Obligation: attribution.** This NOTICE file, plus the inline attribution
  in every file that reproduces one of the constants above, is that
  attribution.

**`vendored/LICENSE`** is a verbatim copy of `wfzyx/von`'s own `LICENSE.md`
(fetched via the GitHub API 2026-09-24, 69 lines, the standard Apache
License 2.0 text) -- included here because there is no vendored *dataset
file* the way `benchmarks/jevbench/`/`benchmarks/jabr_v2/`/
`benchmarks/screenspot_v2/` each have one; the "vendored" content in this
benchmark is the handful of literal constants listed above, embedded
directly in `env.py`/`observe.py`/`rubric.py` with inline attribution
comments, not a separate data file.

## ViZDoom itself

`vizdoom` (the PyPI package, a runtime dependency declared in this repo's
`pyproject.toml`) is **MIT-licensed for its own code** (confirmed on the
installed wheel's own metadata: `License :: OSI Approved :: MIT License`,
not read off a web page). It embeds ZDoom, which descends from id Software's
GPL-released Doom source, and ships **Freedoom** assets
(`freedoom1.wad`/`freedoom2.wad`, used by the stock `defend_the_center`/
`health_gathering` scenarios this benchmark loads from
`vizdoom.scenarios_path`) under the 3-clause BSD license.

**This project consumes the published PyPI wheel as an ordinary dependency
and neither forks, patches, vendors, nor redistributes the engine or its
WADs** -- so the only live obligation is notice/attribution, which this
section is. No original id Software assets are used or needed; the stock
scenarios run entirely on Freedoom. If a future change ever vendors or
patches the ViZDoom engine itself, the GPL-descended terms become relevant
and must be re-checked (flagged here, per PRD.md 7.4, so that would be a
decision and not an accident).
