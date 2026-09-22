# Attribution

`benchmarks/jabr_v2/vendored/v1.toml`, `v2.toml`, and `hashes.json` are vendored,
byte-for-byte, from
[`jabr/classifier-benchmark`](https://github.com/jabr/classifier-benchmark),
fetched 2026-09-22 via the GitHub Contents API at that repo's `main` branch
tip as of that date.

- License: **CC0 1.0 Universal — public domain.** `benchmarks/jabr_v2/
  vendored/LICENSE` is the repo's own LICENSE file, copied verbatim. Note:
  GitHub's API license-detector reports this repo as `license.key: "other"`
  / `spdx_id: "NOASSERTION"` — that is a detector miss, not a real ambiguity:
  the actual `LICENSE` file opens with a short prose sentence ("This
  repository ... is released under CC0 1.0 Universal...") before the
  standard CC0 legal boilerplate, which apparently breaks GitHub's exact-
  text matcher. The repo's own README says plainly: "License: CC0 1.0
  Universal (LICENSE) — public domain. The test cases may be reused
  freely." Read the vendored `LICENSE` file directly if you want to check
  this yourself rather than trust either detector.
- Copyright: none asserted — CC0 is a public-domain dedication.
- Provenance of the cases themselves, per the repo's own `cases/README.md`:
  **synthetic**, generated and cross-reviewed by a committee of LLMs (GLM
  5.3 Flash, GLM 5.3, Kimi K3, Qwen3.8 2.4T, Qwen3.8 Flash, DeepSeek V4.1
  Flash, MiMo V2.5 Pro), screened for single-defensible-gold-label cases,
  debatable items dropped rather than relabeled. Not real user data.

## Correcting this task's starting assumption

The task brief this harness was built under stated jabr-v2 "has no confirmed
public repo" in PRD.md. That turned out to be checkable and wrong: **Von's
own README** (`wfzyx/von`, the model PRD.md Section 3.1/G3 is benchmarked
against) links directly to
[`jabr/classifier-benchmark/blob/main/results/v1v2-summary.md`](https://github.com/jabr/classifier-benchmark/blob/main/results/v1v2-summary.md)
and describes it in the same sentence as "The 49-task, 869-case jabr v2
benchmark". That file, vendored here as `vendored/v2.toml` (the actual cases) and
readable at the URL above (the results), is the real jabr-v2. PRD.md Section
8.1 should be updated to reflect this the next time someone works on it —
out of scope for this task (PRD.md is explicitly off-limits here), but
recorded here so it isn't lost.

## A real discrepancy worth flagging, not smoothing over

The upstream repo's own README and `results/v1v2-summary.md` describe v2 as
**49 tasks / 869 cases**. The `v2.toml` file actually vendored here (fetched
2026-09-22) parses to **49 tasks / 866 cases** — 3 fewer. Per-task diff
against the published results table: `commit_intent` has 15 cases here vs.
16 published, `grammar_issue` 21 vs. 22, `fair_housing_violation` 18 vs. 19.
This is consistent with upstream's own stated status for `v2.toml`:
**"under review" / unlocked** (`cases/hashes.json` only has a locked hash
for `v1`, not `v2` — see `cases/README.md`'s "Locking" section upstream).
An unlocked suite is, by the repo's own documented rules, editable at any
time with no hash pin to detect drift against. This vendored copy's own
sha256 (below) is the only frozen reference this harness has; it is *not*
guaranteed to match either an earlier or a later fetch of the same URL, and
is not guaranteed to match whatever Von's published 72.0%-macro number was
actually computed against.

Vendored file hashes (sha256, computed 2026-09-22, of the file bytes
exactly as vendored):

```
90c16ab8c8fc87641b39de1336bd14ed254298ab9d91f496e09a3607d3747360  v1.toml
97e380b175a18819b3baa49c550246862fce7e5d8f58dfce6c256db93a4e6426  v2.toml
```

These do **not** match `cases/hashes.json`'s recorded `v1` hash
(`22342af4e2c68f02964a684a7d5876211e6ac5c073931468e8e7a174c8cf072c`) — that
upstream hash is evidently computed over something other than the raw file
bytes (a canonicalized/parsed representation, per `bench/validate.py`
upstream, not independently re-derived here). Not claimed as bit-identical
to upstream's lock; only claimed as an honest record of exactly what this
repo vendored and when.
