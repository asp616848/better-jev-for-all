# ekVachan — status

Living task list for `better-jev-for-all` (the model/serving/benchmark-harness repo). Update this file in the same commit as the work it describes — mark items done only after verifying them against real repo/server state, not after an agent or a prior summary claims they're done (see `skills/dev-guidelines/SKILL.md`). Full reasoning for every decision lives in `PRD.md`; this file is the checklist, PRD.md is the why.

Sibling project: **better-jev-bench** (separate repo, separate PRD, own `STATUS.md`) — a standalone dataset/benchmark corpus. Don't mix work across the two repos in one commit.

Last verified: 2026-09-23 (independent review pass — see "Review log" at the bottom).

## Done

- [x] Encoder trained & evaluated (`ekvachan-base-run2`): 86.32% acc, ECE 0.0344 calibrated — PRD 13a.1
- [x] Decoder trained (Qwen3.5-4B LoRA, `ekvachan-decoder-qwen-run1`): 92.25% acc, ECE 0.0232 raw, beat encoder on 20× less data — PRD 13a.2
- [x] Architecture decision: **decoder is primary**, decided 2026-09-23 — PRD 14 Q4
- [x] Multischema smoke+full test (10-option cap, `ekvachan-decoder-multischema-run1`): 98.5% zero-shot — flagged as an easy-subset construction, not fully decisive — PRD 13a.5
- [x] True wide-schema test (26-option cap, full 14-way DBpedia + wide 15–26-way CLINC150, no easy-subset artifact, `ekvachan-decoder-qwen-wideschema`): **96.10% zero-shot accuracy** — PRD 13a.6
- [x] Real benchmark harness integration (`benchmarks/common/backends.py` `DecoderMultischemaBackend`, `schema_filter.filter_supported_multischema`) — built, verified via mock/selftest, then run for real
- [x] Real accuracy on real third-party benchmarks, zero training exposure to either: **JevBench 83.45%** (139/231 items), **jabr-v2 88.89%** (387/944 items) — PRD 13a.7, `results/jevbench-decoder_multischema-20260923T084856Z.manifest.json`, `results/jabr_v2-decoder_multischema-20260923T085009Z.manifest.json`
- [x] License checks for reusable game/browser harnesses (`jev-ultrafast`, `tsai-sc`, `heist-one`) — all MIT, PRD 7.4
- [x] PRD.md fully reconciled between local/server and stale git refs after finding two "reported done but wasn't" gaps (2026-09-23) — see dev-guidelines skill for the postmortem
- [x] **SDK test suite verified passing** (2026-09-23 review): `cd sdk/python && uv run --extra test pytest -q` → **7 passed**. The earlier "fails to collect / missing uvicorn" entry was a **misdiagnosis** — `uvicorn`, `fastapi` and `pytest` are all already declared under `[project.optional-dependencies].test` in `sdk/python/pyproject.toml`. The failure was only ever `uv run pytest` invoked *without* `--extra test`. Nothing to fix in the package; the correct command is now recorded here.
- [x] **README.md rewritten** (2026-09-23 review) — now states the decoder-primary decision, the real 13a.2/13a.6/13a.7 numbers, the `is_complete_benchmark_score: false` caveat and what the unattempted remainder actually is, and the serving-layer gap. It previously claimed the decoder was still training and that no competitive benchmark had been run.
- [x] **`skills/ekvachan-setup/SKILL.md` corrected** (2026-09-23 review) — its dated status note was already current (the prior entry claiming it was stale was wrong); the genuinely stale line was its capability list, which still said benchmark suites and evidence bundles "don't exist". Fixed.

## In progress / needs fixing

- [ ] **`score`/`noul` training data built and a training run launched** (2026-09-23, GPU-bound, in progress): `training/build_primitives_slice.py` (BoolQ -> `noul`, CC-BY-SA-3.0; Sp1786 3-level sentiment -> `score`, Apache-2.0; both license-checked directly against the HF Hub API, not assumed) + `training/train_decoder_lora_primitives.py` (forked from `train_decoder_lora_wideschema.py`; the one real mechanism change is that `score` rows skip the per-example letter shuffle and use a fixed ascending scale order instead, since ordinal position needs to be consistent for the softmax's neighboring-letter mass to mean "landed between levels"). 24k train / 1.5k eval_id / 3k eval_ood (CLINC150 wide, reused as a choice-generalization regression check). Log: `~/ekvachan/primitives_run1.log`, checkpoint: `checkpoints/ekvachan-decoder-qwen-primitives`. **What this run does and does not establish**: whether the restricted-logit mechanism CAN be trained on noul/score at all, using the existing choice-style accuracy/ECE/Brier as a proxy metric — not `noul`'s real float output or `score`'s real scale-position output, which still need `serve/`/`benchmarks/` integration after this.
- [ ] **Verify the live decoder server end-to-end** (needs GPU, ~free VRAM check first) — real request against the actual loaded `DecoderChoiceModel`, real latency measurement (the 27–36ms in PRD 13a.3 is an *encoder* number, will not carry over to a 4B decoder). Deferred from the serve/ port above specifically because the GPU was occupied by the primitives training run when that was built.

## Needs fixing (no GPU required)

- [x] **`serve/` now defaults to the decoder** (2026-09-23) — `serve/inference.py` gained `DecoderChoiceModel` (a thin wrapper around `benchmarks.common.backends.DecoderMultischemaBackend`, reusing the same restricted-logit mechanism validated in PRD 13a.6/13a.7 rather than a third implementation) and `load_default_decoder()`; `serve/server.py`'s `get_model()` now calls it by default, and `/v1/systemone` passes `instructions` through and accepts any 2–26 option `choice` request instead of one fixed schema. `EncoderChoiceModel`/`load_default()` are kept unchanged (not deleted) since `benchmarks/common/backends.InProcessBackend` still targets them directly for the original fixed-schema comparison runs. Verified: real imports succeed with real torch installed, and the SDK's mocked test suite still passes 7/7 after updating `FakeModel`'s signature to accept the new `instructions` kwarg. **Not yet verified**: a real end-to-end request against the actual loaded decoder checkpoint and a real latency measurement — deferred because the GPU was mid-training (task below) when this was built, and loading a second 4B model alongside an active training run risked OOM-ing both. Do this next once the primitives run finishes.
- [x] **Training/eval manifests committed** (2026-09-23): `results/{ekvachan-base-run2,ekvachan-decoder-qwen-run1,ekvachan-decoder-multischema-run1,ekvachan-decoder-qwen-wideschema}.train-manifest.json` — copies of each checkpoint's real `manifest.json`. Weights stay gitignored (that's the large, disk-only artifact); the provenance JSON now survives a server failure.
- [x] **Hardware recorded** (2026-09-23): single NVIDIA L40S, 46068 MiB VRAM, driver 580.126.09, compute cap 8.9 — see PRD 14 Q1.
- [x] ~~Untracked `sdk/python/uv.lock`~~ — resolved 2026-09-23: added to `.gitignore` (it is a dev/test lockfile for a library package, not a shipped pin), so `git status` is clean again per dev-guidelines rule 11.

## Not started

- [ ] **`score` and `noul` primitives** — no model is trained for either. See the reprioritization note below: this, *not* the cross-attention head, is what the remaining benchmark items are actually blocked on.
- [ ] Cross-attention decision head (PRD 5.1a) — **reframed 2026-09-23, see review log.** Its original headline justification (the route to a general `choice` primitive) was **discharged by PRD 13a.6**: the decoder reached wide-schema generality through training data alone, 96.10% zero-shot on 15–26-way held-out schemas, no architecture change needed. It remains a legitimate *accuracy* experiment and the insurance policy for >26-option schemas — but no such item exists in either benchmark today (widest real option set in both is 6, PRD 13a.5), so it is no longer the critical path.
- [ ] Rust/ONNX serving port (PRD 7.1) — no code exists
- [ ] Quantization export (PRD 7.2)
- [ ] Multi-tier model family: nano tier, vision tier, large tier (PRD 5.2/5.2a) — only base (encoder) + one 4B decoder exist
- [ ] Conformal prediction calibration upgrade (PRD 5.3a)
- [ ] Fix the decoder's calibration procedure (PRD 13a.2 anomaly) — temperature scaling makes the decoder's ECE/Brier *worse*; the logit surrogate is the suspected cause, still unconfirmed. Every decoder number in the repo currently uses raw probabilities because of this.
- [ ] DAgger data collection method for the game/computer-use slice (PRD 5.3b)
- [ ] Game/computer-use harnesses: ViZDoom, StarCraft (`tsai-sc`), browser-use/jev-ultrafast (PRD 8.1) — zero code
- [ ] Cascade serving architecture, nano→base escalation (PRD 4.1 G5)

## Next up, with GPU time estimates

Anchor: this server's real Qwen3.5-4B LoRA runs took 65–75 min for 24k examples / 1 epoch. Single GPU, ~46GB VRAM (exact GPU model still unrecorded — see above).

Reordered 2026-09-23 by the review pass. The load-bearing change: **`score`/`noul` training moves up and the cross-attention head moves down**, because the schema-filter manifests show the unattempted benchmark items are *entirely* a primitive-coverage problem, not an option-width problem — JevBench 74 `noul` + 18 `score` = exactly its 92 unsupported items; jabr-v2 337 `noul` + 220 `score` = exactly its 557. Zero items in either dataset are unsupported for exceeding the 26-option cap. A cross-attention head over option text does nothing for a scalar `score` or a `noul`, so it cannot be the unlock for those 649 items; training the missing primitives is.

| # | Task | GPU time | Blocked on |
|---|---|---|---|
| 1 | ~~Commit training/eval manifests to `results/`; record the GPU model~~ | none | done 2026-09-23 |
| 2 | ~~Wire decoder support into `serve/`~~ done (code); **re-measure real latency against the loaded checkpoint** still open | minutes, to benchmark | GPU free (training run below must finish first) |
| 3 | Build `score` + `noul` training data, then train — **data built, training run launched** 2026-09-23 (`checkpoints/ekvachan-decoder-qwen-primitives`, log `~/ekvachan/primitives_run1.log`) | ~1–2 hrs, in progress | — |
| 4 | Fix the decoder calibration procedure (13a.2 anomaly) | ~minutes to re-fit | no retrain needed — it's a post-hoc fitting bug |
| 5 | Cross-attention head (5.1a), then train | ~1–2 hrs | dev work first; now an accuracy experiment + >26-option insurance, not the critical path |
| 6 | Nano tier training | ~30–60 min | data mix decision |
| 7 | Vision tier | ~1–2 hrs | data pipeline (non-GPU) first |
| 8 | Game harness wiring + eval | <30 min/harness | harness integration (non-GPU) first |
| 9 | Quantization export + eval | ~10–20 min | a serving target to quantize — i.e. #2 |
| 10 | Large tier (optional/stretch) | ~5–6 hrs | cascade proving it's needed |
| 11 | ONNX/Rust port validation | ~10–20 min | the port itself (non-GPU) |

## Review log

**2026-09-23 — independent review pass (PM/roadmap audit, no training run).**

Scope: read `skills/dev-guidelines/SKILL.md`, `PRD.md`, and this file in full; spot-checked every load-bearing claim against real repo state rather than prose.

Verified as accurate: all four headline training/eval numbers against their checkpoint manifests (86.32 / 92.25 / 96.10 / 98.5, including ECEs); both third-party benchmark numbers and item counts against the committed `results/` manifests; `DecoderMultischemaBackend` and `filter_supported_multischema` exist as described; `serve/` really is encoder-only; git sync clean in both directions against `origin/main` on both repos.

Corrected: the SDK item was misdiagnosed (tests pass, 7/7 — no missing dependency); the `ekvachan-setup` skill item was wrong (its status note was already current, a different line was stale); the cross-attention head's claim to be "the real path past the remaining 92/231 and 557/944" was **wrong** and is now reframed — those items are 100% `score`/`noul` type, per the schema filters' own unsupported-reason counts.

Found and recorded as new open items: training/eval manifests are gitignored and uncommitted (PRD 8.2 / rule 10 gap); hardware still unrecorded; untracked `sdk/python/uv.lock`.

Implemented in this pass (one item, deliberately): the **README.md rewrite** — chosen over the other open items because it is pure documentation (zero runtime risk), fully specified, and was the repo's most externally-visible false claim ("a second architecture arm is training now; no numbers yet", "None of the competitive benchmarks have been run yet") for a project whose entire differentiator is evidence discipline. The `serve/` decoder port was explicitly *not* attempted here: it is real engineering against the live wire contract, needs a GPU to validate, and is the wrong thing to half-do inside a review. It stays as task #2.

Also corrected in `PRD.md`: Section 13's Phase 1 roadmap still described the decoder arm as mid-run with no numbers and benchmarks/evidence bundles as "not started", directly contradicting 13a.2 and 13a.7 in the same document; and Section 5.1 now carries an explicit superseded-by-14-Q4 banner instead of silently reading as current direction (dev-guidelines rule 1).
