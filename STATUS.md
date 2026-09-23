# ekVachan — status

Living task list for `better-jev-for-all` (the model/serving/benchmark-harness repo). Update this file in the same commit as the work it describes — mark items done only after verifying them against real repo/server state, not after an agent or a prior summary claims they're done (see `skills/dev-guidelines/SKILL.md`). Full reasoning for every decision lives in `PRD.md`; this file is the checklist, PRD.md is the why.

Sibling project: **better-jev-bench** (separate repo, separate PRD, own `STATUS.md`) — a standalone dataset/benchmark corpus. Don't mix work across the two repos in one commit.

Last verified: 2026-09-24 (vision-tier planning pass — PRD 5.2b, `results/vision-path-probe-20260923T201758Z.manifest.json`). Prior: 2026-09-23 independent review pass — see "Review log" at the bottom.

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

- [x] **`score`/`noul` training run complete** (2026-09-23, 71m54s, `checkpoints/ekvachan-decoder-qwen-primitives`, `results/ekvachan-decoder-qwen-primitives.train-manifest.json`) — PRD 13a.8. Real numbers: `choice` 92.02% (regression check, barely moved), **`noul` 88.80%** (looks real on a first pass), **`score` 75.40%** (honest weak spot — worst Brier in the table by a wide margin, 0.355 vs 0.11-0.16 for everything else; **resolved 2026-09-23: confusion matrix shows 96.8% of errors are adjacent-level (neg<->neu, neu<->pos), only 3.2% skip to the opposite pole — genuine task difficulty at class boundaries, not a broken mechanism; more/wider score data is the right next lever, not a design change**). CLINC150 zero-shot choice regression check: 95.73% vs 13a.6's 96.10% baseline — noise-level difference, primitive mix did not hurt existing choice generalization. Still a proxy metric (argmax-over-letters), not `noul`'s real float output or `score`'s real scale-position output — that's still open, see below.
- [x] **Live decoder server verified end-to-end** (2026-09-23) — real `uvicorn serve.server:app` process, real HTTP requests: 2-way noul-shaped, 3-way NLI, 14-way DBpedia-style, all correct. `score` correctly 501s. **Measured warm latency: 87–109ms/call** (encoder was 27–36ms, PRD 13a.3 — expected gap for an unoptimized 4B decoder vs a much smaller encoder). `serve/inference.load_default_decoder()` now points at `checkpoints/ekvachan-decoder-qwen-primitives` (the newest checkpoint — same choice accuracy as wideschema within noise, per 13a.8's regression check) instead of the original wideschema checkpoint.

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
- [ ] Multi-tier model family: nano tier, large tier (PRD 5.2) — only base (encoder) + one 4B decoder exist. **Vision tier now has a full plan, see its own block below**

### Vision tier (`ekvachan-vision`) — planned in full 2026-09-24, **PRD 5.2b**

**Decision: one backbone, one adapter lineage, mixed text+image batches in a single run — not a separate model.** A second text-only adapter is a *contingency the measurement decides*, not a plan. Full reasoning and every measured number in PRD 5.2b; evidence bundle `results/vision-path-probe-20260923T201758Z.manifest.json`. Items below are ordered and each is specific enough to pick up without re-deriving the plan.

- [x] **Image-path capability probe built and run** (2026-09-24) — `training/probe_vision_path.py`, re-runnable as `uv run python3 -u -m training.probe_vision_path --gpu-forward`. Verified on this server: the chat template accepts `{"type":"image"}` content blocks and renders `<|vision_start|><|image_pad|><|vision_end|>`; **`AutoProcessor` (`Qwen3VLProcessor` + `Qwen2VLImageProcessor`), not `AutoTokenizer`, is what expands that placeholder** — 196 tokens for a 448×448 image, plus `pixel_values`/`image_grid_thw`; **`AutoModelForCausalLM` silently loads the text-only `Qwen3_5ForCausalLM` (4.841B, no vision tower)** and the vision run needs `AutoModelForImageTextToText` → `Qwen3_5ForConditionalGeneration` (5.175B, 333.5M visual); the **existing LoRA `target_modules` attach to the same 128 modules, none in the vision tower** (no config change needed); **mixed image+text batches run in one forward** and left-padding keeps the final token in the last column of every row, so `labels[:, last_col]` masking and the `logits[:, -1, :]` restricted-logit read are **unchanged**; VL weights 9.51 GB, peak 9.60 GB on the L40S.
- [x] **`pillow` and `torchvision` declared in `pyproject.toml`** (2026-09-24) — `transformers` 5.17.0 makes torchvision a hard dependency of *any* image processor (`AutoImageProcessor` raises `ImportError` without it), so the committed `training/probe_vision_path.py` does not run from a fresh checkout without them. `torchvision==0.29.0+cu130` resolves cleanly against the installed `torch==2.14.0+cu130`; `uv lock` added exactly those two packages and left torch untouched. **Still open:** fetch `preprocessor_config.json` into the snapshot (one network-enabled `AutoProcessor.from_pretrained` call — the text-only path never downloaded it).
- [ ] **Step 0 control run, ~20 min, no training.** Remap `checkpoints/ekvachan-decoder-qwen-benchcorpus/adapter` keys (`base_model.model.model.layers.N.…` → `base_model.model.model.language_model.layers.N.…`), load into `Qwen3_5ForConditionalGeneration`, re-run 13a.10's **text-only** eval. Purpose: prove the model-class swap is neutral *before* any vision data exists, so later movement is attributable to the data and not the class. Write the result into 13a with its manifest path either way.
- [ ] **`training/build_vision_slice.py`** — fork of `build_benchcorpus_slice.py` (fork, don't edit; same policy as every step in this lineage). Consumes image-bearing `bjb export` records; keeps 13a.10's exact 62,000 text rows unchanged so the CLINC150 continuity check stays comparable; resolves each image to the content-addressed cache by sha256 and asserts the hash matches; **asserts `image_tokens + text_tokens < max_length` per row and fails loudly rather than truncating**; refuses to reorder an ordinal `score` scale.
- [ ] **`training/train_decoder_lora_vision.py`** — fork of `train_decoder_lora_benchcorpus.py` with exactly four changes: `AutoProcessor` for `AutoTokenizer`; `AutoModelForImageTextToText` for `AutoModelForCausalLM`; a collate that carries `pixel_values`/`image_grid_thw`/`mm_token_type_ids`; and an assertion that **no LoRA module landed in `model.visual`** (make the accidental freeze deliberate). Set `max_pixels=256*28*28` and `--max-length 768`. Raise the shared-GPU free-VRAM guard from 10 GB to 14 GB. Everything else — letter-token assertion, shuffle-vs-ordinal order, OOM skip, temperature fit, `_report_by_key` — carries over verbatim.
- [ ] **Smoke run before the real one** — 2,000 examples (1,500 text / 500 vision), ~15 min. This lineage has always smoked first (`…-wideschema-smoke`, `smoke2`, `smoke3`), and the silent truncation failure PRD 5.2b describes (an uncapped 720p screenshot is **880 image tokens** and would eat the entire text prompt at `--max-length 384`) is exactly what a smoke run catches.
- [ ] **Real run: ~84,000 examples, 1 epoch, est. 7–8 GPU-hours** (62,000 text unchanged + ~10,000 Atari-HEAD + ~8,000 GUI + ~4,000 vision breadth). Estimate derived from 13a.10's measured 62,000-in-4h40m rate scaled by token cost, **not** from parameter count — per PRD §9's own caution. Launch with the dev-guidelines §4 `setsid`/`nohup`/`PYTHONUNBUFFERED=1` pattern; confirm the `causal_conv1d` / `flash-linear-attention` kernel situation in the preflight (both still fall back to reference implementations, unchanged since 13a.2).
- [ ] **Vision eval + the pre-committed regression gate.** In-distribution: held-out Atari-HEAD **trials** (never frames — frame-level splits leak) and held-out GUI items. Out-of-distribution third-party: **ScreenSpot-v2** (Apache-2.0, ~1.2k items, zero training exposure) — the vision analogue of what JevBench/jabr-v2 are for text in 13a.7. **Gate, committed now so it can't be rationalised later: CLINC150 zero-shot ≥ 95.5% and text `choice` on 13a.10's common sources within 1 pp.** Clear both → ship the single mixed adapter. Fail either → rerun the same builder with `--no-vision` and ship a text-only adapter alongside, with the measured regression stated in the README.
- [ ] **Report vision-`score` as an under-populated, excluded stratum** rather than inventing an ordinal scale for GUI/game data — same discipline better-jev-bench applied to `mod_multimodal` itself.
- [ ] *(blocked on the sibling repo)* **Multimodal data comes from better-jev-bench, not from here** — see its PRD §13 and `STATUS.md`. This repo needs `bjb export` to emit image-bearing records; it should not grow a second, parallel data pipeline. **Exception**: §5.3b's DAgger on-policy harness data (ViZDoom / `tsai-sc` / `jev-ultrafast`) stays here, after the static slate, alongside the game-harness work it depends on.
- [ ] Conformal prediction calibration upgrade (PRD 5.3a)
- [x] **Decoder calibration anomaly resolved** (2026-09-23, PRD 13a.9) — it was never a bug. An oracle temperature sweep on real eval data confirms T=1.0 (raw) is already the global-optimum ECE/Brier; the "logit surrogate" hypothesis in 13a.2 doesn't hold up (the math is provably exact except for a negligible clipping artifact). The earlier fitted temperatures (0.86-1.15) were small-calibration-sample noise pulling a 1-D fit slightly off the true optimum, confirmed by refitting on a 3.3x larger pool and watching it move closer to 1.0. No code change needed — `DecoderMultischemaBackend`'s existing raw-by-default was already correct, now for a verified reason.
- [ ] DAgger data collection method for the game/computer-use slice (PRD 5.3b)
- [ ] Game/computer-use harnesses: ViZDoom, StarCraft (`tsai-sc`), browser-use/jev-ultrafast (PRD 8.1) — zero code
- [ ] Cascade serving architecture, nano→base escalation (PRD 4.1 G5)

## Next up, with GPU time estimates

Anchor: this server's real Qwen3.5-4B LoRA runs took 65–75 min for 24k examples / 1 epoch. Single GPU, NVIDIA L40S, 46068 MiB VRAM.

**Reordered 2026-09-23, owner-directed.** Prior order chased individual open questions (calibration, score's weak spot) as they surfaced — both now resolved (13a.9, PRD 13a.8 follow-up). This reorder is a bigger structural change: **better-jev-bench comes first**, on the reasoning that the model isn't actually done training (13a.8's own finding: `score`/`noul` are real but narrow, one dataset each) and a wider, automated, easy-to-use training/eval corpus is the right lever before spending more GPU hours or making any public comparison claim. Game harnesses (ViZDoom/StarCraft/browser-use) — the actual "beats Von" comparison, since Von's headline numbers are ViZDoom-based — now sit inside the vision/nano fine-tuning phase rather than as a standalone earlier item, since they're real evaluation work best done once the model has had the benefit of the wider bench data. Large tier is explicitly last (owner's call): a stretch/optional tier with no design work started at all.

| # | Phase | Task | GPU time | Notes |
|---|---|---|---|---|
| 1 | **better-jev-bench** | Build the loader/plugin framework, CI validation gates, and automated download/processing pipeline for the ~51 Tier-A datasets; make it genuinely easy to point at for benchmarking or training, not just cataloguing | none | **framework + first 8 datasets done 2026-09-23** (see the cross-reference note under this table); ~43 Tier-A entries and the scoring implementation remain. See `better-jev-bench/STATUS.md` for the live checklist |
| 2 | ~~**Fine-tune: base**~~ done 2026-09-23 (PRD 13a.10, `checkpoints/ekvachan-decoder-qwen-benchcorpus`, 4h40m): **noul 88.80%->95.48% (+6.68pp), score 75.40%->80.95% (+5.55pp)**, CLINC150 zero-shot regression check 95.73%->96.77% (held/improved). `choice` aggregate 92.02%->83.03% is NOT a real regression -- eval composition changed to include genuinely harder new domains (CFPB 59.02%, GoEmotions 57.88%) never tested before; tasks common to both runs (DBpedia, NLI, CLINC zero-shot) all held steady or improved | done | — |
| 3 | **Fine-tune: vision** | Mixed text+image run on the same 4B backbone and the same adapter lineage — **decided and planned in full 2026-09-24, PRD 5.2b**, not a separate model. Image path verified end to end (`training/probe_vision_path.py`) | **~7–8 hrs** (revised up from ~1–2 hrs: the estimate now comes from 13a.10's measured rate scaled by real image-token cost, not from a guess) | Non-GPU work first: deps, the step-0 control, the two forked scripts, and better-jev-bench's `mod_multimodal` stratum. See the vision block above for the ordered checklist |
| 4 | **Fine-tune: nano** | Smaller model, same bench-derived data mix | ~30–60 min | blocked on #1-3 landing first (data mix reuses their work) |
| 5 | **Game harnesses** | ViZDoom/StarCraft/browser-use, run against whichever checkpoint is current at this point in the sequence — the actual head-to-head with Von | <30 min/harness compute; integration work is non-GPU | this is the real "how do we compare to Von" milestone, deliberately placed after the bench-driven retrain so the comparison reflects the improved model, not the narrow one |
| 6 | **Quantization** | Export + eval | ~10–20 min | needs a serving target — already have one (`serve/`) |
| 7 | **Go public** | README/PRD/STATUS final pass, GitHub polish, announcement posts | none | after 1-6 land, so the public story is backed by the wider-data model and a real Von comparison, not the narrow one |
| 8 | **Rust/ONNX port** | Real "faster than Jev" latency work | ~10–20 min GPU for validation; the port itself is non-GPU engineering | can trail the public release — current Python server (87-109ms) already proves the wire contract honestly with a stated target |
| 9 | **Cross-attention head** (5.1a) | Accuracy experiment + >26-option insurance | ~1–2 hrs if pursued | low priority — current mechanism already gets 83-89% on real benchmarks; the widest real option count seen in any real benchmark data is 6, so the >26-option case it uniquely solves hasn't come up yet |
| 10 | **Large tier** | 4-9B, only if a cascade design (not yet built) proves smaller tiers aren't enough | ~5-6 hrs if pursued | explicitly last — owner's call, no design work started |

### Cross-reference: what better-jev-bench now provides for phase 2 (added 2026-09-23)

*Written from the sibling repo's side of the boundary. Authoritative numbers live in
`better-jev-bench`'s own `STATUS.md` and PRD §12 — re-derive from there rather than from this
summary (dev-guidelines rule 3).*

Roadmap phase 1 is far enough along that phase 2 is no longer blocked on it. As of 2026-09-23
[better-jev-bench](https://github.com/asp616848/better-jev-bench) holds **422,878 normalised
items across 8 Tier-A datasets and 11 tasks** — 278,513 in a public training slice and 21,174 in
a frozen, hash-committed held-out slice. All four of its width strata (2 → 151 options) and all
three primitives are populated.

The part that matters for this repo specifically:

- **`bjb export` emits this project's own record shape**, key for key —
  `{state, question_key, question_type, instructions, options, label, label_idx, source}` — read
  off `training/data.py` and `training/build_primitives_slice.py` rather than invented. Verified
  by running `build_primitives_slice.py`'s own `_validate()` verbatim against a real export:
  5,500 records, 0 failures. So a bench slice is drop-in for the next training run; no adapter,
  no new builder script.
- **Option-width narrowing to the 26-letter budget is handled on the bench side**, mirroring
  `_sample_wide_subset()`. The bench stores CLINC150's true 151-way schema; the export narrows it.
- **Ordinal `score` scales are never narrowed or shuffled**, and the bench's CI refuses to ship a
  `score` task whose items disagree on scale order — the invariant
  `build_primitives_slice.py`'s docstring argues for, enforced upstream.
- **It directly addresses 13a.8's stated narrowness.** `score` and `noul` training here is
  currently one dataset each (a 3-level sentiment set and one BoolQ split). The bench adds a
  5-level ordinal `score` source built from real annotator-agreement fractions, and two `noul`
  sources with opposite skew profiles (one balanced 50/50 by construction, one 92% negative).
  Whether that moves the 75.40% `score` number is an open empirical question; neither repo
  predicts it.

What it does **not** yet provide: the scoring spec is still a specification — no `/v1/evaluate`,
no axis scores, and no ekVachan checkpoint has been run against the corpus. The held-out slice is
frozen and tamper-evident but not secret (its labels are in a public repo), so the usual caution
about training on an eval set applies to whoever pulls it.

Practical entry point for phase 2:

```bash
git clone https://github.com/asp616848/better-jev-bench && cd better-jev-bench && pip install -e .
bjb build                                                   # regenerate the public slice
bjb export --out exports/mix --tiers A --max-options 26     # -> exports/mix/public.jsonl
```

## Review log

**2026-09-23 — independent review pass (PM/roadmap audit, no training run).**

Scope: read `skills/dev-guidelines/SKILL.md`, `PRD.md`, and this file in full; spot-checked every load-bearing claim against real repo state rather than prose.

Verified as accurate: all four headline training/eval numbers against their checkpoint manifests (86.32 / 92.25 / 96.10 / 98.5, including ECEs); both third-party benchmark numbers and item counts against the committed `results/` manifests; `DecoderMultischemaBackend` and `filter_supported_multischema` exist as described; `serve/` really is encoder-only; git sync clean in both directions against `origin/main` on both repos.

Corrected: the SDK item was misdiagnosed (tests pass, 7/7 — no missing dependency); the `ekvachan-setup` skill item was wrong (its status note was already current, a different line was stale); the cross-attention head's claim to be "the real path past the remaining 92/231 and 557/944" was **wrong** and is now reframed — those items are 100% `score`/`noul` type, per the schema filters' own unsupported-reason counts.

Found and recorded as new open items: training/eval manifests are gitignored and uncommitted (PRD 8.2 / rule 10 gap); hardware still unrecorded; untracked `sdk/python/uv.lock`.

Implemented in this pass (one item, deliberately): the **README.md rewrite** — chosen over the other open items because it is pure documentation (zero runtime risk), fully specified, and was the repo's most externally-visible false claim ("a second architecture arm is training now; no numbers yet", "None of the competitive benchmarks have been run yet") for a project whose entire differentiator is evidence discipline. The `serve/` decoder port was explicitly *not* attempted here: it is real engineering against the live wire contract, needs a GPU to validate, and is the wrong thing to half-do inside a review. It stays as task #2.

Also corrected in `PRD.md`: Section 13's Phase 1 roadmap still described the decoder arm as mid-run with no numbers and benchmarks/evidence bundles as "not started", directly contradicting 13a.2 and 13a.7 in the same document; and Section 5.1 now carries an explicit superseded-by-14-Q4 banner instead of silently reading as current direction (dev-guidelines rule 1).
