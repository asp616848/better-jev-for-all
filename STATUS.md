# ekVachan — status

Living task list for `better-jev-for-all` (the model/serving/benchmark-harness repo). Update this file in the same commit as the work it describes — mark items done only after verifying them against real repo/server state, not after an agent or a prior summary claims they're done (see `skills/dev-guidelines/SKILL.md`). Full reasoning for every decision lives in `PRD.md`; this file is the checklist, PRD.md is the why.

Sibling project: **better-jev-bench** (separate repo, separate PRD, own `STATUS.md`) — a standalone dataset/benchmark corpus. Don't mix work across the two repos in one commit.

Last verified: 2026-09-24 (**>26-option ceiling resolved by measurement — PRD 5.1b**. Planning-and-probing pass in the 5.2b mould: `training/probe_multitoken_scheme.py` + `results/multitoken-scheme-probe-20260924T104011Z.manifest.json`. The 26-letter ceiling turned out to be a *search* limitation, not an architectural one — **562 of 676 two-uppercase-letter strings are single tokens under Qwen3.5-4B**, giving a 588-code table with 3.9x headroom over the widest real task, at zero mechanism change. Multi-token reads are genuinely unavailable under this tokenizer; 5.1a's cross-attention head is not needed for this and is re-scoped. GPU touched only for the probe's own forward/backward measurements. Spec + ordered checklist in PRD 5.1b; no training run launched — that is the next session's work). Prior: 2026-09-24 (**ViZDoom harness built and validated end to end — PRD 13a.13**. CPU-only session, GPU untouched, 13a.11's vision run in progress throughout. `benchmarks/vizdoom/` + `benchmarks/common/episodic.py`, validated against the real headless ViZDoom engine with rubric-oracle/random/mock policies; real numbers recorded, including a real, honest surprise (the oracle baseline scores above Von's own published 9.00 kills). Real text-arm run against a trained checkpoint is the one remaining step — see "ViZDoom" below). Prior: 2026-09-24 (**ViZDoom design pass — PRD 8.1d**. Planning only, before the build above. ViZDoom verified installable and runnable headless on this server; Von's own ViZDoom harness fetched and read; two honest findings about Von's published numbers recorded). Prior: 2026-09-24 (ScreenSpot-v2 benchmark harness built + validated, PRD 13a.12 — CPU-only session, GPU untouched, vision training run in progress throughout). Prior: 2026-09-24 vision-tier planning pass — PRD 5.2b, `results/vision-path-probe-20260923T201758Z.manifest.json`; 2026-09-23 independent review pass — see "Review log" at the bottom.

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
- [x] **ScreenSpot-v2 benchmark harness built and validated end to end** (2026-09-24, CPU-only, PRD 13a.12) — `benchmarks/screenspot_v2/` (`loader.py`/`run.py`/`README.md`/`NOTICE.md`/`fixtures/`/`vendored/`), the vision analogue of JevBench/jabr-v2, per PRD 5.2b's design. **858 real items** (not the upstream manifest's headline 898 — see 13a.12), vendored via `better-jev-bench`'s `bjb export --slice heldout`, reframed from ScreenSpot-v2's own binary grounding format into genuine **N-way (2-6 option) grounding-as-choice items**, 0 excluded for too-small a candidate pool. `benchmarks/common/backends.py` gained `DecoderVisionMultischemaBackend`/`-Mock` (vision-capable sibling of `DecoderMultischemaBackend`, sharing its restricted-logit mechanism via two factored-out helper functions rather than reimplementing it); `Item` gained an optional `image_path` field. Validated with the mock backend against the real 858-item dataset (real image sha256 re-verification against better-jev-bench's cache, 858/858 filtered as supported) and against the synthetic self-test fixture (exercises the pool-too-small exclusion path for real, since real data never triggers it) — both produced committed manifests in `results/`. JevBench/jabr-v2's own mock/selftest runs re-verified unaffected (139/231, 387/944 unchanged) after the shared-file changes this required. **Not yet run for real** — `checkpoints/ekvachan-decoder-qwen-vision` (PRD 13a.11) was still training throughout this work, and this session was explicitly scoped to never touch the GPU. See 13a.12 for the exact one-line command to run once it lands.
- [x] **ViZDoom episodic harness built and validated end to end against the real engine** (2026-09-24, CPU-only, PRD 13a.13) — `benchmarks/vizdoom/` (`env.py`/`observe.py`/`rubric.py`/`run.py`/`README.md`/`NOTICE.md`/`vendored/LICENSE`) plus the new shared `benchmarks/common/episodic.py::run_episodic_harness()` (the closed-loop counterpart to `run_harness()` — ViZDoom has no gold label and a model-generated item sequence, unlike the three static benchmarks). Validated against the real headless ViZDoom engine (not mocked): the rubric-oracle policy scores **11.125 kills / 19.71s survival** (vs. Von's published 9.00/12.11 — a real, honest surprise, caveated in 13a.13/README), the random-policy baseline reproduces the 8.1d planning pass's own probe (**1.275 kills / 14.65s** vs. the probe's 1.53/14.09), and the `decoder-multischema-mock` wiring self-test runs clean on both scenarios and both rubric conditions. `benchmarks/common/backends.py` gained one new, deliberately general backend, `RandomChoiceBackend` (`--backend random`) — a real publishable control, not a wiring self-test. DAgger-teacher data emission (PRD 5.3b / 8.1d Finding 4) is wired and validated (90 real `(text, frame, action)` triples produced in this session), pulled forward from its original "after the vision arm" sequencing since it was nearly free once the episodic loop existed. JevBench/jabr-v2/ScreenSpot-v2's own mock/selftest runs re-verified unaffected (139/231, 387/944, 858/858 unchanged) after the shared-file changes this required. **Not yet run for real** — this session was explicitly scoped to never touch the GPU. See 13a.13 for the exact command to run once the GPU is free.

- [x] **13a.22 routing fix verified live on GPU** (2026-09-25, PRD 13a.23) — 77-option text vs `benchcorpus` router fails loudly (HTTP 501 naming the adapter and its manifest cap of 26, not a silent 200; 10-option control 200s), same request vs `wide`-in-vision-slot returns 200 (`adapter="vision"`); the real `vision` checkpoint (manifest cap 26) also 501s on 77 options, which is the per-adapter design working, not a regression.
- [x] **`causal-conv1d` closed as uninstallable without root** (2026-09-25, PRD 13a.23) — the `nvidia-cuda-nvcc-cu12` pip wheel ships no `nvcc` driver (source build fails with `FileNotFoundError: '/usr/local/cuda/bin/nvcc'`), PyPI is source-only, and upstream prebuilts top out at torch 2.10 (nothing for torch 2.14.0+cu130/cp313); standing pre-kernel baseline **p50 91.1ms / p95 108.7ms** (n=30, benchcorpus 10-option over HTTP); `RoutingDecoderModel`'s `self._lock` serialization confirmed as documented by-design behavior, not rebuilt.

- [x] **Request-lifecycle profiled, no code changed** (2026-09-25, PRD 13a.24) — forward pass 77-84% of in-server time (quantization-validated as the biggest slice), `set_adapter()` 10-12% pure overhead, HTTP/dispatch ~19%, template ~3%; burst confirms fully-serial serving (8-concurrent wall ~= 8x sequential). Absolute forward latency varies +-20% run-to-run under constant ~12GB shared occupancy. Decision on order (quantize vs overhead/architectural work) explicitly deferred.

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
- [ ] Cross-attention decision head (PRD 5.1a) — **re-scoped 2026-09-24, PRD 5.1b.** Its original headline justification (the route to a general `choice` primitive) was discharged by PRD 13a.6; its fallback justification (insurance for >26-option schemas) became real at 13a.15 (`bjb evaluate` Generality **0.0**, six real corpus tasks over 26 options) and was then **resolved without it** at 5.1b: `training/probe_multitoken_scheme.py` measured a **588-code single-token identifier table** under Qwen3.5-4B's tokenizer, so >26-option `choice` is reachable with zero mechanism change. What genuinely survives for 5.1a, both measured in 5.1b: a **2.4x serving win on fixed schemas** (59.5 ms cached state encode vs 145.5 ms for a 151-option prompt) and option sets beyond 588. Still Phase 4, now on honest grounds.
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
- [ ] **Vision eval + the pre-committed regression gate.** In-distribution: held-out Atari-HEAD **trials** (never frames — frame-level splits leak) and held-out GUI items. Out-of-distribution third-party: **ScreenSpot-v2** (Apache-2.0, 858 real items, zero training exposure) — the vision analogue of what JevBench/jabr-v2 are for text in 13a.7. **The harness side of this is done** (PRD 13a.12, `benchmarks/screenspot_v2/`, validated with a mock backend against all 858 real items) — what's left is exactly `python -m benchmarks.screenspot_v2.run --backend decoder-vision-multischema --checkpoint-dir checkpoints/ekvachan-decoder-qwen-vision` once training finishes. **Gate, committed now so it can't be rationalised later: CLINC150 zero-shot ≥ 95.5% and text `choice` on 13a.10's common sources within 1 pp.** Clear both → ship the single mixed adapter. Fail either → rerun the same builder with `--no-vision` and ship a text-only adapter alongside, with the measured regression stated in the README.
- [ ] **Report vision-`score` as an under-populated, excluded stratum** rather than inventing an ordinal scale for GUI/game data — same discipline better-jev-bench applied to `mod_multimodal` itself.
- [ ] *(blocked on the sibling repo)* **Multimodal data comes from better-jev-bench, not from here** — see its PRD §13 and `STATUS.md`. This repo needs `bjb export` to emit image-bearing records; it should not grow a second, parallel data pipeline. **Exception**: §5.3b's DAgger on-policy harness data (ViZDoom / `tsai-sc` / `jev-ultrafast`) stays here, after the static slate, alongside the game-harness work it depends on.
- [ ] Conformal prediction calibration upgrade (PRD 5.3a)
- [x] **Decoder calibration anomaly resolved** (2026-09-23, PRD 13a.9) — it was never a bug. An oracle temperature sweep on real eval data confirms T=1.0 (raw) is already the global-optimum ECE/Brier; the "logit surrogate" hypothesis in 13a.2 doesn't hold up (the math is provably exact except for a negligible clipping artifact). The earlier fitted temperatures (0.86-1.15) were small-calibration-sample noise pulling a 1-D fit slightly off the true optimum, confirmed by refitting on a 3.3x larger pool and watching it move closer to 1.0. No code change needed — `DecoderMultischemaBackend`'s existing raw-by-default was already correct, now for a verified reason.
- [x] **DAgger data collection method for the game/computer-use slice (PRD 5.3b) — wired and validated 2026-09-24 (PRD 13a.13), pulled forward from its original "step 9, optional, after the vision arm" sequencing.** `benchmarks/common/episodic.py::DaggerSink` emits real on-policy `(text, frame, teacher_action, model_action)` triples (90 rows + 90 PNGs produced in this session's own validation run). Nothing consumes it yet — that's the vision-arm trainer, still not started.
- [ ] Game/computer-use harnesses: **ViZDoom — harness built and validated 2026-09-24 (PRD 13a.13); only the real run against a trained checkpoint remains (one command, see 13a.13/`benchmarks/vizdoom/README.md`).** StarCraft (`tsai-sc`), browser-use/jev-ultrafast (PRD 8.1) — no design, zero code
- [ ] Cascade serving architecture, nano→base escalation (PRD 4.1 G5)

## ViZDoom — harness built and validated 2026-09-24 (PRD 8.1d design, 13a.13 build)

Design resolved in PRD 8.1d (a planning pass); **the harness itself was then built in a second
CPU-only session the same day** — `benchmarks/vizdoom/` + `benchmarks/common/episodic.py`, no GPU
touched, no `training/` lineage file touched. Read PRD 13a.13 for the full build record
(real numbers, the honest oracle-beats-Von surprise, the DAgger emission). The checklist below is
8.1d's own, in checkbox form, now showing real completion status.

**Verified for real on this server (CPU only), not assumed:**

- [x] **ViZDoom installs and runs fully headless here.** `vizdoom==1.3.1` from a **prebuilt
      manylinux wheel** — no compilation, no CMake, no system Doom. 7 packages, ~65 MB, 1m38s.
      `DISPLAY` is empty and both scenarios ran to completion with `set_window_visible(False)`:
      the software renderer needs no GL context, so **`xvfb-run` is not required**. Both
      `defend_the_center` and `health_gathering` (`.cfg` + `.wad`) ship inside the wheel at
      `vizdoom.scenarios_path`. Labels buffer and depth buffer both work headless — which is what
      makes Von's text observation reproducible here at all. Throughput ~1,300–1,800 ticks/s.
      **Trap, recorded:** this server's `python3` is **3.6.15** and `pip install vizdoom` under it
      falls back to a source build and fails — use the repo's `uv`/3.11+ environment.
- [x] **Von's ViZDoom harness fetched and read** (`wfzyx/von@master`, `benchmarks/doom_eval.py`
      + `benchmarks/run_doom_benchmark.py`, via the GitHub API 2026-09-24). **Von's observation is
      text, not pixels** — built from the labels buffer, the depth buffer and 3 game variables;
      `screen_buffer` is read only for `H, W` normalisation. 3-option `choice`, `tics=4`, 300-step
      cap, 8 published seeds per scenario. So the text-vs-vision question is settled by evidence.
- [x] **License item closed** (was explicitly open in PRD 7.4). Von: **Apache-2.0**. ViZDoom: MIT
      for its own code, bundling GPL-descended ZDoom and 3-clause-BSD Freedoom assets — consumed
      as a published wheel, never forked or vendored, so only notice/attribution applies.

**Two honest findings that change what 8.1 item 2 promises:**

- [x] **`Health Gathering` survival does not discriminate.** Measured here (uniform random,
      8 seeds × 10 episodes, stock config, frameskip 4): **14.09 s**, beating **Von's 12.11 s and
      Jev's 13.03 s**. Frameskip sweep (1/2/4/8/12 → 12.83/13.33/14.04/16.19/15.80 s) does not
      explain it away — random meets or beats 12.11 s at every setting. **Von's own
      `run_doom_benchmark.py` prints the same conclusion**: its comparison table lists
      `Random Action Baseline: … 15.77 s`. Consequence: the harness still reports survival, but
      every published figure carries the random baseline beside it, and the bar becomes "beat
      random on the same seeds."
- [x] **`Defend the Center` kills is the real benchmark.** Random 1.53 measured here / 1.88 in
      Von's own table (agreeing within one per-seed sd), always-attack 1.38, always-turn-left
      0.00, Jev 5.62, Von 9.00 — a wide, well-ordered range where a result means something.
- [x] **Von's `instructions` embed a complete decision rule**, and `format_doom_state()` has
      already quantised enemy position into the same buckets that rule keys on. Resolution: run
      **both** `--rubric von` (verbatim; the headline like-for-like number) and `--rubric none`,
      and publish both. The gap is a direct measurement of PRD 3.1's claim that Von degrades
      without rubric text.

**Decision: text arm first (it is the headline Von comparison), vision arm second on identical
seeds as an ekVachan-only capability result.** Not either/or — the marginal cost of the second
arm is one observation function, and vision is the only place in the standing suite where
ekVachan can demonstrate something Von structurally cannot, in the same environment on the same
metric.

### Implementation checklist (PRD 8.1d) — steps 1-5 built and validated 2026-09-24, PRD 13a.13

- [x] **1.** Added `vizdoom>=1.3.1` to `pyproject.toml`; `uv sync` resolved it in <1s, `uv.lock`
      diff is 105 insertions / 0 deletions, zero lines touching `torch`. `uv run python3 -c
      "import vizdoom"` succeeds under this repo's Python 3.13 (not the system 3.6.15 trap).
- [x] **2.** `benchmarks/vizdoom/env.py` — `DoomEnvironment`/`DoomSnapshot`, Von's exact config.
      The `get_available_buttons()` order assertion ran for real against the live engine on both
      scenarios and passed; a throwaway smoke episode registered a real `kills=1` after 20
      `attack` steps before any harness code touched it.
- [x] **3.** `rubric.py` + **the rubric-oracle baseline, run for real**: **11.125 kills** /
      **19.71s survival** (8 published seeds, 1 ep/seed, `--rubric von`) — see PRD 13a.13 for the
      honest surprise (this beats Von's own published 9.00). The random baseline was also
      re-measured with this harness (not just the planning-pass probe): **1.275 kills / 14.65s
      survival**, matching the probe's 1.53/14.09 within noise. Both committed as evidence
      bundles under `results/vizdoom-*` (rule 10).
- [x] **4.** `observe.py::observe_text()`, byte-faithful to `format_doom_state()`; bucket
      boundaries unit-tested (`python -m benchmarks.vizdoom.observe`) — passes.
- [x] **5.** `benchmarks/common/episodic.py::run_episodic_harness()` — the new closed-loop
      counterpart to `run_harness()`, reusing `build_backend`, `make_arg_parser`, `write_bundle`,
      `eval/metrics.py` unchanged. Validated with `--backend decoder-multischema-mock` against the
      real ViZDoom engine (both scenarios, both rubrics, 4 committed evidence bundles) — same
      no-torch/no-checkpoint discipline 13a.12 used for ScreenSpot-v2. JevBench/jabr-v2/
      ScreenSpot-v2's own mock/selftest runs re-verified unaffected (139/231, 387/944, 858/858
      unchanged) after the one new `RandomChoiceBackend` this required in the shared
      `backends.py`/`harness.py`.
- [ ] **6.** **Text arm for real** — `--backend decoder-multischema --checkpoint-dir
      checkpoints/ekvachan-decoder-qwen-benchcorpus`, both scenarios, both rubric conditions, all
      16 published seeds. **This is the milestone and it needs nothing that does not exist** —
      deliberately not attempted in this build (GPU/checkpoint scoped out; a ~6h vision training
      run was active throughout).
- [ ] **7.** Write results into PRD 13a and here, win or lose, with manifest paths and Finding 2's
      caveat in the text rather than a footnote. (13a.13 already records everything *except* step
      6's real-checkpoint numbers, which don't exist yet.)
- [ ] **8.** **Vision arm** once `checkpoints/ekvachan-decoder-qwen-vision` lands and ScreenSpot-v2
      has produced its first real number. Same seeds/scenarios/metrics; reported as an
      ekVachan-only capability result, explicitly **not** a Von comparison.
- [x] **9.** **Pulled forward, ahead of its original "optional, after 8" sequencing** — DAgger data
      emission is wired and validated now (`benchmarks/common/episodic.py::DaggerSink`), since it
      is nearly free once the episodic loop exists. Real run: `--dagger-out
      training/dagger_data/vizdoom` on one seed produced 90 real `(text, frame, teacher_action)`
      JSONL rows + 90 matching real PNG frames. Gitignored, nothing consumes it yet — see PRD
      13a.13.

Out of scope, deliberately: StarCraft and browser-use — separate harnesses, separate design
questions, and neither blocks the Von comparison.

## Wide-option `choice`: the >26 ceiling resolved by measurement, 2026-09-24 (**PRD 5.1b**)

**Decision: extend the existing restricted-logit table from 26 single letters to a generated 588 single-token codes (A-Z, then the 562 two-uppercase-letter strings that are single tokens under Qwen3.5-4B). Not a multi-token read — that is genuinely unavailable under this tokenizer. Not 5.1a's cross-attention head — not needed for this.** Full reasoning and every measured number in PRD 5.1b; evidence bundle `results/multitoken-scheme-probe-20260924T104011Z.manifest.json`.

- [x] **Identifier-scheme capability probe built and run** (2026-09-24) — `training/probe_multitoken_scheme.py`, re-runnable as `uv run python3 -u -m training.probe_multitoken_scheme --gpu-forward`. Measured on this server: A-Z re-verified single-token (ids **32-57, contiguous**); **562 of 676 two-uppercase-letter strings are single tokens**, giving a **588-code table with 3.9x headroom over the widest real task (clinc150, 151-way)**; **all 588 context-stable, 0 failures**, and appending any code to the real chat template's generation prompt extends the sequence by exactly that one id; a genuine two-position read **does not exist** (the 562 that merge cannot be split); full-width prompts fit (**clinc150 839 tokens**, under the existing `--max-length 1280`); inference prefill **74.7 ms at n=26 -> 145.5 ms at n=151** (one pass, one position, mechanism unchanged); training fwd+bwd **687.7 ms at width 26 -> 3825.1 ms at width 151** (5.56x, peak VRAM 12.20 -> 20.35 GB), with **batch 2 both faster per row and 26% lighter than batch 4** at width 151; the wide table's untrained prior is **flatter than the 26-letter table already in production** (normalized entropy 0.479 vs 0.362), so no code-selection engineering is warranted.
- [x] `build_code_table(tokenizer)` in `training/decoder_lora_lib.py`, derived and asserted at runtime — never hardcoded (5.1b item 0). Done 2026-09-24, PRD 13a.17 -- verified live against the real tokenizer to reproduce the probe exact 588-code finding.
- [x] Widen the four hand-built letter tables: `decoder_lora_lib.py:52`, `benchmarks/common/backends.py:475` and `:711`, `serve/inference.py:375`, plus `training/build_vision_slice.py:224` (5.1b item 1). Done 2026-09-24, PRD 13a.17.
- [x] Width-conditional answer line: `n <= 26` byte-identical to today's, `n > 26` names a range. Unit-test the byte-identity (5.1b item 2). Done 2026-09-24, PRD 13a.17 -- byte-identity verified for real, not assumed.
- [x] Caps: `DECODER_MULTISCHEMA_MAX_OPTIONS` 26 -> 588; `min_free_vram_gb` 14 -> 24; `--max-length` stays 1280 (5.1b item 3). Done 2026-09-24, PRD 13a.17.
- [x] Re-export the wide tasks at full width and add full-width rows **alongside** the existing 26-option subsets; re-run the per-row token-budget assertion over every row, old and new (5.1b item 4). Done 2026-09-25, PRD 13a.18 -- 5 tasks not 6 (clinc150 deliberately excluded, kept as the untrained zero-shot check), 12,500 rows not 15,000, documented deviation.
- [x] Length handling for the wide bucket (5.1b item 5). Done 2026-09-25, PRD 13a.18 -- not literal length-bucketed batching; instead capped the continue-train set to <=768 rendered tokens (excludes 496/26,250 long-tail rows, mostly LEDGAR) after live measurement showed a single long row could stall an optimizer step 10-12+ minutes even with flash-linear-attention installed. batch=1/grad-accum=64, not 2x32.
- [x] Smoke run skipped in favor of going straight to the real run (5.1b item 6) -- reasonable given the continue-train's already-modest 26,250-row scope and the periodic adapter_inprogress checkpoint (every 25 steps) added as a safety net. Noted as a literal checklist deviation, PRD 13a.18.
- [x] Full run + pre-committed evals (5.1b item 7). Done 2026-09-25, PRD 13a.18 -- Generality 69.88 (was 0.0/None), 100% coverage (2800/2800), CLINC150 96.07% (pass), aggregate choice 89.26% (miss vs 90.71%+-1pp, explained: the new wide tasks are genuinely harder, not a regression on the original mix), full per-width-bucket accuracy/Brier/ECE table recorded.
- [x] Wire contract to 2-588: `serve/server.py`, `serve/inference.py`, `README.md`, PRD 6.2 (5.1b item 8). Done 2026-09-24, PRD 13a.17 -- README.md/PRD 6.2 already scoped to specific historical checkpoints, not a general ceiling claim, no change needed there.

## Next up, with GPU time estimates

Anchor: this server's real Qwen3.5-4B LoRA runs took 65–75 min for 24k examples / 1 epoch. Single GPU, NVIDIA L40S, 46068 MiB VRAM.

**Reordered 2026-09-23, owner-directed.** Prior order chased individual open questions (calibration, score's weak spot) as they surfaced — both now resolved (13a.9, PRD 13a.8 follow-up). This reorder is a bigger structural change: **better-jev-bench comes first**, on the reasoning that the model isn't actually done training (13a.8's own finding: `score`/`noul` are real but narrow, one dataset each) and a wider, automated, easy-to-use training/eval corpus is the right lever before spending more GPU hours or making any public comparison claim. Game harnesses (ViZDoom/StarCraft/browser-use) — the actual "beats Von" comparison, since Von's headline numbers are ViZDoom-based — now sit inside the vision/nano fine-tuning phase rather than as a standalone earlier item, since they're real evaluation work best done once the model has had the benefit of the wider bench data. Large tier is explicitly last (owner's call): a stretch/optional tier with no design work started at all.

| # | Phase | Task | GPU time | Notes |
|---|---|---|---|---|
| 1 | **better-jev-bench** | Build the loader/plugin framework, CI validation gates, and automated download/processing pipeline for the ~51 Tier-A datasets; make it genuinely easy to point at for benchmarking or training, not just cataloguing | none | **framework + first 8 datasets done 2026-09-23** (see the cross-reference note under this table); ~43 Tier-A entries and the scoring implementation remain. See `better-jev-bench/STATUS.md` for the live checklist |
| 2 | ~~**Fine-tune: base**~~ done 2026-09-23 (PRD 13a.10, `checkpoints/ekvachan-decoder-qwen-benchcorpus`, 4h40m): **noul 88.80%->95.48% (+6.68pp), score 75.40%->80.95% (+5.55pp)**, CLINC150 zero-shot regression check 95.73%->96.77% (held/improved). `choice` aggregate 92.02%->83.03% is NOT a real regression -- eval composition changed to include genuinely harder new domains (CFPB 59.02%, GoEmotions 57.88%) never tested before; tasks common to both runs (DBpedia, NLI, CLINC zero-shot) all held steady or improved | done | — |
| 3 | ~~**Fine-tune: vision**~~ **done** 2026-09-24 (PRD 13a.14, `checkpoints/ekvachan-decoder-qwen-vision`, 6h39m): choice 90.71%, noul 95.34%, score 81.92%, CLINC150 zero-shot 96.47%. Regression check vs 13a.10 resolves cleanly *in favor* of vision: aggregate text choice +7.68pp (83.03%->90.71%), not a regression. **Decision: one adapter for everything** -- `EKVACHAN_TEXT_ADAPTER` default now resolves to the vision adapter, verified live | done | — |
| 4 | **Fine-tune: nano** | Smaller model, same bench-derived data mix | ~30–60 min | blocked on #1-3 landing first (data mix reuses their work) |
| 5 | **Game harnesses** | ViZDoom/StarCraft/browser-use, run against whichever checkpoint is current at this point in the sequence — the actual head-to-head with Von. **ViZDoom harness built and validated 2026-09-24, PRD 13a.13** (text arm; the vision arm is second, on identical seeds, once #3's checkpoint lands): `python -m benchmarks.vizdoom.run --backend decoder-multischema --checkpoint-dir checkpoints/ekvachan-decoder-qwen-benchcorpus --scenario both --rubric both` is the one remaining command | **none for the text arm** — env is ~1,300-1,800 ticks/s on CPU, so a full 16-seed 2-scenario run is ~4,800 decisions ≈ 3 min of inference; vision arm needs #3's checkpoint | this is the real "how do we compare to Von" milestone. 13a.13's oracle baseline already scores *above* Von's own published 9.00 kills — a real, honest, unresolved signal worth reading before assuming the real run will simply "win" |
| 6 | **Quantization** | Export + eval | ~10–20 min | needs a serving target — already have one (`serve/`) |
| 7 | **Go public** | README/PRD/STATUS final pass, GitHub polish, announcement posts | none | after 1-6 land, so the public story is backed by the wider-data model and a real Von comparison, not the narrow one |
| 8 | **Rust/ONNX port** | Real "faster than Jev" latency work | ~10–20 min GPU for validation; the port itself is non-GPU engineering | can trail the public release — current Python server (87-109ms) already proves the wire contract honestly with a stated target |
| 3b | **Wide-option `choice` (588-code table)** | Extend the restricted-logit table from 26 to 588 single-token codes and retrain — the fix for `bjb evaluate`'s Generality **0.0** (6 of 14 corpus tasks exceed 26 options, 13a.15). Full spec, measured evidence and ordered checklist in **PRD 5.1b**; probe + manifest already committed | **~10-11 hrs** full retrain (93,000 rows, one epoch), or ~4h30m as a continue-train from the vision adapter | **highest-leverage open item.** Zero mechanism change — one derived code table, four call sites, a width-conditional answer line, and genuinely-wide training rows |
| 9 | **Cross-attention head** (5.1a) | Accuracy experiment + cached-schema serving win | ~1–2 hrs if pursued | **re-scoped 2026-09-24 (PRD 5.1b)** — the >26-option case it was insurance for is solved by 3b at zero architecture cost. Remaining real value: 2.4x cached-schema serving win (measured) and option sets beyond 588 |
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

## JevBench/jabr-v2: complete coverage reached, 2026-09-24

Fixed the stale schema filter (still rejected noul/score with a comment that stopped being true at 13a.8) and two loader bugs it had been hiding (jabr-v2 discarded options for non-choice tasks; JevBench's score items compared an int index against string options). Both benchmarks now run to 100% coverage: **JevBench 68.40% on 231/231**, **jabr-v2 84.53% on 944/944**, both `is_complete_benchmark_score: true` for the first time. See PRD 13a.16.
