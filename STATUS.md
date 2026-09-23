# ekVachan — status

Living task list for `better-jev-for-all` (the model/serving/benchmark-harness repo). Update this file in the same commit as the work it describes — mark items done only after verifying them against real repo/server state, not after an agent or a prior summary claims they're done (see `skills/dev-guidelines/SKILL.md`). Full reasoning for every decision lives in `PRD.md`; this file is the checklist, PRD.md is the why.

Sibling project: **better-jev-bench** (separate repo, separate PRD, own `STATUS.md`) — a standalone dataset/benchmark corpus. Don't mix work across the two repos in one commit.

Last verified: 2026-09-23.

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

## In progress / needs fixing (no GPU required)

- [ ] `serve/inference.py` + `serve/server.py` only serve the **old fixed-3-way encoder** — never updated for the decoder, despite it being the chosen architecture (PRD 14 Q4). This blocks everything that needs a live server against the real model.
- [ ] SDK test suite (`sdk/python/`) currently fails to collect — missing `uvicorn` dev dependency. Fix env, then re-verify "tested" claims for real.
- [ ] `README.md` stale — still says decoder is untrained, no mention of the Q4 decision or real JevBench/jabr-v2 numbers.
- [ ] `skills/ekvachan-setup/SKILL.md` status note stale — same issue as README.

## Not started

- [ ] Rust/ONNX serving port (PRD 7.1) — no code exists
- [ ] Quantization export (PRD 7.2)
- [ ] Cross-attention decision head (PRD 5.1a) — unlocks `noul`/`score` and >26-option items; the real path past the remaining 92/231 and 557/944 unanswerable items
- [ ] Multi-tier model family: nano tier, vision tier, large tier (PRD 5.2/5.2a) — only base (encoder) + one 4B decoder exist
- [ ] Conformal prediction calibration upgrade (PRD 5.3a)
- [ ] DAgger data collection method for the game/computer-use slice (PRD 5.3b)
- [ ] Game/computer-use harnesses: ViZDoom, StarCraft (`tsai-sc`), browser-use/jev-ultrafast (PRD 8.1) — zero code
- [ ] Cascade serving architecture, nano→base escalation (PRD 4.1 G5)

## Next up, with GPU time estimates

Anchor: this server's real Qwen3.5-4B LoRA runs took 65–75 min for 24k examples / 1 epoch. Single GPU, ~46GB VRAM.

| # | Task | GPU time | Blocked on |
|---|---|---|---|
| 1 | Wire decoder support into `serve/` | none | — |
| 2 | Fix SDK test env, re-verify | none | — |
| 3 | Rewrite README + setup skill status note | none | — |
| 4 | Build cross-attention head (5.1a), then train | ~1–2 hrs | dev work first |
| 5 | Nano tier training | ~30–60 min | data mix decision |
| 6 | Vision tier | ~1–2 hrs | data pipeline (non-GPU) first |
| 7 | Game harness wiring + eval | <30 min/harness | harness integration (non-GPU) first |
| 8 | Quantization export + eval | ~10–20 min | a serving target to quantize |
| 9 | Large tier (optional/stretch) | ~5–6 hrs | cascade proving it's needed |
| 10 | ONNX/Rust port validation | ~10–20 min | the port itself (non-GPU) |
