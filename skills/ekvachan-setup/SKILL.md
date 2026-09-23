---
name: ekvachan-setup
description: Install, serve, benchmark, and fine-tune ekVachan (the open System One model in this repo) — an open-weight, self-hostable, API-compatible alternative to TypeSafe's Jev. Use this skill when a user asks to set up, run, test, or customize ekVachan, or to point an existing Jev integration at a local/self-hosted model instead.
---

# ekVachan setup skill

ekVachan is an open-weight "System One" decision model: given a `state` and typed `questions` (`choice` / `score` / `noul`), it returns calibrated probabilities in one forward pass — no text generation. It ships with an HTTP API compatible with TypeSafe's Jev (`POST /v1/systemone`), so anything built against Jev works against a local ekVachan server by changing a base URL.

Full design rationale, architecture decisions, and benchmark commitments live in [`../../PRD.md`](../../PRD.md) at the repo root — read that before making any implementation decisions on behalf of a user, since it documents *why* choices were made (model sizing, why encoder-over-logit-reading, cascade design, etc.), not just what to run.

> **Status note (updated 2026-09-23):** See `STATUS.md` at the repo root for the current living checklist — read that first, it's kept more current than this note. Summary as of this update: the decoder (Qwen3.5-4B LoRA) is the **primary architecture** (PRD 14 Q4, decided 2026-09-23), not the encoder this note originally described — it beat the encoder decisively (92.25% vs 86.32% accuracy, better calibration, 20x less data), then proved genuine wide-schema generalization (96.10% zero-shot on a 15-26-way held-out schema, PRD 13a.6) and real accuracy on third-party benchmarks (JevBench 83.45%, jabr-v2 88.89%, PRD 13a.7). **However, `serve/inference.py` + `serve/server.py` still only serve the old fixed-3-way encoder** — the decoder was never wired into the serving layer, so the compatibility server does not yet reflect the chosen architecture. There is **no `ekvachan` CLI** — every command below is a direct Python invocation (`python -m ...` / `uvicorn ...`), not a packaged command. `benchmarks/jevbench/` and `benchmarks/jabr_v2/` now exist and have real results (see above). Still not built: ViZDoom/StarCraft harnesses, `/v1/finetune`, the nano/vision/large tiers, the cascade — if a user asks for any of these, say so plainly and point to `STATUS.md` / `PRD.md` Section 13 (Roadmap) instead of fabricating output.

## When to use this skill
- User wants to train or re-train the encoder or decoder-LoRA comparison arm.
- User wants to run the local reference server and query `/v1/systemone`.
- User wants to swap a Jev-integrated project (e.g. a `browser-use/jev-ultrafast`-style agent) over to a self-hosted model — subject to the one-schema limitation below.
- User wants to understand what's real vs. still aspirational in this repo (this skill's guardrails apply here specifically).

Things this skill **cannot** do yet, because the code doesn't exist: run a benchmark suite, produce an evidence bundle, fine-tune via a one-command flow, or serve `score`/`noul` questions or an arbitrary `choice` option set. Say so rather than improvising a substitute.

## 1. Install
```bash
git clone <this-repo-url> && cd better-jev-for-all
uv sync   # or: pip install -e .  — installs torch, transformers, datasets, accelerate, peft, fastapi, uvicorn (pyproject.toml)
```
There is no weight-pulling command — trained checkpoints live in `checkpoints/` (gitignored) on whatever machine trained them; they are not published to Hugging Face yet.

## 2. Train

**Encoder (`ekvachan-base`, ModernBERT-large + classification head, CE+Brier+temperature-scaling — PRD.md Section 5.3):**
```bash
python -m training.train_encoder \
  --base-model answerdotai/ModernBERT-large \
  --data-dir data/processed/nli_slice \
  --output-dir checkpoints/ekvachan-base \
  --epochs 2 --batch-size 96 --grad-accum-steps 1 --eval-batch-size 32 \
  --lr 2e-5 --max-length 256 --brier-lambda 0.5 --seed 42 \
  --train-subset 0 --eval-subset 0 --calib-fraction 0.3
  # add --grad-checkpointing only if you actually OOM; off by default
```
All flags above are optional — every one has the default shown (read the `argparse` block at the top of `training/train_encoder.py` if these drift). `--train-subset`/`--eval-subset` (0 = full set) are the fast-smoke-test knobs — use a small nonzero value to sanity-check a change before committing to a full run. On a shared/CUDA GPU the script refuses to start below 4GB free VRAM and auto-shrinks batch size to fit what's actually free, printing a warning when it does.

**Decoder-LoRA comparison arm (Qwen3.5-4B, restricted-logit read, PRD.md Section 5.1/3.1a):**
```bash
python -m training.train_decoder_lora \
  --base-model Qwen/Qwen3.5-4B \
  --data-dir data/processed/nli_slice \
  --output-dir checkpoints/ekvachan-decoder-qwen \
  --epochs 1 --batch-size 4 --grad-accum-steps 16 --eval-batch-size 8 \
  --lr 1e-4 --max-length 384 --lora-r 16 --lora-alpha 32 --seed 42 \
  --train-subset 0 --eval-subset 0 --calib-fraction 0.3
  # --grad-checkpointing: off by default, costs ~30-40% more compute for memory headroom
```
Same defaults-shown convention. Be aware before running a full pass: per `PRD.md` Section 13a.2, this model's SSM/linear-attention layers fall back to slow, memory-hungry un-fused kernels unless `causal_conv1d`/`flash-linear-attention` are installed — the measured run needed a 60k-example subset (`--train-subset 60000`) to stay tractable on a shared GPU; a naive full-dataset run can extrapolate to ~60+ hours. Check fused-kernel availability before promising a time estimate.

Both scripts write `checkpoints/<output-dir>/manifest.json` on completion — weight hash (encoder only), hyperparameters, `train_size`/`calib_size`/`test_size`, fitted `temperature`, and `raw_report`/`calibrated_report` (each: `n`, `accuracy`, `brier`, `ece`). Use `training/compare_architectures.py` (point it at one or more checkpoint dirs) to print a side-by-side comparison table once both arms have manifests.

## 3. Serve
```bash
uvicorn serve.server:app --host 0.0.0.0 --port 8080
```
This loads the checkpoint at `checkpoints/ekvachan-base-run2` (see `serve/inference.py`'s `load_default()`) on first request. If that path doesn't exist on the machine you're running on, point `load_default()`/`EncoderChoiceModel` at whatever checkpoint dir you actually have, or say plainly that no trained checkpoint is available rather than pretending the server works.

Smoke-test — **note the fixed schema**, this is not an arbitrary-options example:
```bash
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Premise: The cat sat on the mat.\nHypothesis: An animal was on the mat.",
    "questions": {
      "nli": {"type": "choice", "options": ["entailment", "neutral", "contradiction"]}
    }
  }'
```
Expect a JSON response (`results.nli.choice`, `.probabilities`, `.confidence`) in roughly 27–36ms warm (measured, `PRD.md` Section 13a.3 — well above the <15ms target, expected for unoptimized eager-mode Python). `GET /health` is also available.

**Any other option set, or `type: "score"`/`type: "noul"`, returns HTTP 501.** This is the checkpoint's real, current limitation (`PRD.md` Section 10a) — the classification head was never trained on `options` text, so it cannot generalize to an arbitrary option list the way Jev's documented API can. Don't work around this by mapping user options onto the fixed schema; report the 501 as what it is.

## 4. Compatibility check against an existing Jev integration
Point the target codebase's Jev base URL at the local ekVachan server instead of `https://api.typesafe.ai`. Request shape matches Jev's documented contract (`PRD.md` Section 1.2/6.1); response shape (`results`, `usage` field names) is a best-effort reconstruction, never diffed against a real Jev response. For any integration beyond the exact NLI `choice` schema above, expect and report a 501 rather than silently degrading behavior.

## Not yet available — say so, don't improvise
- **Benchmarks** (jabr-v2, ViZDoom, StarCraft, browser-use task, JevBench — `PRD.md` Section 8) and `results/` evidence bundles: no harness code exists in this repo yet.
- **`/v1/finetune`** / one-command fine-tuning on user data (G6): not built; today's path is running `training/train_encoder.py` by hand against a differently-formatted dataset, which is a real gap from the "one command" goal.
- Nano tier, cascade serving, vision tier, Rust/ONNX production server, Python/TS SDKs: all still design-only, per `PRD.md` Section 12/13.

## Guardrails
- Don't claim a benchmark result (Section 8 of `PRD.md`) is true unless you've actually run it in this session and can point to the evidence bundle — and today, that harness code doesn't exist, so no benchmark claim is possible at all yet.
- Don't fabricate CLI output for commands that don't exist (there is no `ekvachan` CLI) — the real entry points are the `python -m training....` invocations and `uvicorn serve.server:app` above.
- Don't quote a training number (accuracy/ECE/Brier) without reading it from an actual `manifest.json` or from `PRD.md` Section 13a — don't estimate or round a number you haven't seen.
- Don't imply the server answers an arbitrary `choice`/`score`/`noul` question — state the fixed-schema limitation every time this comes up.
- Model/license questions: everything here is Apache-2.0, including weights once published (`PRD.md` Section 10).
