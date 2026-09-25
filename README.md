# ekVachan

**An open, self-hostable "System One" decision model — an API-compatible alternative to [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), with real third-party benchmark numbers to back it.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)
[![Docs: PRD.md](https://img.shields.io/badge/docs-PRD.md-informational)](./PRD.md)
[![Status: STATUS.md](https://img.shields.io/badge/status-STATUS.md-lightgrey)](./STATUS.md)

Give it a `state` and a set of typed `questions` (`choice`, `score`, `noul`), and it returns calibrated probabilities in **one non-autoregressive forward pass** — no chain-of-thought, no text generation, no token-by-token decoding. Same shape of problem Jev solves, same request contract, but open-weight, self-hostable, and with every number below backed by a committed, re-runnable evidence file, not a marketing claim.

```bash
curl -s http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The customer says the package never arrived.",
    "questions": {
      "resolution": {"type": "choice", "options": ["refund", "replace", "escalate"]}
    }
  }'
# -> {"results": {"resolution": {"choice": "refund", "probabilities": {...}, "confidence": 0.81}}, ...}
```

## Why this exists

Jev is closed: hosted API only, no weights, waitlist-gated. Since its launch, a wave of open reproductions has shipped (`Von`, `Rizzo Flow`, `Kev`, and ~30 others). The goal here isn't to be first — it's to be the best one: **beat the current open state of the art on its own published benchmarks**, ship a multimodal tier, and make self-hosting genuinely one command, all while keeping every claim traceable to a committed run.

## Real numbers, full benchmark coverage

Run against two real, independently-maintained, third-party benchmarks — **every item, not a convenient subset** (`is_complete_benchmark_score: true` in both manifests) — through the actual production serving class (`serve.inference.RoutingDecoderModel`, CUDA graphs on):

| Benchmark | Items | Accuracy | Brier | ECE |
|---|---|---|---|---|
| [JevBench](https://github.com/fstandhartinger/jevbench) | 231/231 | **70.99%** | 0.393 | 0.115 |
| [jabr-v2](https://github.com/jabr/classifier-benchmark) | 944/944 | **85.49%** | 0.208 | 0.033 |

**vs. the field** — third-party numbers below are external, fetched from the respective projects' own READMEs, not run by us (version/methodology may not match exactly, noted inline):

| Benchmark | ekVachan | Von | Jev |
|---|---|---|---|
| JevBench accuracy | **70.99%** | 59.3%¹ | — |
| jabr-v2 accuracy | **85.49%** | 72.0% (macro, v1.1) | 96.6% (macro) |

¹ Von 1.2's README reports per-difficulty-tier only (easy 100.0%/48, standard 63.9%/72, hard 38.7%/111 — no published overall figure); 59.3% is a case-count-weighted aggregate we derived from those tiers, not Von's own claim. No Jev JevBench number found in either source.

**Latency**, current reference server (CUDA graphs on by default, PRD 13a.34): **~112ms p50 / ~131ms p95** on realistic (100–250 token) request shapes, ~45ms on short (~15-token) prompts — latency scales with the CUDA-graph bucket a request falls into (128–8192 tokens, 7 buckets), not a flat number. Published field numbers: Von <18ms, Rizzo Flow 49–52ms, Laya ~16ms. **We're honest that we're behind here** — CUDA graphs are a real, accuracy-neutral ~15% win over eager mode, not a fix for the remaining gap. See PRD 13a.23–13a.34 for the full, occasionally unflattering, investigation (int8 quantization: 4.7x slower, rejected; vLLM: architecturally right, still slower on this workload, rejected; `torch.compile`: blocked/negative, rejected).

Every number above is backed by a manifest in `results/` you can re-run yourself — see PRD Section 13a for the full run log, including the negative results we didn't hide.

<details>
<summary>Earlier Phase 1 numbers (2026-09-23, superseded above — kept for history, not for citing)</summary>

The very first third-party benchmark run, before `score`/`noul` training existed, only attempted the subset of items this architecture could answer at the time (`choice` questions with 2–26 options): **JevBench 83.45%** (139/231 items attempted), **jabr-v2 88.89%** (387/944 items attempted), both explicitly `is_complete_benchmark_score: false`. This is a different, incomparable metric (accuracy-over-attempted-subset, not accuracy-over-full-dataset) from the full-coverage numbers above — kept here only so the historical record in `results/` stays legible, not because it's a number worth quoting today.
</details>

## Architecture, in short

- **Decoder arm** — Qwen3.5-4B + LoRA, read via a restricted single-letter-logit trick: build a prompt with a short code table (A, B, C, ...), take the last token's logits, restrict to just the relevant code ids, softmax. One forward pass, no generation. Beat a from-scratch encoder classifier decisively (92.25% vs 86.32% accuracy) on **20x less training data** (PRD 13a.1/13a.2).
- **Wide-option support** — up to **588 options** in one request, via a measured 588-code single-token table under Qwen3.5-4B's tokenizer (PRD 5.1b) — no architecture change needed to go past the naive 26-letter ceiling.
- **Multi-adapter routing** — one base model, hot-swapped named LoRA adapters (text vs. vision-capable), each validated against its own manifest's option-count cap, not a shared global ceiling (PRD 13a.20/13a.22).
- **Multimodal** — image-bearing requests route to the vision-capable adapter automatically; the restricted-logit read works unchanged whether the forward pass came from text or a vision-language model (PRD 5.2b).
- **CUDA graphs, on by default** — manual capture/replay of the forward pass against dedicated scratch tensors (never the model's live parameters, so a graph failure can never corrupt the eager fallback), fail-closed on any error. Real, accuracy-neutral ~15% latency win (PRD 13a.29–13a.34).

## Quickstart

```bash
git clone https://github.com/asp616848/better-jev-for-all && cd better-jev-for-all
uv sync   # or: pip install -e .

# checkpoints/ is gitignored -- point --checkpoints-dir at wherever your
# trained adapters live, or set EKVACHAN_TEXT_ADAPTER + place them under
# ./checkpoints/ekvachan-decoder-qwen-{benchcorpus,vision}/ (see serve/inference.py)
EKVACHAN_TEXT_ADAPTER=vision uv run uvicorn serve.server:app --port 8000
```

Then `POST /v1/systemone` with the same request shape as the curl example above. `Question.type` can be `choice` (2–588 options), `score` (ordinal levels), or `noul` (yes/no, returns a bare probability). See `sdk/` for Python and TypeScript clients, `skills/ekvachan-setup/SKILL.md` for the full train/serve/benchmark/fine-tune walkthrough.

**Hugging Face weights + a try-it-in-browser demo Space are in progress** — not published yet; this repo's `hf_model/`/`hf_space/` hold the (unexecuted, GPU-pending) scaffolding. Until then, self-hosting via the quickstart above is the way to run it.

## Honest gaps

- **Latency**: behind the fastest field entries (see table above) — this is the open, tracked next lever, not a hidden weakness.
- **Response-shape verification**: request shape follows Jev's documented contract; response shape (`results`, `usage` field names) is a best-effort reconstruction — there's no live Jev API to diff against byte-for-byte, so treat it as "same request shape, best-effort response shape," not a verified wire-compatible contract.
- **`score`/`noul` accuracy**: real but weaker than `choice` (PRD 13a.8) — `score`'s errors are overwhelmingly adjacent-level confusions (genuine task difficulty at class boundaries), not a broken mechanism, but it's the honest weak point in the primitive set.

`STATUS.md` is the living, continuously-updated checklist of what's done, what's in progress, and what's not started — read that (not this README) for the current bleeding edge.

## Repo layout

```
PRD.md                  — full design doc (read this first — everything above is the short version)
STATUS.md               — living task list: what's done, what's broken, what's next
training/               — data pipeline + both training arms (encoder, decoder-LoRA, multischema/wideschema variants)
eval/                   — shared accuracy/Brier/ECE metrics used by every training arm
serve/                  — Python reference inference + the /v1/systemone FastAPI server (decoder + CUDA graphs, on by default)
benchmarks/             — real vendored third-party harnesses (jevbench/, jabr_v2/) + shared backends and schema filter
results/                — committed evidence bundles (manifest per run), per PRD Section 8.2
sdk/                    — Python and TypeScript clients for /v1/systemone
hf_model/, hf_space/    — Hugging Face model card + ZeroGPU demo Space (scaffolded, not yet published)
skills/ekvachan-setup/  — agent-usable skill: install, serve, benchmark, fine-tune
skills/dev-guidelines/  — operating rules for any agent working on this repo (read first)
docs/                   — research notes, benchmark write-ups
```

`checkpoints/` and `data/` are gitignored — trained weights and processed data live on the training server, not in this repo (yet — see the Hugging Face note above).

## License

Apache-2.0 (code and, once published, model weights).
