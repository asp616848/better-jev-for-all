# better-jev-for-all

An open, self-hostable, faster "System One" decision model — an API-compatible, open-weight alternative to [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Given a `state` and a set of typed `questions` (`choice`, `score`, `noul`), this returns calibrated probabilities in one non-autoregressive forward pass instead of routing the decision through a full text-generating LLM — same idea as Jev, but open-weight, self-hostable, no waitlist.

**Status (2026-09-23): Phase 1 is real, and has measured third-party benchmark numbers.**

The **decoder** arm — Qwen3.5-4B + LoRA, read via restricted single-letter logits — is the **primary architecture** as of 2026-09-23 ([`PRD.md`](./PRD.md) Section 14 Q4). It beat the `ekvachan-base` encoder arm decisively on the same held-out 35,486-example split: **92.25% accuracy / 0.0232 raw ECE**, versus the encoder's **86.32% / 0.0344 calibrated ECE**, on **20x less training data** (PRD 13a.1/13a.2). The wide-schema variant (`ekvachan-decoder-qwen-wideschema`, up to 26 options) then generalized **zero-shot to a genuinely unseen 15–26-way schema at 96.10%** (PRD 13a.6) — no easy-subset construction.

That checkpoint was then run against two real, third-party, independently-maintained benchmarks, with **zero training exposure to either**:

| Benchmark | Items attempted | Accuracy | Brier | ECE |
|---|---|---|---|---|
| JevBench (231 public items) | 139 | **83.45%** | 0.2610 | 0.0855 |
| jabr-v2 (944 items, v1+v2) | 387 | **88.89%** | 0.1508 | 0.0397 |

Evidence bundles are committed in `results/` (`jevbench-decoder_multischema-20260923T084856Z.manifest.json`, `jabr_v2-decoder_multischema-20260923T085009Z.manifest.json`). PRD Section 13a is the source of truth for every number quoted here.

**What those numbers are not**, stated up front rather than in a footnote: they are **not complete benchmark scores** (`is_complete_benchmark_score` is `false` in both manifests). Each run only attempted the items this architecture can legitimately answer — `choice` questions with 2–26 options whose gold label is in the option list. The unattempted remainder (**92/231 JevBench, 557/944 jabr-v2**) is *entirely* `score`- and `noul`-type questions, which no ekVachan model is trained for yet — not wide-option `choice` items; the widest real option set in either dataset is 6. So this is **not** a like-for-like comparison against Von's or Jev's published figures, and no claim of beating either is made here. See PRD Sections 8.3 and 13a.7.

**How we compare to the field (2026-09-25, `stage3` checkpoint served via `routing-decoder` with CUDA-graphs).** ekVachan JevBench **70.99%**, jabr-v2 **85.49%**. Third-party numbers below are external, fetched from the respective projects' own READMEs, not run by us — version/methodology may not match exactly (noted inline):

| Benchmark | ekVachan | Von | Jev |
|---|---|---|---|
| JevBench accuracy | 70.99% | 59.3%¹ | — |
| jabr-v2 accuracy | 85.49% | 72.0% (macro, v1.1) | 96.6% (macro) |

¹ Von 1.2's README reports per-difficulty-tier only (easy 100.0%/48, standard 63.9%/72, hard 38.7%/111 — no published overall figure); 59.3% is a case-count-weighted aggregate we derived from those tiers, not Von's own claim. No Jev JevBench number found in either source.

Latency, current reference server (CUDA graphs **on by default** as of PRD 13a.34, serving actual JevBench-shaped items): **~112ms p50 / ~131ms p95** (vs ~131ms p50 / ~156ms p95 with graphs off). Published field numbers from the original task brief: Von <18ms, Rizzo Flow 49–52ms, Laya ~16ms — ekVachan remains behind on this axis; CUDA graphs are a real, accuracy-neutral ~15% win, not a fix for the remaining gap. See PRD 13a.29–13a.34 for the full investigation.

**The server now runs the decoder.** `serve/` (`POST /v1/systemone`) serves `serve.inference.RoutingDecoderModel` by default (PRD 5.2b/14 Q4) — the decoder arm the numbers above come from, not the old fixed-3-way encoder. All three primitives (`choice` 2–588 options, `score`, `noul`) are answered for real, `Question.image` routes to the vision-capable adapter, and the CUDA-graph fast path (PRD 13a.29–13a.34) is **on by default** (`EKVACHAN_USE_CUDA_GRAPHS=0` forces eager, e.g. for a non-CUDA dev box). Request shape follows Jev's documented contract; response shape is a best-effort reconstruction (there is no live Jev API to diff against), so this is "same request shape, best-effort response shape," not verified byte-for-byte compatibility. `STATUS.md` tracks remaining open items.

Model family name: **ekVachan**.

## Why this exists

Jev is closed: hosted API only, no weights, waitlist-gated. Since its launch on 2026-09-15, a wave of open reproductions has already shipped (`Von`, `Rizzo Flow`, `Kev`, and ~30 others). The goal here isn't to be first — it's to be the best one: beat the current open state of the art (`Von`) on its own published benchmarks, add a multimodal tier nobody else has shipped, and make fine-tuning/self-hosting genuinely one command.

## Repo layout

```
PRD.md                  — full design doc (read this first — everything below is the short version)
STATUS.md               — living task list: what's done, what's broken, what's next
training/               — data pipeline + both training arms (encoder, decoder-LoRA, multischema/wideschema variants)
eval/                   — shared accuracy/Brier/ECE metrics used by every training arm
serve/                  — Python reference inference + the /v1/systemone FastAPI server (decoder + CUDA graphs, on by default)
benchmarks/             — real vendored third-party harnesses (jevbench/, jabr_v2/) + shared backends and schema filter
results/                — committed evidence bundles (manifest per run), per PRD Section 8.2
sdk/                    — Python and TypeScript clients for /v1/systemone
skills/ekvachan-setup/  — agent-usable skill: install, serve, benchmark, fine-tune
skills/dev-guidelines/  — operating rules for any agent working on this repo (read first)
docs/                   — research notes, benchmark write-ups
```

`checkpoints/` and `data/` are gitignored — trained weights and processed data live on the training server, not in this repo. Weights aren't published to Hugging Face yet.

For anything longer than this summary — architecture rationale, the full 13a.1–13a.7 run log, roadmap, benchmark commitments — see `PRD.md`, and `STATUS.md` for the current checklist.

## License

Apache-2.0 (code and, once published, model weights).
