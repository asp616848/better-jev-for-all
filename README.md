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

**The server is one phase behind the model.** `serve/` (`POST /v1/systemone`) still loads the old fixed-3-way **encoder** checkpoint: it returns `501` for any option set other than `["entailment", "neutral", "contradiction"]`, and for `score`/`noul`. The decoder — the chosen architecture, and the one all the numbers above come from — **is not wired into the serving layer yet**. Separately, request shape follows Jev's documented contract but response shape is a best-effort reconstruction (there is no live Jev API to diff against), so this is "same request shape, best-effort response shape," not verified byte-for-byte compatibility. `STATUS.md` tracks this as the top open item.

Model family name: **ekVachan**.

## Why this exists

Jev is closed: hosted API only, no weights, waitlist-gated. Since its launch on 2026-09-15, a wave of open reproductions has already shipped (`Von`, `Rizzo Flow`, `Kev`, and ~30 others). The goal here isn't to be first — it's to be the best one: beat the current open state of the art (`Von`) on its own published benchmarks, add a multimodal tier nobody else has shipped, and make fine-tuning/self-hosting genuinely one command.

## Repo layout

```
PRD.md                  — full design doc (read this first — everything below is the short version)
STATUS.md               — living task list: what's done, what's broken, what's next
training/               — data pipeline + both training arms (encoder, decoder-LoRA, multischema/wideschema variants)
eval/                   — shared accuracy/Brier/ECE metrics used by every training arm
serve/                  — Python reference inference + the /v1/systemone FastAPI server (encoder-only today)
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
