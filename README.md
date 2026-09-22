# better-jev-for-all

An open, self-hostable, faster "System One" decision model — an API-compatible, open-weight alternative to [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Given a `state` and a set of typed `questions` (`choice`, `score`, `noul`), this returns calibrated probabilities in one non-autoregressive forward pass instead of routing the decision through a full text-generating LLM — same idea as Jev, but open-weight, self-hostable, no waitlist.

**Status (2026-09-22): Phase 1 is partially real.** `ekvachan-base` (ModernBERT-large encoder) is trained and evaluated — **86.32% accuracy, 0.0344 calibrated ECE** on a held-out 35,486-example test split, trained on 1,206,855 NLI examples — and served behind a working `POST /v1/systemone` reference server. A second architecture arm (Qwen3.5-4B LoRA decoder) is training now; no numbers yet. See [`PRD.md`](./PRD.md) Section 13a for the full run log — it's the source of truth for any number quoted here.

**The server has one hard limitation, stated plainly rather than oversold**: request shape follows Jev's documented contract, but response shape is a best-effort reconstruction (there's no live Jev API to diff against), and only `choice` is implemented — and only for the exact fixed option set `["entailment", "neutral", "contradiction"]` the checkpoint was trained on. Any other option set, or `score`/`noul`, returns `501` rather than a guess. This is "same request shape, best-effort response shape," not verified byte-for-byte compatibility.

None of the competitive benchmarks (jabr-v2, ViZDoom, StarCraft, JevBench) have been run yet, so no claim of beating Jev or any open alternative is made here — see `PRD.md` Section 8/13.

Model family name: **ekVachan**.

## Why this exists

Jev is closed: hosted API only, no weights, waitlist-gated. Since its launch on 2026-09-15, a wave of open reproductions has already shipped (`Von`, `Rizzo Flow`, `Kev`, and ~30 others). The goal here isn't to be first — it's to be the best one: beat the current open state of the art (`Von`) on its own published benchmarks, add a multimodal tier nobody else has shipped, and make fine-tuning/self-hosting genuinely one command.

## Repo layout

```
PRD.md                  — full design doc (read this first — everything below is the short version)
training/               — data pipeline + both training arms (encoder, decoder-LoRA)
eval/                   — shared accuracy/Brier/ECE metrics used by both training arms
serve/                  — Python reference inference + the /v1/systemone FastAPI server
skills/ekvachan-setup/  — agent-usable skill: install, serve, benchmark, fine-tune
docs/                   — research notes, benchmark write-ups
```

`checkpoints/` and `data/` are gitignored — trained weights and processed data live on the training server, not in this repo. Weights aren't published to Hugging Face yet.

For anything longer than this summary — architecture rationale, the full 13a.1–13a.4 run log, roadmap, benchmark commitments — see `PRD.md`.

## License

Apache-2.0 (code and, once published, model weights).
