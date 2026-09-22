# better-jev-for-all

An open, self-hostable, faster "System One" decision model — an API-compatible, open-weight alternative to [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Given a `state` and a set of typed `questions` (`choice`, `score`, `noul`), this returns calibrated probabilities in one non-autoregressive forward pass instead of routing the decision through a full text-generating LLM — same idea as Jev, but open-weight, self-hostable, no waitlist.

**Status: research/design phase.** See [`PRD.md`](./PRD.md) for the full design doc: what Jev actually is, the (already crowded) open-alternative landscape we're benchmarking against, the architecture decisions, training plan, serving design, benchmark commitments, and compute budget.

Model family working codename: **ekVachan**.

## Why this exists

Jev is closed: hosted API only, no weights, waitlist-gated. Since its launch on 2026-09-15, a wave of open reproductions has already shipped (`Von`, `Rizzo Flow`, `Kev`, and ~30 others). The goal here isn't to be first — it's to be the best one: beat the current open state of the art (`Von`) on its own published benchmarks, add a multimodal tier nobody else has shipped, and make fine-tuning/self-hosting genuinely one command.

## Repo layout

```
PRD.md                  — full design doc (read this first)
skills/ekvachan-setup/  — agent-usable skill: install, serve, benchmark, fine-tune
docs/                   — research notes, benchmark write-ups
```

## License

Apache-2.0 (code and, once published, model weights).
