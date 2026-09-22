---
name: argus-setup
description: Install, serve, benchmark, and fine-tune Argus (the open System One model in this repo) — an open-weight, self-hostable, API-compatible alternative to TypeSafe's Jev. Use this skill when a user asks to set up, run, test, or customize Argus, or to point an existing Jev integration at a local/self-hosted model instead.
---

# Argus setup skill

Argus is an open-weight "System One" decision model: given a `state` and typed `questions` (`choice` / `score` / `noul`), it returns calibrated probabilities in one forward pass — no text generation. It ships with an HTTP API compatible with TypeSafe's Jev (`POST /v1/systemone`), so anything built against Jev works against a local Argus server by changing a base URL.

Full design rationale, architecture decisions, and benchmark commitments live in [`../../PRD.md`](../../PRD.md) at the repo root — read that before making any implementation decisions on behalf of a user, since it documents *why* choices were made (model sizing, why encoder-over-logit-reading, cascade design, etc.), not just what to run.

> **Status note:** As of this skill's authoring, Phase 1 (the actual `argus-base` model, training code, and server) has not been built yet — this repo is at the PRD/design stage. The commands below describe the target workflow once Phase 1 ships. If `server/`, `training/`, or `eval/` don't exist yet in the repo root, say so plainly and point the user to the Roadmap section of `PRD.md` instead of fabricating a working install.

## When to use this skill
- User wants to install/run Argus locally.
- User wants to swap a Jev-integrated project (e.g. a `browser-use/jev-ultrafast`-style agent) over to a self-hosted model.
- User wants to run the benchmark suite (jabr-v2 reproduction, ViZDoom, StarCraft, browser task — see `PRD.md` Section 8) and get an evidence bundle.
- User wants to fine-tune a custom decision head on their own labeled data.

## 1. Install & serve
```bash
git clone <this-repo-url> && cd better-jev-for-all
# once Phase 1 ships:
uv sync
uv run argus pull --tier base   # downloads the right weight tier for host hardware
uv run argus serve               # starts local server, default http://127.0.0.1:8080
```
Smoke-test:
```bash
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state": "test", "questions": {"ok": {"type": "noul", "instructions": "Is this a test?"}}}'
```
Expect a JSON response with a probability for `ok` in well under 100ms.

## 2. Compatibility check against an existing Jev integration
Point the target codebase's Jev base URL at the local Argus server instead of `https://api.typesafe.ai`. Because the wire contract matches (Section 6.1 of `PRD.md`), most integrations need no other code change. If a request uses a feature Argus doesn't yet support (check `PRD.md` for current primitive/feature coverage), report the specific gap rather than silently degrading behavior.

## 3. Benchmark
```bash
uv run argus bench --suite jabr-v2      # accuracy + calibration (ECE)
uv run argus bench --suite vizdoom      # game benchmark, same seeds as Von's published numbers
uv run argus bench --suite browser-use  # reproduces the jev-ultrafast flight-search task
```
Every run writes a signed evidence bundle to `results/` (raw outputs, seeds, weight hash) — this is required, not optional, per `PRD.md` Section 8.2. Never report a benchmark number to a user without the corresponding evidence bundle existing in `results/`.

## 4. Fine-tune on custom data
```bash
uv run argus finetune --data <path-to-labeled-jsonl> --base-tier base
```
Produces a new head + an automatic eval report (accuracy + ECE on a held-out split). Refuse to skip the eval report step even if the user doesn't ask for it — an unevaluated fine-tune is not a deliverable.

## Guardrails
- Don't claim a benchmark result (Section 8 of `PRD.md`) is true unless you've actually run it in this session and can point to the evidence bundle.
- Don't fabricate install/CLI output if the corresponding code doesn't exist yet in the repo — check first.
- Model/license questions: everything here is Apache-2.0, including weights once published (`PRD.md` Section 10).
