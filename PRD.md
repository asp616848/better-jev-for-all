# PRD: ekVachan — an open, self-hostable, faster System One model

Status: draft v0.4 · Owner: Niyati Singh · Date: 2026-09-22
Model family name: **ekVachan** (Hindi एक वचन, "one word"/"one utterance" — fits a model that returns one committed, typed answer per question instead of generating a stream of text).

*v0.2 note: two independent research passes ran on this task in parallel and were merged here. This document's own research (Sections 1–14 as originally drafted) found `Von`, the architecture-reverse-engineering writeup, and the jabr-v2/ViZDoom benchmark trail. A second pass added the sections marked "(addendum, merged from a parallel research pass)" — JevBench (a second standardized leaderboard, Section 3.1a), conformal prediction as a calibration upgrade (5.3a), a concrete DAgger data-collection method for the game slice (5.3b), and two benchmark-design lessons (8.1a, 8.1b). Where the two passes disagreed — this PRD's Section 5.1 bets on encoder+heads while JevBench's leaderboard is currently dominated by small decoder LLMs — that tension is flagged, not silently resolved (end of 3.1a).*

*v0.3 note: renamed Argus → ekVachan per project owner's decision; Section 7.4 license check completed (all three harness repos confirmed MIT); Section 9 rewritten for self-hosted compute (project owner has their own Linux server, not a rented GPU marketplace); Section 14 open questions resolved and closed out. Both v0.2 and v0.3 passes were done by an AI agent working on this repo — none of the above should be read as independently peer-reviewed; every factual claim still traces to the cited primary source, and that's the standard to hold any future edit to as well.*

*v0.4 note: **the first version of this document written after real training actually ran**, rather than before it. Phase 1 work happened on the owner's own server: `ekvachan-base` (encoder) is trained, evaluated and served behind a working `/v1/systemone`; the decoder comparison arm is mid-run. Section 13a is the new dated run log carrying those measured numbers; Sections 12 (repo tree), 13 (roadmap), 6.1 and 10a are corrected to match what the code actually does today, and Section 14 Q4 gains a cost axis the decoder run surfaced. Everything above v0.4 is still the pre-training plan — where a measured result now contradicts or narrows a planned one, 13a says so explicitly rather than quietly rewriting the plan.*

---

## 0. TL;DR

Jev (TypeSafe AI, launched 2026-09-15) is a closed, hosted "System One" model: given a `state` and a set of typed `questions` (`choice` / `score` / `noul`), it returns calibrated probabilities in one non-autoregressive forward pass instead of generating text. It's fast (70–500ms, ~100ms typical) and cheap ($0.042/M input tokens, output free) compared to routing the same decision through a frontier chat LLM, and TypeSafe has shown it wired into browser agents (browser-use), StarCraft, a stealth game (HEIST//ONE), Minecraft, a robot arm, and voice/hand-tracking computer control.

**Jev is not open. No weights, no self-hosting, waitlist-gated API.** That gap is real and worth closing.

**It is also not novel anymore.** In the week since launch, ~30+ open reproductions and alternatives have shipped (`Von`, `Rizzo Flow`, `Kev`, `Laya`, `open-alternative-jev`, `NanoJev`, etc.), several of which already beat Jev on latency (trivial — local beats a multi-tenant cloud API) and at least one (`Von`) beats Jev outright on published game-decision benchmarks (ViZDoom `Defend the Center`: 9.00 vs 5.62 kills) using nothing more exotic than a 395M-parameter encoder with a classification head.

So the honest framing for this project is **not** "be first to open-source a faster Jev" — that's already done, multiple times, by small teams over a weekend. The bar is: **build the best one**, prove it on the same public, reproducible evidence trail this ecosystem already uses (frozen benchmarks + game harnesses + signed evidence traces), and make it radically easier to adopt and extend than anything that exists today. Section 3 lays out exactly what "best" has to beat, with numbers.

Everything the user asked for is achievable. Nothing below requires a capability that doesn't exist. The one thing to be upfront about: **training a genuinely better model is an empirical claim you earn by running the benchmarks, not something a PRD can promise in advance.** This document sets concrete, falsifiable targets (Section 8) instead of guaranteeing "we will beat X."

**Where the project actually stands (2026-09-22)**: one model is really trained (`ekvachan-base`: 86.32% accuracy, 0.0344 calibrated ECE on our own held-out NLI test split) and really served behind `/v1/systemone`, the decoder comparison arm is mid-run, and **none of Section 8's benchmarks have been run yet** — so every competitive claim in this document is still a target. Section 13a is the measured record; read it before quoting any number from Sections 5–9 as if it were a result.

---

## 1. What Jev actually is (researched, not guessed)

Sourced from TypeSafe's own launch post, third-party technical coverage, and an independent black-box reverse-engineering writeup. Marketing claims are labeled as such.

### 1.1 Company / model
- **TypeSafe AI**, founded by Diogo Almeida (OpenAI alum, InstructGPT/RLHF background). Emerged from stealth 2026-09-15 with $40M seed led by DCVC. ([typesafe.ai](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [nextbigfuture](https://www.nextbigfuture.com/2026/09/typesafe-ai-jev-is-a-transformational-unlock-for-ai-decisions-classifications-and-huge-speed-and-cost-savings-for-ai-software.html))
- Jev is TypeSafe's first "System One Model" — a new model *class*, not a smaller LLM. Explicitly: "transformer-based, but not a large language model," cannot generate free text. ([jevai.net](https://jevai.net/articles/what-is-system-one-jev/))
- **Non-autoregressive**: one forward pass evaluates every question in parallel against shared state and returns direct probability readouts — no decode loop. This is *why* latency is flat regardless of how many questions you ask or how long the "answer" conceptually is. ([latent.space](https://www.latent.space/p/ainews-jev-a-system-one-model-that), [typesafe.ai](https://typesafe.ai/blog/introducing-system-one-models-and-jev))
- **Training**: "Reinforcement Learning for Calibrated Decisions" (RLCD) — TypeSafe's term, not RLHF. Optimizes for probabilities matching real-world outcome frequency (a stated 90% should be right ~90% of the time), not human preference ranking. Loss function itself is **not published**. ([typesafe.ai](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [flaviocopes](https://flaviocopes.com/jev/))
- **Architecture internals are not disclosed.** An independent black-box study (10,000 API calls, timing/token-accounting/option-reordering/information-isolation probes) *estimates*, at medium-low confidence: causal transformer, sparse MoE, ~10B active params, shared KV-cache across questions in one request, question branches can't see each other's answers, adding an irrelevant option measurably shifts other options' odds (log-odds shift −0.28) implying listwise (not independent-logit) scoring. Base model / tokenizer lineage unidentified. **Treat all of this as a hypothesis, not fact.** ([archerhume.com](https://archerhume.com/posts/jevs-architecture-unmasked/))
- **No open weights, no self-hosting, no fine-tuning API.** Hosted API only, early access / waitlist. ([jevai.net gist](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965))

### 1.2 API contract (this is what we copy for compatibility — see Section 6)
Single endpoint: `POST https://api.typesafe.ai/v1/systemone`

```json
{
  "state": "free text or structured context",
  "model": "jev-<version>",
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "...",
      "options": ["billing", "technical", "other"]
    },
    "urgency": {
      "type": "score",
      "instructions": "...",
      "levels": ["low", "medium", "high", "critical"]
    },
    "is_spam": {
      "type": "noul",
      "instructions": "..."
    }
  }
}
```

Three primitives, always:
| Primitive | Meaning | Output |
|---|---|---|
| `choice` | pick 1 of up to 255 options | `{choice, probabilities{}, confidence}` |
| `score` | place on an ordered 2–10 level scale | `{score, probabilities{}, confidence}` (can land between levels — probability-weighted) |
| `noul` | yes/no proposition | `float` 0–1 |

Questions in one request are **independent** — they share `state` but not each other's answers. Response also carries `model` (versioned, for reproducibility) and `usage`. ([lilting.ch](https://lilting.ch/en/articles/typesafe-ai-jev-system-one-model), [datacamp](https://www.datacamp.com/blog/system-one-models-jev), [flaviocopes](https://flaviocopes.com/jev/))

Explicit non-goals of the primitive design, per TypeSafe's own docs: no long-form generation, no math derivation, no open-ended exploration, no chain-of-thought. "Code calculates. Jev judges. Reasoning models reason and generate." ([gist](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965))

### 1.3 Performance (TypeSafe's numbers — marketing, not independently reproduced by TypeSafe itself)
- Latency: 70–500ms end-to-end, most calls ~100ms from US-West. ([flaviocopes](https://flaviocopes.com/jev/))
- One head-to-head workflow eval: Jev 114ms / $0.000081 per call vs "GPT-5.6 Terra" 8.566s / $0.013880 → **193.6× faster, 444.6× cheaper**. TypeSafe's own capabilities team wrote the eval workflows and explicitly states it "cannot prove the price is unsubsidized" and the gains "sit at the high end of real use." ([datacamp](https://www.datacamp.com/blog/system-one-models-jev))
- Pricing: $0.042 / million input tokens, output free ("too cheap to meter"). Throughput ceiling in early access: 250k tok/s, 1,200 req/min. ([flaviocopes](https://flaviocopes.com/jev/))
- Independent(ish) calibration check by LangChain: 500/500 agreement with human oracle on binary judgments, 433× lower variance than a comparable LLM-judge — this one reads as more credible since it's a third party, but it's still self-reported by a partner, not a neutral academic eval. ([langchain blog](https://www.langchain.com/blog/building-a-harness-with-jev))
- TypeSafe is explicit that "zero hallucination" means *schema* compliance only — it can still be confidently wrong. ([datacamp](https://www.datacamp.com/blog/system-one-models-jev))

### 1.4 The demos the user referenced
- **browser-use/jev-ultrafast** (real repo, `browser-use/jev-ultrafast`, MIT-style, 16.5k★): Jev picks the *operation* (CLICK/TYPE_TEXT/SELECT/SCROLL/WAIT/DONE/BLOCKED) and *target DOM element* from a structured, indexed element table built off an atomic DOM snapshot — no screenshots, no vision, no coordinate math. A small LLM (`inception/mercury-2.5` via OpenRouter) is invoked *only* to fill in free-text when the op is `TYPE_TEXT`. Result: Zurich→London Google Flights search in 7.07s median (down from 9.45s baseline, −25%), CDP protocol calls cut 1,092→101. This is the core trick worth stealing: **collapse "reason about the page" into "classify against a bounded, pre-enumerated action space," and only pay LLM-generation cost for the rare free-text step.** ([repo](https://github.com/browser-use/jev-ultrafast), [explainx.ai](https://www.explainx.ai/blog/jev-ultrafast-browser-use-typesafe-2026))
- **StarCraft** — community repo `phyous/tsai-sc` runs Jev through the original shareware's first campaign mission ("Strongarm") via keyboard/mouse, with recorded action probabilities and an evidence-verified victory screen. Community-built, not an official TypeSafe demo. ([github](https://github.com/phyous/tsai-sc))
- **HEIST//ONE** — an observable browser stealth game where Jev gives batched typed judgments for 6 guards; deterministic code owns the simulation and validates every Jev proposal (Jev never controls state directly). Ships a "Decision Lens," offline scripted mode, evidence traces, tests. Community-built. (from search results, title/description only — verify repo directly before relying on internals)
- **Minecraft**: Jev *solo* falls into repetitive loops (built six crafting tables, no coherent base) — it's described as strong at instant decisions, weak at "what happened five steps ago" / planning ahead. Paired with a reasoning model ("Astra") doing high-level strategy while Jev executes combat/navigation/smelting, results are much better. **This is an important, non-marketing data point**: System One models are a *component*, not a standalone agent, for anything requiring multi-step memory or planning. ([mindstudio.ai](https://www.mindstudio.ai/blog/jev-computer-use-minecraft-robotics-demos))
- **Robot arm**: Jev picks control actions over pre-processed geometry/contact data; picked up a cube, placed it in a matching hole. Explicitly not claimed to generalize to messier real-world scenes. ([mindstudio.ai](https://www.mindstudio.ai/blog/jev-computer-use-minecraft-robotics-demos))
- **Voice + hand-tracking computer control**: works, but the article is blunt that "the harness is the real engineering work" — Jev just picks from a pre-bounded action set the harness constructs. ([mindstudio.ai](https://www.mindstudio.ai/blog/jev-computer-use-minecraft-robotics-demos))

**Takeaway for our design**: in every real deployment, the hard engineering is the *harness* that turns a messy environment (a webpage, a game frame, a robot's sensors) into a bounded, well-labeled decision space Jev can classify against. The model itself is comparatively simple (small, non-autoregressive, classification-shaped). This directly shapes Section 4 and 7.

---

## 2. Feasibility verdict on every ask in the brief

| Ask | Verdict | Notes |
|---|---|---|
| Open-source alternative to Jev's system-one model | ✅ Fully possible | Already has ecosystem precedent (Section 3); we're not inventing the category |
| Faster than Jev | ✅ Already trivially true for any local model (no network hop, no multi-tenant queue) | Real target is being faster/better than the *other open alternatives*, not just Jev |
| Fine-tuning / architectural changes / inference optimization / "lower level things" | ✅ Standard, achievable ML engineering at this model scale (hundreds of millions to low billions of params) | No frontier-scale compute needed — see Section 9 |
| Prove it on fast-paced games + computer use, à la jev-ultrafast | ✅ Achievable — public harnesses already exist (ViZDoom via Von's methodology, StarCraft via `tsai-sc`, browser via `jev-ultrafast`) we can point our model at, or fork | Must credit/license-check any harness we reuse (see 7.4) |
| Build "much better" than `rizzo-flow` and the wider field | ⚠️ Achievable, not guaranteed in advance | `rizzo-flow`'s own authors admit no superiority over baseline; the real bar is `Von` (SOTA-open right now). Section 3 gives exact numbers to beat. This is an empirical outcome of training + eval, not a design decision |
| Copy Jev's API design for ease of adoption | ✅ Legally fine and already common practice (`Von`, `Rizzo Flow`, `jev-local` all do this) | We're reproducing a thin JSON interface for interop, not TypeSafe's code, weights, or training method — same posture Rizzo Flow explicitly states |
| Source heavy GPU compute if worth it | ✅ Not needed — this problem is encoder-scale (0.1B–9B params), not frontier-LLM-scale, and the project owner's own Linux server covers every workload in the plan. See Section 9 | |
| Ship an agent-usable setup/access skill in its own folder | ✅ Straightforward — scaffolded in this repo at `skills/ekvachan-setup/` | |
| "Prove it's better than Jev" as the main focus | ⚠️ Achievable on specific, named benchmarks; **not** a claim we can make true by writing a PRD | Section 8 defines exactly what "prove" means and the concrete numbers we're accountable to |

**Nothing here is impossible.** The only adjustment from the brief as stated: "better than jev" needs to be read as "better than jev **and** better than the current open-source state of the art (Von)," because the open SOTA has already cleared "beats Jev" on at least one headline benchmark. Aiming only at Jev would ship something already obsolete on day one.

---

## 3. Competitive landscape (as of 2026-09-22 — this space is < 1 week old and moving fast)

### 3.1 The open alternatives that already exist
33+ projects catalogued at [systemonemodels.org/examples/alternatives](https://systemonemodels.org/examples/alternatives/). The ones that matter:

| Project | Approach | Base model | License | Key numbers | Gaps |
|---|---|---|---|---|---|
| **Von** (`wfzyx/von`) | Fine-tuned bidirectional encoder + classification heads | ModernBERT-Large, 395M, pretrained on 2T tokens | Apache-2.0, weights on HF (`wfzyx/von-1.0`) | jabr-v2 (49-task, 869-case OOD benchmark): **72.0% macro-acc**. ViZDoom *Defend the Center*: **9.00 kills** vs Jev's 5.62 (+60%). ViZDoom *Health Gathering*: **12.11s** vs Jev's **13.03s** — Jev is actually slightly ahead here, worth being precise about. Latency: **sub-18ms** local. | Degrades without explicit rubric text in `instructions`; no multimodal input; single dense model, no cascade; training corpus (~290k examples) is text-only |
| **Rizzo Flow** | LLM logit-reading (prefill once, branch questions off shared KV-cache, read answer-letter logits A–Z, softmax) | Spark-X2.5 4B / 1.7B, Apache-2.0 | Apache-2.0 | 49–52ms p50/p95 (RTX 5060 Ti). 81.2–84.8% on own fixtures — **authors explicitly claim no superiority over their own "SemIf" baseline**. Jev-compatible `/v1/systemone` endpoint. One-command install (`uv run rizzo serve`, auto hardware backend detect, prebuilt sha256-verified binaries) — genuinely good DX, worth matching regardless of which architecture we ship. | Max 26 options (Jev/Von support 255). **Uncalibrated probabilities by default** — needs a separate calibration pass. Confidently wrong on 6/36 missing-evidence cases (SemIf: 1/36). Residual position bias despite shuffling. Single-platform tested (Windows/CUDA only) — macOS/Metal and Linux/AMD explicitly untested. No rate limiting, single resident model, requests serialize. KV-cache footprint ~144 KiB/token, ~4× worse than their own prior MLX runtime. Not on the JevBench leaderboard (self-reported fixtures only). |
| **Kev** (`jaredpalmer/kev`) | Same encoder-classification pattern | Qwen3.5-based, 0.8B/4B/9B | Apache-2.0 | ~160ms for 6-question batch on Apple Silicon; trains in ~1h45m on Apple Silicon | Smaller/slower than Von at comparable accuracy tier; less benchmark disclosure |
| **open-alternative-jev** (`ikermoel`) | Logit-reading over *any* open-weights LLM via HF or vLLM (needs single-token option labels + ChatML) | Tested Qwen 0.5B–27B | Apache-2.0 | Qwen3.6-27B-8bit: 73.7% acc / ECE 0.020 / 582ms/case on a community benchmark — beats a reported Jev-1.13.0 number (72.7% acc, but ECE only 0.020 vs Jev's much worse calibration, KL 0.27 vs 1.44) **on that specific benchmark only**. Packed mode: 2.5× cheaper but 6–9% answer drift below 4B params. | 582ms is *slower* than both Jev and Von — this approach trades latency for "works on anything you already run." Not competitive on speed. |
| **Laya** | Fine-tuned encoder, calibration-focused | ModernBERT-large, 421M | — | Only alt publishing ECE directly: 0.081 post temp-scaling. ~16ms. 2.2k★ | Smaller ecosystem, less benchmark breadth than Von |
| **NanoJev** | Trained from scratch | 0.6B custom | — | Ships full training pipeline + dataset (1.2k★) — most reproducible of the field | Smaller model, weaker headline numbers, but most useful as a *training reference* |
| Wrappers (openjev-sglang, Decider, litjev) | Same logit-reading idea on Qwen3.5 | Qwen3.5 2B–35B | — | "Not calibrated estimates of correctness" per landscape page | Interface-only reproductions, not real competitors on quality |

General-purpose typed-output tooling people compare against (not System-One-specific, but adjacent): **DSPy** (38k★), **Outlines** (16k★), **Instructor** (14k★) — these do grammar-constrained/validated structured output over any LLM, not calibrated probability readouts. Different tool, same "avoid free-text parsing" instinct.

### 3.1a Addendum: JevBench — a second, independently-maintained standardized leaderboard (merged from a parallel research pass)

Separately from the landscape catalogue above, there's a third-party, MIT-licensed benchmark harness — **JevBench** ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)) — that's already become the field's shared yardstick: 534 frozen decisions, four axes (**Intelligence**, **Calibration**, **Speed**, **Cost**), combined by **geometric mean** at 25% each, with an explicit anti-gaming ruleset (held-out items committed before any run, no schema repair, no retries, label-mapping frozen before execution). Worth running *in addition to* jabr-v2/ViZDoom/StarCraft (Section 8), not instead of — it's a different, complementary evidence trail with its own credibility (independent maintainer, public methodology, other entrants already on it).

Current top-10 (v1.2–1.3):

| Rank | System | Score | Intel | Calib | Speed | Cost |
|---|---|---|---|---|---|---|
| 1 | Jev 1.13.0 (hosted) | 74.4 | 86 | 83 | 83 | 52 |
| 2 | **SemIf**, Qwen3.5-4B ([TheoLeeCJ](https://github.com/TheoLeeCJ/SemIf)) | 73.1 | 79 | 73 | 84 | 60 |
| 3 | **djev**, DiffusionGemma ([mmastrac](https://github.com/mmastrac/djev)) | 73.0 | 83 | 65 | 91 | 58 |
| 4 | Winnow-12B Q8 | 71.2 | 82 | 72 | 82 | 53 |
| 5 | **reflex-4B**, Qwen3.5 ([kshetrajna12](https://github.com/kshetrajna12/reflex)) | 70.3 | 80 | 75 | 68 | 60 |
| 11 | OpenJev (razorback16) | 66.4 | 79 | 65 | 83 | 46 |
| 19/27 | kev 0.6B / 4B | 62.5 / 59.7 | 52/65 | 51/42 | 76/76 | 76/62 |
| 33 | **Laya**, ModernBERT-large (421M encoder) | 54.4 | 46 | 62 | 71 | 86 |

`Von` was not visible in JevBench's top 10 in this pass — I did not confirm whether it's ranked lower or simply not submitted; don't treat that as evidence either way without checking `jevbench`'s full results directly.

**A real tension worth flagging rather than resolving silently**: on *this* benchmark, the architecture family that dominates the top 5 is exactly the one Section 5.1 rejects — small **decoder** LLMs (Qwen3.5-4B class) read via restricted-logit scoring, not fine-tuned bidirectional encoders. The one pure-encoder entrant on the board, Laya (ModernBERT-large, the same weight class Section 5.2 sizes `ekvachan-base` at), lands at rank 33 with an Intelligence score of 46 — *below* JevBench's own chance-adjusted floor of 50, which triggers a squared penalty (`total × (Intelligence/50)²`) on top of an already-weak raw score. That's a structural, not incidental, result: JevBench's Intelligence axis spans harder/judge-tier reasoning-shaped decisions where a small bidirectional classifier without a language-model backbone appears to lose real ground, independent of Von's own (different) benchmark showing an encoder winning on ViZDoom.

This isn't a reason to reverse Section 5.1's decision on the strength of one leaderboard — Von's numbers are real too, and jabr-v2/ViZDoom test a different, more game/action-shaped distribution than JevBench's text-decision-heavy set. But it is a reason to **treat 5.1 as a hypothesis the Phase 1 benchmark run needs to actually stress-test on JevBench specifically, not just on jabr-v2** — if `ekvachan-base` lands near Laya's Intelligence score on JevBench's harder tiers, that's a real signal the encoder bet needs a language-model-backboned variant (closer to the reflex/SemIf recipe: LoRA-tune a small Qwen3.5, read restricted next-token logits, skip building classification heads from scratch) for the tiers where reasoning-shaped judgment matters, even if the pure encoder stays the latency-optimal choice for simple/fast game-tick decisions. Recommendation: **run both families on JevBench in Phase 1 before locking 5.1 in as final** — the cost of testing this is a few LoRA-training-hours (Section 9), trivial next to the cost of discovering it post-launch.

If the decoder family does turn out to matter for a tier, note for Section 7: that family's dominant technique (encode `state` once into a KV cache, branch each question off a restricted-token read) is a close structural match for **SGLang's RadixAttention** (automatic prefix-cache sharing across requests with a common prefix) — a serving-engine option worth having on the shortlist alongside the Rust/ONNX/TensorRT stack Section 7.1 already commits to for the encoder tiers, if/when a decoder-backboned tier gets added.

### 3.2 What this means for us
1. **The floor is already high.** A weekend project (`Von`) beats Jev's own headline game benchmark. Don't scope "beat Jev" as the finish line.
2. **Nobody has shipped multimodal input.** Every alternative above is text-only. Jev itself hints at "possibly image inputs... yet…" in its launch post but hasn't shipped it. For the computer-use / game-playing use case specifically (DOM state *and* a frame, a robot's depth map, a game's minimap), this is a real, currently-empty gap.
3. **Nobody ships a cascade/two-tier architecture as a product.** TypeSafe's own docs mention "confidence-gated branching" as a *usage pattern* you build yourself; nobody bakes a nano-tier (sub-5ms, handles the easy 80%) + escalation-to-base-tier (handles the hard, low-confidence 20%) into the *model serving layer itself*. This is a serving-architecture win, not a training win — cheaper to build than a bigger model, and directly reduces mean latency below Von's 18ms without touching accuracy on easy cases.
4. **Fine-tuning is undersold everywhere.** Kev and NanoJev ship training code; none ship a guided, one-command "point this at your own labeled taxonomy and get a calibrated head back" pipeline. Given the user explicitly wants "faster... by fine-tuning," this is where real, defensible differentiation lives — not in yet another slightly-larger encoder.
5. **`rizzo-flow` specifically is not a high bar.** Its own README disclaims superiority over its own baseline. Beating it is close to a given; it's not the benchmark that matters, `Von` is.

---

## 4. Goals and non-goals

### 4.0 Who this is actually for

Stated explicitly once, rather than left implicit across the benchmark sections: this targets (1) **agent-framework builders** who need a fast, cheap routing/decision layer inside an agent loop instead of paying full LLM-generation cost for a bounded choice (the `browser-use/jev-ultrafast` pattern — Section 1.4); (2) **game-AI / computer-use developers** building the kind of bounded-action-space harnesses Section 1.4 describes (guard judgments, unit orders, DOM element selection); (3) **triage/moderation/support-routing teams** who currently call a full chat LLM just to classify or score something, and want the cost/latency win without giving up self-hosting or calibration guarantees. If your use case is open-ended generation, summarization, or reasoning, this isn't the tool (Section 4.2) — stay on your existing LLM.

### 4.1 Goals
- G1 — **API-compatible** with Jev's `/v1/systemone` contract (drop-in swap by changing a base URL), plus a richer native API.
- G2 — **Self-hostable, open-weight**, no waitlist, no vendor lock-in. Apache-2.0 throughout (code + weights), matching ecosystem norm and maximizing adoption.
- G3 — Beat **Von's published numbers** (jabr-v2 macro-acc, ViZDoom kills/survival, latency) on our own reproduction of the same benchmarks, with signed evidence artifacts. If we don't beat Von on a given axis, say so in the README — no oversell.
- G4 — Ship a **multimodal (vision-capable) tier** for computer-use/game-state decisions — the one gap nobody else has filled.
- G5 — Ship a **cascade serving architecture** (nano → base escalation) that beats Von's flat sub-18ms on *average* latency across a realistic confidence distribution, without regressing hard-case accuracy.
- G6 — Ship a **one-command fine-tuning pipeline**: bring your own labeled examples, get a calibrated custom head back, with an eval report (accuracy + ECE) automatically generated.
- G7 — Reproduce or extend the **public game/computer-use harnesses** (ViZDoom, StarCraft Strongarm, browser-use flights task) with our model swapped in, evidence-traced, and license-checked.
- G8 — Ship an **agent-usable setup skill** so any AI coding agent (Claude Code, etc.) can install, run, benchmark, and fine-tune ekVachan with minimal human hand-holding.

### 4.2 Non-goals (explicitly out of scope for v1)
- We are **not** building a general chat/reasoning LLM. Same philosophy as Jev: "code calculates, ekVachan judges, a real LLM reasons and generates." Anyone needing open-ended text stays on their existing LLM.
- We are **not** trying to match frontier-LLM-scale pretraining. This is encoder-scale (0.1B–9B).
- We are **not** promising to beat Jev/Von on every single axis — the PRD sets targets, not guarantees (see Section 2).
- We are **not** building our own browser/game engines — we reuse or fork existing public harnesses (`browser-use`, ViZDoom, `tsai-sc`) rather than reinventing them.

---

## 5. Model & architecture design

### 5.1 Core architecture decision: encoder + classification heads, not logit-reading over a causal LM

Two families exist in the field (Section 3): (a) fine-tuned **bidirectional encoder** with dedicated classification heads (Von, Laya, Kev-style), and (b) **read the next-token logits of an existing causal LLM** at the answer position (Rizzo Flow, open-alternative-jev).

**Decision: (a), encoder + heads, as the primary architecture.**

Reasoning:
- Latency: encoder approach is already proven sub-20ms (Von: <18ms, Laya: ~16ms) vs logit-reading's 49ms (Rizzo Flow, *with* shared-prefix caching) to 582ms (open-alternative-jev at 27B). The gap is architectural, not an optimization detail — a bidirectional encoder sees the whole state in one pass with no causal-masking waste, and a purpose-built head is one matmul, not a full vocab softmax.
- Calibration: purpose-trained heads with an explicit calibration loss term (CE + Brier, temperature scaling) directly optimize the thing we're selling (calibrated probabilities), rather than repurposing a causal LM's next-token distribution, which was never trained to be calibrated over a small option set.
- Every top performer in the field (Von, Laya) is already this family. Fighting the logit-reading approach on its own turf (Rizzo Flow, open-alternative-jev) means fighting a slower architecture family — not worth it.
- Tradeoff we accept: encoder approach requires actually training/fine-tuning a model per capability tier, vs. logit-reading's "point it at any LLM you already have." We accept this because G6 (one-command fine-tuning) turns that cost into a feature, not a liability.

> **Superseded 2026-09-23 — read Section 14 Q4 first.** This section's decision (encoder + heads as primary) was reversed after both arms produced real numbers: the **decoder (Qwen3.5-4B LoRA, restricted-logit read) is now the primary architecture**, per 13a.2/13a.6/13a.7 and the Q4 decision entry. The reasoning below is retained as the original, pre-measurement argument and as the record of which of its bullets survived contact with data — not as current direction.

**Status 2026-09-22**: this is still a *decision on paper* that Phase 1 is actively testing, per 3.1a and Section 14 Q4 — both arms are implemented (`training/train_encoder.py`, `training/train_decoder_lora.py`) and share one metrics module so their numbers compare fairly. The encoder arm has real results (13a.1); the decoder arm is mid-run. Two of the three reasons above are now partly measured rather than argued: the encoder's serving latency is 27–36 ms unoptimized Python (13a.3, above the <15 ms target but nowhere near the logit-reading family's range), and the calibration bullet held — CE+Brier+temperature scaling produced 0.0344 ECE. The third bullet ("every top performer is this family") remains the weakest of the three, since 3.1a's leaderboard says the opposite on its own distribution.

### 5.1a Third track: a cross-attention decision head (addendum, post-Phase-1-baseline)

Neither 5.1's two families is a from-scratch architecture — both are standard recipes (classification head; restricted-logit reading) applied to existing pretrained backbones. Inventing and pretraining a genuinely new architecture (what a from-scratch alternative to ModernBERT's 2T-token pretrain would take) is correctly out of scope per Section 4.2 — that's a different order of compute budget entirely.

There is a real middle ground worth pursuing once the two baseline numbers exist to compare against: replace the flat classify-the-concatenated-state head with a **cross-attention / late-interaction decision head** — encode the state once, encode each candidate option, and score each via a lightweight interaction layer (ColBERT-style MaxSim, or a MICE-style light cross-attention from a frozen state representation into the option encoding) instead of naive concatenation into one classification head. This is an established, proven technique in reranking/retrieval (ColBERT, MICE) that nobody in the JevBench field has applied to typed-decision scoring — Von/Laya use flat heads, SemIf/reflex do logit-reading over a decoder. It's a genuine architectural differentiator, not a moonshot, and it composes with either base backbone.

Sequencing: this is a Phase 1.5 experiment, not a blocker — run it only after `ekvachan-base` (encoder) and the Qwen3.5-4B LoRA variant both have real numbers, so there's a floor to prove it beats before investing the extra engineering.

**Reweighted 2026-09-22 (13a.3/13a.4)**: this is no longer only an accuracy play. The Phase 1 checkpoint's hard limitation — a fixed head that can't answer an option set it wasn't trained on — is exactly the limitation an option-encoding interaction head removes by construction. So 5.1a is now also the most plausible route to a *general* `choice` primitive, which Section 6's contract requires and today's checkpoint can't provide. Sequencing (after both baselines have numbers) is unchanged; its priority relative to other Phase 2 work goes up.

### 5.2 Multi-tier model family (mirrors Kev/Von's tiering, sized against their published numbers)

| Tier | Params | Target latency (local, batch=1) | Role |
|---|---|---|---|
| `ekvachan-nano` | ~0.3–0.5B (ModernBERT-base scale) | <10ms | First-pass tier in the cascade (G5); handles the confidently-easy majority of decisions |
| `ekvachan-base` | ~0.4B (ModernBERT-large scale, matches Von's weight class for a fair head-to-head) | <18ms, target <15ms | Primary tier; this is the model we benchmark against Von directly |
| `ekvachan-vision` | see 5.2a — no longer "fuse our own vision encoder from scratch" | <30ms target | The multimodal differentiator (G4) — DOM-screenshot + text state, game frames, robot camera input |
| `ekvachan-large` (stretch, phase 3+) | 4–9B | <150ms | For the hardest cases the cascade escalates to; optional, only if benchmarking shows the cascade needs a third tier |

Rationale for sizing at the small end (not chasing Kev's 9B or an even bigger model): every disclosed benchmark in Section 3 shows near-Jev or better accuracy from *sub-1B* encoders. Size isn't the bottleneck in this problem class — training data quality and calibration method are. Spending compute on a bigger dense model is the lowest-leverage lever available; spending it on data and the cascade/vision work is higher-leverage. Revisit only if `ekvachan-base` benchmarks show a real accuracy ceiling.

### 5.2a Multimodal plan, revised: two paths that both avoid inventing vision fusion ourselves (addendum)

The original plan for `ekvachan-vision` ("base tier + a vision encoder fused before the pooling head") implied a from-scratch research project — nobody in this space had done exactly that when 5.2 was drafted. Two things checked since then change that:

1. **If the decoder track (5.1a / Section 5.1's comparison) ends up competitive**: `Qwen/Qwen3.5-4B` is *already natively multimodal* — text, image, and video input, confirmed on its own model card and release docs. Multimodal capability would come essentially free, no separate fusion work at all, just feeding image input through the same chat-template path already used for the restricted-logit read.
2. **For the encoder track**: **ModernVBERT** ([HuggingFace: `ModernVBERT`](https://huggingface.co/ModernVBERT)) already exists — a published, open-weight 250M model that fuses ModernBERT with a SigLIP2 vision tower via MLM (10B tokens) + InfoNCE training, with weights, intermediate checkpoints, and training code all public. `ekvachan-vision` should build directly on this proven recipe/checkpoint rather than inventing our own vision-fusion approach from scratch — same de-risking logic as reusing ModernBERT-large itself instead of pretraining an encoder from zero (Section 5.3).

Net effect: multimodal is no longer the PRD's highest-uncertainty line item. Sequencing unchanged (still Phase 3), but the *how* is now concrete and low-risk regardless of which base architecture Section 5.1's comparison favors.

### 5.3 Training method
- Base: pretrained open encoder checkpoint (ModernBERT-large is the proven starting point in this space — reuse it, don't pretrain from scratch; NanoJev is the only from-scratch project and it has the weakest headline numbers, which is a real signal, not a coincidence).
- Loss: cross-entropy + Brier-score calibration penalty (Von's disclosed recipe, λ≈0.5) + post-hoc temperature scaling. This is publishable and reproducible, unlike Jev's undisclosed RLCD.
- Data mix (informed by Von's disclosed ~290k-example corpus balance, adjusted for our differentiators):
  - Operational/enterprise workflow decisions ~20%
  - Security/DevOps/compliance ~15%
  - Safety/policy/moderation ~15%
  - Linguistic/content semantics ~10%
  - Triage & services ~10%
  - Adversarial NLI (ANLI/WANLI/MultiNLI/SNLI — standard, license-clean, public) ~15%
  - **New**: game/computer-use decision traces (DOM element choice, ViZDoom action choice, StarCraft unit-order choice) sourced from the public harnesses in G7, replayed and relabeled ~15% — no other open alternative trains on this distribution directly, and it's exactly our target eval
- Vision tier: contrastive/paired (screenshot region ↔ DOM element, game frame ↔ action) data generated by instrumenting the same public harnesses — cheap to produce because the harnesses already exist and are scriptable.
- Everything about this training recipe gets published (data sources, loss function, hyperparameters, eval harness) — this is itself a differentiator, since Jev's RLCD is proprietary and even Von doesn't fully disclose data provenance beyond category percentages.

### 5.3a Calibration upgrade path: conformal prediction (addendum, architecture-agnostic)

Section 5.3's cross-entropy + Brier + temperature-scaling recipe is the right default — it's what every credible entrant in this space ships (reflex, Laya, Von all use some form of it) and it's cheap. Worth adding on top, as an *opt-in* response mode rather than the default: **split-conformal prediction** on a held-out calibration set, returning a prediction set with a stated coverage guarantee ("the true answer is in this set with ≥95% probability, provably, not just empirically calibrated") instead of a single point probability. This is a genuinely different claim from temperature scaling — temperature scaling makes a confidence number locally honest on average; conformal prediction gives a per-call guarantee. It's unclaimed territory in this ecosystem (the awesome-jev catalogue lists one exploratory project, `jev-certify`, probing conformal routing thresholds, but nobody ships it as a first-class, tested primitive) and directly answers rizzo-flow's own documented weak point above ("uncalibrated by default... weak abstention"). Cheap to add regardless of which architecture 5.1 lands on — it wraps *any* model's output distribution, doesn't require retraining.

### 5.3b Data-collection method for the game/computer-use slice (addendum)

Section 5.3's data mix already earmarks ~15% for "game/computer-use decision traces... replayed and relabeled" — worth being concrete about *how*, since static replay of existing harness logs will under-cover the actual on-policy distribution our model will face. `PlayJev` (an independent 0.8B model trained to play ten browser games from raw pixels, [OmniJev/PlayJev](https://github.com/OmniJev/PlayJev)) used a proven two-stage recipe worth adopting for this slice specifically: (1) behavior-clone one epoch over teacher-labelled frames, then (2) two rounds of **DAgger** (the student plays, a teacher relabels every frame the student actually visited, retrain on the aggregated set) — this closes the distribution-shift gap that pure offline replay leaves open, which matters most exactly where Jev itself is documented to struggle (Section 1.4's Minecraft finding: solo System One models loop/fixate without on-policy correction). Apply this to whichever harness(es) Section 8 targets (ViZDoom, StarCraft, the browser-use flight task) before relying on their logs as training data.

### 5.4 What we are *not* speculating on
We will not claim to have reverse-engineered or replicated Jev's actual internals. Section 1.1's architecture hypothesis is explicitly someone else's medium-confidence guess. Our architecture is our own documented decision (5.1–5.3), not an attempt to clone an unknown black box.

---

## 6. API & SDK design

### 6.1 Compatibility layer (copy Jev's shape, per the brief)
`POST /v1/systemone` — request/response shape matching Jev's (Section 1.2), so existing Jev/Rizzo-Flow/Von client code works by changing a base URL. This is the "as easy as possible for people to use" lever the brief asked for — low migration cost for anyone already integrated with Jev's ecosystem (and there already are 12+ language SDKs and a dozen framework integrations built against this exact shape — Section "ecosystem" findings, not reproduced here for brevity but see research dossier).

Two corrections to this section's original wording, made 2026-09-22 after the server was actually built (13a.3):

- It used to say "byte-for-byte compatible." That's not a claim we're in a position to make. The *request* shape follows Section 1.2, which is itself reconstructed from secondary sources; the *response* field names (`results`, `usage`) are our best-effort reconstruction and have never been diffed against a real Jev response body, because there's no live Jev API to diff against. Honest framing: **same request shape, best-effort response shape**. If anyone gets API access, diffing a real response is cheap and should be done before the README claims compatibility.
- **The Phase 1 implementation is a fraction of this contract.** Only `choice` is implemented, and only for one fixed option set (10a). `score` and `noul` return 501. This section describes the target contract, not today's server.

### 6.2 Native API (richer than the compatibility shim)
- Adds a 4th primitive TypeSafe doesn't have: **`vision_choice`** — same as `choice` but `state` may include an image/region reference, backed by `ekvachan-vision`.
- Exposes **cascade control** explicitly: request can pass `max_latency_ms` or `min_confidence`, and the server decides nano-only vs nano+base escalation transparently — this is the serving-layer differentiator from G5, exposed as an API knob rather than hidden.
- Returns **which tier answered** and its actual latency in the response, for observability — nobody else in the landscape surfaces this.
- `POST /v1/finetune` (local-only, no auth needed for self-hosted) kicks off the one-command fine-tuning flow from G6, returns a job id and eval report on completion.

### 6.3 SDKs
v1: Python + TypeScript (matches Jev's own official-tier languages, and what browser-use/jev-ultrafast expects). Everything else (Go/Rust/Java/etc.) — accept community contributions rather than building 12 SDKs ourselves; document the wire format precisely enough that this is easy (Jev's own ecosystem shows this works: most of its 12 SDKs are community-built against a stable, well-documented HTTP contract).

---

## 7. Serving / inference architecture

### 7.1 Server
- Rust (axum) service, ONNX Runtime or TensorRT for the encoder forward pass, exposing the HTTP contract from Section 6. Python is the training/fine-tuning language; Rust is the serving language. Reasoning: every latency-sensitive project in this space that discloses its stack leans on optimized non-Python inference (Rizzo Flow uses llama.cpp precisely because pure-Python serving couldn't hit its own latency targets, and even then it's 1.8× slower than its own prior MLX runtime for the same reason — runtime choice is a first-order latency lever here, not a detail).
- Shared-prefix / prefill caching for multi-question requests against one `state` (same trick Rizzo Flow already validated: process `state` once, branch per question). Batch scheduler for the cascade: nano tier runs first for every request; only low-confidence requests get forwarded to base (or large) tier.

### 7.2 Quantization
Q8/Q4 export paths (matching what Rizzo Flow already validates as viable at this model scale: Q8_0 4.4GB → Q4_K_M 2.6GB for a 4B model, negligible accuracy loss reported). Ship all tiers pre-quantized on Hugging Face alongside full-precision weights.

### 7.3 Latency budget (targets, to be validated empirically — Section 8 has the actual accountability numbers)
| Stage | Target |
|---|---|
| `ekvachan-nano` (cascade first pass) | <10ms |
| `ekvachan-base` (escalation / direct call) | <15ms |
| `ekvachan-vision` (with image encode) | <30ms |
| Mean latency across a realistic confidence distribution (cascade) | target: beat Von's flat 18ms on **average**, not necessarily on every single call |

Measured so far (13a.3): `ekvachan-base` serves at **27–36 ms warm** through the Python reference server — above target, as expected for eager-mode PyTorch. Treat that as the unoptimized baseline the ONNX/Rust port (7.1) and quantization (7.2) have to improve on, and as the first honest data point that the <15 ms target is a target, not a measurement.

### 7.4 Reusing public harnesses — license check (completed 2026-09-22)
Checked directly against each repo's GitHub API license endpoint, not assumed from org reputation:

| Repo | License | Implication |
|---|---|---|
| `browser-use/jev-ultrafast` | MIT | Free to fork/vendor with attribution — keep the MIT copyright + permission notice in any copied file |
| `phyous/tsai-sc` | MIT | Same |
| `AbdelStark/heist-one` (HEIST//ONE) | MIT | Same |

All three clear. When we actually fork any of them in Phase 2/3, keep the original LICENSE file (or the MIT notice block) alongside the vendored code, and note in our own README which parts are adapted from which upstream repo — MIT only requires the notice be preserved, but doing this makes the provenance trail auditable, which matters for a project whose whole pitch is evidence discipline (Section 8.2). Von's ViZDoom harness license was not checked in this pass (we don't yet know its exact repo path) — check before Phase 2 forks it specifically.

---

## 8. Benchmarking & "proof" strategy — what "better" is accountable to

This is the section that makes G3/G7 falsifiable instead of a slogan.

### 8.1 Standing benchmarks we commit to running and publishing, win or lose
1. **jabr-v2** (49-task, 869-case OOD benchmark Von reports on) — reproduce Von's exact eval harness/seeds if publicly available; if not, use the same task categories and disclose the difference. Target: beat Von's 72.0% macro-accuracy. Publish our number regardless of outcome.
2. **ViZDoom** `Defend the Center` and `Health Gathering`, same 8 shared seeds Von used. Targets: beat Von's 9.00 kills *and* its 12.11s survival (note: Von itself trails Jev's 13.03s on survival — so "beat Jev" and "beat Von" are two different bars here; report both).
3. **StarCraft Strongarm** mission via `tsai-sc`'s harness (license permitting) or a faithful reimplementation — measure win rate and attempts-to-first-win, compared against the "attempt 16" figure reported for Jev.
4. **browser-use/jev-ultrafast task** (Zurich→London Google Flights, plus its Wikipedia and hotel-search variants) — swap ekVachan in for Jev via the compatibility API, no other code changes, measure wall-clock and CDP call count against the disclosed 7.07s / 101-call baseline.
5. **Calibration**: publish ECE (expected calibration error) the way Laya and Von do — this metric is currently a differentiator few alternatives report; we report it by default, always, not just when favorable. First numbers, on our own eval split rather than any of the benchmarks above, are in 13a.1 — published here whether or not a shared-benchmark number ever flatters them.

### 8.1a JevBench as an additional standing benchmark (addendum)
Add JevBench (Section 3.1a) to the standing suite alongside jabr-v2/ViZDoom/StarCraft/browser-use — it's a third-party-maintained, MIT-licensed, already-public leaderboard, which makes a result there harder to dismiss as self-graded than a benchmark we run entirely ourselves. Its own anti-gaming ruleset is worth adopting as a template for 8.2 regardless of which benchmark it's applied to: held-out items and label-mappings committed *before* any run starts, no schema-repair retries, no per-model spending exceptions, one shared budget across all runs.

### 8.1b A benchmark-design lesson from an existing audit (addendum)
An independent KoBBQ audit of hosted Jev found it answers "unknown" on 95% of *deliberately ambiguous* items — worth citing here not as a knock on Jev but as a caution for our own eval design: with the ambiguous-item gold label removed, forced-choice accuracy on that slice is trivially 0%, and forcing an answer anyway pushed the model toward the dataset's stereotype 79% of the time. Section 8's own accuracy numbers (jabr-v2, ViZDoom) should score **abstention separately from forced-choice correctness** wherever the harness allows an "insufficient evidence" response — conflating the two makes a well-calibrated model that correctly declines to guess look worse than a model that guesses confidently and wrong, which is exactly backwards for a project whose stated differentiator is calibration.

### 8.2 Evidence discipline
Every benchmark run ships as a signed evidence bundle (raw outputs, seeds, prompt/state hashes, weight hash, timestamp) in `results/` — matching the norm this ecosystem has already converged on (Rizzo Flow's evidence traces, jabr-v2's "frozen benchmark" design, HEIST//ONE's evidence traces, JevBench's frozen-and-hashed test cases). This is non-negotiable for credibility in a field this benchmark-literate.

### 8.3 What "we proved it" means in practice
A claim like "ekVachan beats Jev at StarCraft" is only true once section 8.1's harness has actually run and the evidence bundle is in the repo. Until then, it's a target, and the README must say so.

---

## 9. Compute plan

**This is not a frontier-pretraining project.** Every disclosed competitor operates at 0.4B–9B parameters and reports training/fine-tuning times in the range of "under 2 hours on a single consumer GPU" (Kev: ~1h45m on Apple Silicon for its full family) to a few A100/H100-hours for larger runs. Concretely:

| Workload | Scale | Suggested compute | Rough time |
|---|---|---|---|
| `ekvachan-nano`/`ekvachan-base` fine-tune from ModernBERT checkpoint | 0.3–0.5B | 1 GPU, 16GB+ VRAM is comfortable | Hours, not days |
| `ekvachan-vision` fusion training | base + small vision encoder | 1 GPU, 24GB+ VRAM recommended (image batches are the constraint) | Low single-digit hours per iteration |
| `ekvachan-large` (stretch tier, only if the cascade proves it's needed) | 4–9B | 1 GPU, 40GB+ VRAM (80GB comfortable for larger batch sizes); multi-GPU only if you want faster wall-clock, not because it's required | Under a day per run |
| Full benchmark suite (Section 8) | inference only | Runs fine on a single consumer GPU, or CPU for the smaller tiers | Hours |

**Decision: self-hosted, on the project owner's own Linux server** — no rented GPU marketplace needed for v1. This removes the spend question entirely (Section 2's "source heavy GPU compute if worth it" resolves to: not worth renting, your own hardware covers every workload above at this model scale) and simplifies the training setup (no spot-instance preemption handling, no egress cost for moving checkpoints/data, full control over the environment).

**Update 2026-09-22**: Phase 1 has now actually run on this server (13a) and the plan held at the encoder scale — but with one constraint this table didn't anticipate. It sizes workloads by parameter count and VRAM; the decoder arm was bottlenecked by **kernel availability** instead, falling back to un-fused reference implementations for Qwen3.5-4B's linear-attention/SSM-style layers and blowing up both time and memory (13a.2). Add that to the preflight checklist for any future run on a hybrid-architecture backbone: confirm the fused kernels are installed before trusting a time estimate derived from parameter count. The paragraph below is left as originally written, since the hardware spec it asks for still hasn't been recorded in this repo.

**What we still need to know before Phase 1 sizing is final**: the server's GPU model and VRAM (run `nvidia-smi` and share the output), and how many GPUs. That determines batch size and whether `ekvachan-vision`/`ekvachan-large` are same-session-feasible or need to wait/queue behind `ekvachan-base`. Everything in the table above assumes a single modern datacenter or high-end consumer GPU (e.g. anything from a 3090/4090 up through an A100/H100 class card) — if the server's card is smaller (e.g. under 16GB VRAM), `ekvachan-base` is still very achievable, just with smaller batch sizes and gradient accumulation, and `ekvachan-vision`/`ekvachan-large` would need either more VRAM or an int8/QLoRA fine-tuning path instead of full fine-tuning.

**Recorded 2026-09-23** (`nvidia-smi`, shared lab server): single **NVIDIA L40S, 46068 MiB VRAM**, driver 580.126.09, compute capability 8.9. Comfortably in the "single modern datacenter GPU" bracket the table above assumes -- every real Phase 1 run so far (encoder 1.2M examples, decoder 24-60K examples) fit within this budget with room to spare, and the decoder's actual VRAM constraint has consistently been missing fused kernels (13a.2/13a.4), not raw capacity. Shared with other lab users -- observed utilization/load varies outside this project's own jobs.

No cloud spend budget needed for this plan as it stands. If the server ever becomes a bottleneck (e.g. wanting to parallelize multiple experiments), a rented spot GPU (RunPod/Lambda/Vast.ai) remains a fallback option, not a requirement.

---

## 10. Open-source strategy & licensing
- **Code**: Apache-2.0 (matches Von, Rizzo Flow, Kev, open-alternative-jev — the de facto norm in this exact ecosystem; maximizes commercial + academic adoption, which directly serves the "easy for anyone to use" goal).
- **Weights**: same, published on Hugging Face, all tiers, full-precision + quantized.
- **Training data**: publish sources/mix (Section 5.3) and any harness-derived data we generate ourselves; do not redistribute any dataset whose license forbids it — audit before publishing.
- **API compatibility posture**: stated explicitly in the README, mirroring Rizzo Flow's own disclaimer language — "reproduces Jev's interface pattern for interoperability; does not reproduce TypeSafe's proprietary architecture, weights, or RLCD training." This is both accurate and legally clean (thin JSON interfaces for interoperability are well-trodden ground, and three other projects already operate this exact way in the open without incident).

---

## 10a. Known limitations & responsible use

Stated plainly, in one place, rather than left implicit:

- **Calibrated is not the same as correct.** Section 1.3 already notes TypeSafe's own admission that Jev's "zero hallucination" means schema compliance, not truthfulness — it can be confidently wrong. The same is true of ekVachan by construction: temperature scaling and the Brier penalty (Section 5.3) make confidence scores *track* accuracy on average, they don't guarantee any single answer is right. A well-calibrated 90% confidence is still wrong 1 time in 10.
- **Training data carries its own biases.** The Phase 1 data mix (Section 5.3) is sourced from public NLI corpora (ANLI/WANLI/MultiNLI/SNLI) plus harness-derived traces — none of it is bias-audited beyond the dedup/leakage filtering Section 5.3/the actual pipeline already does. A model fine-tuned on this data for **safety/policy/moderation decisions specifically** (an explicit ~15% slice of the data mix) should not be treated as a ground-truth arbiter without human review in the loop, especially for consequential or ambiguous cases — Section 8.1b's KoBBQ finding (forced answers on ambiguous items skew toward stereotype 79% of the time) is a reason to score abstention separately, not a reason to trust forced answers on hard cases.
- **The Phase 1 checkpoint answers exactly one question schema** (added 2026-09-22, measured not hypothetical — 13a.3). `ekvachan-base` as trained today has a fixed 3-way classification head that never saw `options` text during training. It can only answer `choice` questions whose options are literally `["entailment", "neutral", "contradiction"]`; it has no mechanism for an arbitrary option list, including the `["billing", "technical", "other"]` example in Section 1.2. `score` and `noul` have no trained model at all. The server enforces this with a 501 rather than mapping an unknown option set onto the head it has — a wrong-but-confident answer here would be the single most misleading failure mode this project could ship, given that calibration is the entire pitch. Anyone reading Sections 5/6/7 should read them as the target design; this bullet is the current capability.
- **Self-hosting means the operator owns these tradeoffs.** Because this is self-hosted open-weight software, not a hosted product with a shared safety team behind it, whoever deploys it is responsible for appropriate review/human-in-the-loop design for their own use case — this is stated here so it isn't left as an unstated assumption.

This is intentionally proportionate to the project's actual scale (a small, self-hosted, open-weight classifier) — not a claim that heavier processes (RLHF pipelines, dedicated red-teaming programs, external audits) are warranted here; those belong to a different class of system than this one.

---

## 11. Agent skill (the "internal skill folder" ask)

Scaffolded at `skills/ekvachan-setup/`. Purpose: let a user hand this repo to their own coding agent (Claude Code or similar) and have the agent self-serve setup, without the human reading install docs.

v1 skill capabilities (see `skills/ekvachan-setup/SKILL.md` for the actual instructions):
1. **Install & serve** — clone, pull the right weight tier for the host's hardware, start the local server, verify with a smoke-test call.
2. **Compat-check** — point an existing Jev-integrated codebase at the local server and flag any request shape it doesn't yet support.
3. **Benchmark** — run the Section 8 suite locally and produce the evidence bundle.
4. **Fine-tune** — take a user-supplied labeled dataset, run the one-command pipeline (G6), report back accuracy + ECE.

This mirrors the fact that TypeSafe itself already ships "an official agent skill for Claude Code integration" for Jev — so this is table stakes for adoption in this ecosystem, not a nice-to-have.

**Status 2026-09-22**: all four capabilities above are still described against an `ekvachan` CLI that doesn't exist — `SKILL.md` documents the target workflow, and its own status note tells the agent to say so plainly rather than fabricate install output. That note now needs updating: it claims `training/`/`eval/`/`server/` don't exist, which is stale for `training/` and `eval/` (they do) and misnamed for serving (`serve/`, not `server/`). The honest current state for an agent to report: training and eval code exist and are runnable as Python modules, the reference server runs and answers one schema, and none of it is wrapped in a CLI yet.

---

## 12. Repo structure (as created)

```
better-jev-for-all/
  README.md              — short public-facing overview, points to this PRD
  PRD.md                 — this document
  LICENSE                — Apache-2.0
  pyproject.toml         — Python deps for training + the reference server (torch/transformers/datasets/peft/fastapi)
  training/
    data.py               — Phase 1 data pipeline: ANLI+WANLI+MultiNLI+SNLI → the `choice` schema, deduped + leakage-filtered
    train_encoder.py      — `ekvachan-base`: ModernBERT-large + CE/Brier loss (5.3) + post-hoc temperature scaling, writes an evidence manifest
    train_decoder_lora.py — the 5.1 comparison arm: Qwen3.5-4B LoRA, restricted-logit read, per-example letter shuffling
  eval/
    metrics.py            — accuracy, multi-class Brier, equal-mass ECE, temperature fitting; shared by both training paths so their numbers are directly comparable
  serve/
    inference.py          — Phase 1 Python reference wrapper around the trained checkpoint
    server.py             — `POST /v1/systemone` (FastAPI) + `/health`
  skills/
    ekvachan-setup/
      SKILL.md            — agent-facing setup/benchmark/fine-tune skill
  docs/                   — research notes, landscape tracking, benchmark write-ups (grows over time)
```

`data/` and `checkpoints/` are gitignored — the processed dataset and trained weights live on the training server, not in the repo. Weights aren't published to Hugging Face yet (G2 remains open until they are).

Still **not** built, and deliberately so at this stage: the Rust/ONNX production server (7.1 — `serve/` is a Python reference implementation whose job is to validate the wire contract before committing to a runtime port), the `results/` evidence bundles (8.2), the nano tier and cascade (5.2/G5), the vision tier (G4), `/v1/finetune` (6.2/G6), and the Python/TS SDKs (6.3). Section 13 sequences the rest; Section 13a records what the built parts actually measured.

---

## 13. Roadmap

- **Phase 0 (this PRD)**: research, decisions, repo + skill scaffold. ✅ this document.
- **Phase 1** — partially done; per-item status as of 2026-09-22, numbers in 13a:
  - ✅ Data pipeline (`training/data.py`): the NLI slice of the 5.3 mix, built and verified against the real datasets, 1,206,855 train examples after dedup + train/eval leakage filtering.
  - ✅ `ekvachan-base` encoder fine-tune off ModernBERT-large with the 5.3 CE+Brier+temperature-scaling recipe — trained, evaluated on a held-out test split, manifest with weight hash saved.
  - ✅ Decoder comparison arm (`training/train_decoder_lora.py`, the Qwen3.5-4B restricted-logit variant 3.1a/5.1 asked for) — **finished and it won**: 92.25% / 0.0232 raw ECE on a 60k-example subset, vs. the encoder's 86.32% / 0.0344 on 20× the data (13a.2). 13a explains why it's a subset.
  - ✅ Schema-general `choice` via the decoder — `training/train_decoder_lora_multischema.py` then `..._wideschema.py` (`MAX_OPTIONS=26`): **96.10% zero-shot on a genuinely unseen wide 15–26-way schema** (13a.6). This is the capability 13a.4 point 2 called a Phase 1 blocker, reached through training data rather than a 5.1a architecture change.
  - ✅ Architecture decision locked — **decoder is primary** (Section 14 Q4, 2026-09-23). 5.1's encoder-first framing is superseded.
  - ✅ Compatibility-layer server (`serve/`) — `POST /v1/systemone` now defaults to the decoder (`serve.inference.DecoderChoiceModel`), verified end to end over real HTTP: 87-109ms warm latency, any 2-26 option `choice` request, `score`/`noul` correctly 501 (13a.8).
  - 🔄 jabr-v2 and JevBench, with evidence bundles in `results/` — **harnesses built, vendored, and run for real** (13a.7): JevBench 83.45% on 139/231 schema-answerable items, jabr-v2 88.89% on 387/944. Still **not** a competitive claim against Von or Jev — neither is a complete benchmark score. G3 is testable now, not met.
  - ❌ ViZDoom, StarCraft, and browser-use/jev-ultrafast benchmarks (8.1 items 2-4) — not started, no harness code. **Deliberately resequenced (2026-09-23, owner-directed) to run after the bench-driven retrain below**, not before -- Von's own headline numbers are ViZDoom-based, so this is the real head-to-head, and it should reflect the wider-data model, not today's narrow one.
  - ✅ `score` and `noul` primitives — **first training run complete** (13a.8): `noul` 88.80%, `score` 75.40% (diagnosed as genuine task-boundary difficulty, not a broken mechanism -- see 13a.8's confusion-matrix follow-up). Narrow (one dataset each) -- widening this is precisely what better-jev-bench's Tier-A pull is for, per the resequencing below. **Status as of 2026-09-23: that pull now exists** -- 422,878 items across 8 Tier-A datasets and 11 tasks, including a 5-level ordinal `score` source and two `noul` sources with opposite skew profiles, exported in this project's own training-record shape. See the cross-reference note in `STATUS.md` and better-jev-bench's PRD §11.4/§12.
  - ❌ Weights published to Hugging Face -- not yet.
- **Phase 2, resequenced 2026-09-23 (owner-directed) -- bench-first, not tier-first:**
  1. **better-jev-bench**: loader/plugin framework, CI gates, automated download/processing for the ~51 already-catalogued Tier-A datasets -- made genuinely easy to point at for training or benchmarking, not just a license-tiered list. **Framework and the first 8 datasets landed 2026-09-23** (422,878 items, frozen hash-committed held-out slice, `bjb export` into this repo's exact training-record shape); ~43 Tier-A entries and the scoring implementation remain. See `better-jev-bench/STATUS.md` and the cross-reference note in this repo's `STATUS.md`.
  2. **Retrain the base decoder** on the resulting wider corpus -- more/wider `choice`/`noul`/`score` sources than today's two narrow primitive datasets.
  3. **`ekvachan-vision`**: multimodal training data, likely the same 4B backbone (Qwen3.5 is natively multimodal) rather than a separate model -- see 5.2a.
  4. **`ekvachan-nano`**: smaller tier, same bench-derived data mix.
  5. **Game harnesses** (ViZDoom/StarCraft/browser-use/jev-ultrafast) run against the post-bench checkpoint -- the real Von comparison.
  6. **Quantization** export + eval.
  7. **Public release**: README/PRD/STATUS final pass, GitHub polish, announcement posts -- deliberately after 1-6, so the public story is backed by the wider-data model and a real Von comparison, not today's narrower one.
- **Phase 3 (trails the public release, not a blocker for it)**: Rust/ONNX serving port (7.1) -- the real "faster than Jev" latency work; today's Python server (87-109ms) already proves the wire contract honestly with a stated target. Full SDK polish (6.3), `/v1/finetune` (6.2/G6).
- **Phase 4**: cross-attention decision head (5.1a) -- demoted 2026-09-23; current mechanism already gets 83-89% on real benchmarks and its unique value (>26-option items) hasn't come up in any real benchmark data seen so far (widest real option count: 6, per 13a.5).
- **Phase 5 (stretch, explicitly last, owner-directed)**: `ekvachan-large` third tier, only if cascade benchmarking (itself not yet built) shows a real accuracy ceiling the smaller tiers can't clear. No design work started.

The original Phase 1 exit condition is now **partially met, honestly scoped**: 13a.7 put a real, evidence-backed number against JevBench/jabr-v2 (not yet Von's own ViZDoom numbers, and not a complete-benchmark score) -- a floor, not the full comparison. The resequencing above is exactly the plan to close that gap for real, rather than publishing a comparison against a model that hasn't yet had the benefit of the wider bench data.

---

## 13a. Phase 1 run log — the first real measured results (addendum, 2026-09-22)

Everything above this line was written before any training ran. This section is the first entry that isn't a plan. Rule for it, same as the rest of the doc: only numbers that were actually produced by a run go here, and each one says what it does *not* prove.

### 13a.1 `ekvachan-base` (encoder) — trained, evaluated, served

Recipe as specified in 5.3, no deviations: ModernBERT-large, classification head, CE + Brier (λ=0.5), post-hoc temperature scaling fit on a held-out calibration split. Trained 2 epochs on the full NLI slice from `training/data.py` — **1,206,855 examples** (ANLI + WANLI + MultiNLI + SNLI, deduped, with eval pairs appearing in train stripped out).

Results on the held-out **test** split (n = 35,486). Temperature was fit on a **disjoint** 15,207-example calibration split, so nothing is fit on what it's scored on:

| Metric | Raw (T=1) | Calibrated (T=1.345) |
|---|---|---|
| Accuracy | 86.32% | 86.32% (temperature scaling is monotonic — accuracy is unchanged by construction) |
| ECE (equal-mass, 15 bins) | 0.0689 | **0.0344** |
| Brier (multi-class) | 0.2172 | **0.2078** |

Manifest with weight SHA256, hyperparameters, and both reports saved alongside the checkpoint.

**What this proves**: the 5.3 recipe works end to end on real data at this scale, and temperature scaling roughly halves ECE — the calibration commitment in Section 8 is implementable, not aspirational. It's a usable floor.

**What this does not prove, stated plainly**:
- It is **not** G3. G3 is beating Von's published numbers on a reproduction of Von's benchmarks; none of those have been run. 86.32% on 3-way NLI and Von's 72.0% jabr-v2 macro-accuracy across 49 tasks are different tasks on different distributions — putting them in the same sentence as if one were higher than the other would be exactly the kind of number-laundering this doc exists to avoid.
- The ECE comparison to **Laya** (0.081 post-temperature-scaling, Section 3.1 — still the only alternative publishing ECE directly) is suggestive, not a win: 0.0344 is lower, but it's measured on our own protocol and our own NLI-only distribution, not a shared benchmark either of us both ran. It becomes a real comparison at JevBench/jabr-v2, not before.
- The eval distribution is narrow. This checkpoint has seen one task family. Section 5.3's other six data slices (operational, security, safety/policy, semantics, triage, game/computer-use traces) are all still unbuilt, and the fixed-schema limitation in 13a.3 is a direct consequence of that.

### 13a.2 Decoder comparison arm — finished, and it wins decisively on this split

`training/train_decoder_lora.py` (Qwen3.5-4B, LoRA, restricted-logit read) trained on a **60,000-example subset** (measured decision, see below), 1 epoch, then evaluated on the **same** held-out test split as the encoder (n=35,486, same 15,207-example disjoint calibration split, same `eval/metrics.py`):

| Metric | Raw (T=1) | "Calibrated" (T=1.1455) |
|---|---|---|
| Accuracy | **92.25%** | 92.25% (monotonic, unchanged by construction) |
| ECE (equal-mass, 15 bins) | **0.0232** | 0.0437 |
| Brier (multi-class) | **0.1190** | 0.1220 |

Side-by-side against the encoder (13a.1), via `training/compare_architectures.py`:

| Metric | Encoder (`ekvachan-base-run2`) | Decoder (`ekvachan-decoder-qwen-run1`) |
|---|---|---|
| Train size | 1,206,855 | 60,000 (20× less) |
| Epochs | 2 | 1 |
| Total train time | 5h24m | 3h11m |
| Accuracy | 86.32% | **92.25%** (+5.93pp) |
| Best ECE | 0.0344 (calibrated) | **0.0232** (raw) |
| Best Brier | 0.2078 (calibrated) | **0.1190** (raw) |

**The decoder wins on every axis measured here, and by a wide margin, on ~5% of the training data and in less wall-clock time.** Its raw (uncalibrated) ECE alone beats the encoder's best (calibrated) ECE. This is the first real, direct evidence in Section 14 Q4's favor of the decoder family — on our own eval protocol, not yet on JevBench/jabr-v2 (13a.4 still applies).

**One genuine anomaly, reported rather than smoothed over**: temperature scaling made the decoder's ECE and Brier *worse*, not better (0.0232→0.0437, 0.1190→0.1220) — the opposite of the encoder's result, where it roughly halved ECE. The likely cause, not yet confirmed: `train_decoder_lora.py`'s calibration fitting works from `log(restricted_softmax_probs)` as a logit surrogate (there's no access to true pre-softmax logits over an unbounded vocabulary the way the encoder has), and that surrogate may not behave like a true logit under temperature rescaling the way the fitting procedure assumes. This is a measured artifact of the current calibration *method* for this architecture, not evidence the decoder's underlying confidence is poorly calibrated — its raw ECE is the best number in this whole comparison. Worth fixing the calibration procedure before trusting the decoder's "calibrated" column for anything; use raw for now.

The subset-size decision behind these numbers, and the measurement that motivated it, stands as originally recorded:

- Qwen3.5-4B is a hybrid architecture — it carries linear-attention/SSM-style layers (`causal_conv1d`, `chunk_gated_delta_rule`). On this server, the optimized kernel packages for those paths (`causal_conv1d`, `flash-linear-attention`) aren't installed, so the model falls back to un-fused reference PyTorch implementations: slow, and much more memory-hungry.
- Consequence, measured on the actual run, not projected from a spec sheet: **batch size 8 OOM'd; only batch 4 with gradient checkpointing is stable**, and a full-dataset epoch extrapolates to roughly **62 hours** — not a viable thing to hold a shared, multi-tenant lab GPU for. Hence 60k.
- For contrast, on the same machine the encoder arm trained on the **full 1.2M examples, twice over**, and finished.

**The generalizable point, revised now that both numbers exist**: the two architecture families differ by a large factor in *per-example* training cost — but the decoder needed **20× less data** to beat the encoder outright, so total wall-clock for this comparison actually favored the decoder (3h11m vs 5h24m), not the encoder. The honest, now-measured version of the cost story is narrower than 13a.2 originally framed it: per-example cost is real and driven by missing fused kernels (still true, still unmeasured how much they'd help), but "decoder = more expensive to iterate on" is not simply true once data-efficiency is accounted for — a self-hoster fine-tuning on their own modest labeled set may find the decoder *cheaper* in practice, not more expensive, if their dataset is small like this one. The full-dataset 62-hour extrapolation (below) is a real number for *that specific scenario* (all 1.2M examples), not a general verdict on the architecture's cost. Feeds Section 14 Q4, now with a genuine tension instead of a one-sided caution: better accuracy/calibration and better data-efficiency, against worse per-example throughput and a less trustworthy calibration procedure.

**Honest scoping of that claim** — it is partly a property of *this server's software environment*, not purely of the architecture:
- Installing the fused kernels would narrow the gap by an unknown amount. Nobody has measured how much, here or, as far as this pass found, publicly for this model. "Unknown" is the accurate answer; don't round it to "it would be fine."
- A decoder backbone without SSM-style layers (a plain-attention 4B) wouldn't hit this fallback path at all, so this is not evidence that "decoders are expensive" in general.
- It *is* evidence about the decoder track as a self-hoster would actually meet it: default install, shared GPU, no hand-tuned kernel setup. That's the deployment story 4.0 cares about, so it's a legitimate input to Q4 — weighted as one axis, not treated as decisive.

### 13a.3 The reference server — works, and is narrower than Section 6 reads

`serve/inference.py` + `serve/server.py` implement `POST /v1/systemone` against the trained checkpoint, tested end to end, not just written.

- **Measured latency, warm: 27–36 ms per call.** Above Section 7.3's <15 ms target for `ekvachan-base`, which is expected and not alarming — this is unoptimized eager-mode PyTorch in Python, i.e. precisely the thing 7.1 argues you don't ship for latency-sensitive serving. It's the pre-optimization baseline the eventual Rust/ONNX port has to beat, and the first real evidence that 7.1's "runtime choice is a first-order latency lever" claim will get tested rather than assumed.
- **Capability, stated exactly**: only the `choice` primitive, and only for the literal option set `["entailment", "neutral", "contradiction"]`. The classifier head never sees `options` text at training time, so it has no mechanism for answering an arbitrary option list — including Jev's own `["billing", "technical", "other"]` example from Section 1.2. The server returns **501** for any other option set rather than guessing, and `score`/`noul` return 501 too, since no model has been trained for them. See 10a.
- The response body's field names are a best-effort reconstruction from secondary sources. There is no live Jev API to diff against byte-for-byte, so 6.1's "byte-for-byte compatible" has been corrected to say what's actually verified.

### 13a.4 Consequences for the plan

1. The 3.1a/Q4 encoder-vs-decoder comparison will first resolve on **our own NLI split**, not on JevBench — the two arms share `eval/metrics.py`, so they're fairly comparable to each other, but neither has a JevBench number. 3.1a's recommendation (test both on JevBench before locking 5.1) is therefore still outstanding after this comparison lands.
2. Making the `choice` head schema-general — conditioning on `options` text rather than a fixed 3-way head — is now a concrete Phase 1 blocker for anything resembling Jev's actual API, not a later refinement. It is also exactly what 5.1a's cross-attention decision head architecturally provides (encode options, score by interaction), which promotes 5.1a from "nice differentiator" to "the likely path to a general `choice` primitive."
3. Section 9's compute plan held up at this scale, with one correction: its per-workload sizing assumed full fine-tuning throughput on a healthy software stack. The decoder run shows kernel availability, not just VRAM, can be the binding constraint. Worth checking fused-kernel availability before sizing any future run on a hybrid-architecture backbone.

### 13a.5 A first, narrow test of whether the decoder's schema-generality is real (2026-09-23)

13a.2/13a.4 identified the open question directly: the decoder's restricted-logit mechanism is architecturally *not* fixed-width (unlike the encoder's classification head), but had only ever been trained on one fixed 3-class schema. A same-night follow-up (`training/train_decoder_lora_multischema.py`, `training/build_multischema_slice.py`) built a small test of whether that architectural slack is real, ahead of committing to the much larger companion project (`better-jev-bench`) this finding also motivated.

**What was tested**: a LoRA-tuned Qwen3.5-4B, trained on a mix of the existing NLI data (3-way) and DBpedia-14 (CC-BY-SA — real 14-category ontology, each example presented as a random 4–8-way subset always containing the true label), then evaluated on **CLINC150 held out entirely** — zero examples from that dataset/schema appeared anywhere in training. A width-aware version of the letter-shuffle mechanism (extended from 3 to up to 10 options, unused letter-logit columns masked to exact `-inf` before softmax) makes this possible; `eval/metrics.py` works unchanged on the resulting ragged option-count eval set.

**Smoke test (320 train examples)**: 99.17% accuracy / 0.039 raw ECE on 120 held-out CLINC150 examples — promising, but n=120 is small.

**Full run (24,000 train examples, 1 epoch, 64m26s)** — the real result:

| Split | n | Accuracy | Raw ECE | Raw Brier |
|---|---|---|---|---|
| eval_id (in-distribution: held-out NLI + DBpedia-14) | 2,100 | 92.14% | 0.0131 | 0.111 |
| **eval_ood (CLINC150, zero training exposure)** | **3,000** | **98.50%** | **0.0339** | **0.0306** |

Not chance-level (~15–25% for 4–8-way options), not degenerate, and now statistically solid (n=3,000 on the held-out schema). One real, unsurprising texture in the in-distribution breakdown: the adversarially-constructed ANLI/WANLI splits score far weaker (46–81%) than SNLI/MNLI/DBpedia (89–99%) — expected, since ANLI/WANLI are specifically built to be hard, not evidence of a problem with this run.

**The load-bearing caveat, stated as precisely as the finding**: this is real, now well-powered evidence the *mechanism* works — variable option count, a genuinely unseen dataset, no crashes, sensible calibrated output — but it is **not yet evidence of accuracy on hard, wide, realistic classification**. Both the DBpedia-14 training schema and the CLINC150 held-out schema use the same random-4-to-8-option-subset-always-containing-the-answer construction, which is a substantially easier task than genuine 14-way or 151-way (150 intents + `oos`) classification — DBpedia's own in-distribution eval also scored 99.3% for the same reason. What's shown is that the masking/variable-width machinery is correct and the model generalizes cleanly to an unseen dataset under this task design; whether it holds up on genuine wide-schema classification (true 14-way, true 150-way, no subsetting) is a distinct, harder, not-yet-run test — the natural next experiment.

Feeds 13a.4 point 2 and Section 14 Q4 directly: if this holds up under a genuinely-hard follow-up (true wide-schema, no subsetting), it's evidence the decoder path may reach a general `choice` primitive through **training data alone**, without needing 5.1a's cross-attention head architecture change — a materially cheaper path if true. Not concluded yet — the easy-subset caveat above is exactly why. A true-wide-schema run (full-width DBpedia-14, wider CLINC150 subsets, still capped at 26 options — the single-letter restricted-logit mechanism's real ceiling) is in progress as of this writing. **The more decisive test after that isn't another synthetic construction — it's pointing the resulting checkpoint at the real, already-built `benchmarks/jevbench` and `benchmarks/jabr_v2` harnesses (§13a's own motivating finding was 0/231 and 0/944 answerable)**; that comparison is queued as the next step once the wide-schema run lands.

**Already computable, and computed, without waiting for any checkpoint**: the harness's schema filter was extended from exact-3-class matching to "any `choice` item with 2–26 options" (26 is the real ceiling of the single-letter restricted-logit mechanism, verified against Qwen3.5-4B's tokenizer, not assumed). Re-run against the same real, already-vendored JevBench (231 items) and jabr-v2 (944 items): **139/231 JevBench items and 387/944 jabr-v2 items are now schema-answerable** by a decoder/multischema-class model, up from 0/231 and 0/944 for the fixed-schema encoder — computed by counting option widths alone, no model required. The 26-option cap doesn't currently bind on either dataset; the widest real option set found in both is 6. This is real progress on G3 becoming testable, not G3 itself — accuracy on those 139/387 items is still unmeasured until the wide-schema checkpoint is run against them.

### 13a.6 The true wide-schema follow-up, without the easy-subset construction (2026-09-23)

The follow-up 13a.5 flagged: `training/build_wideschema_slice.py` / `training/train_decoder_lora_wideschema.py`, `MAX_OPTIONS=26` (A–Z), trained on **full-width DBpedia-14 (true 14-way, no subsetting)** mixed with the existing NLI data, evaluated zero-shot on **CLINC150 presented as genuinely wide 15–26-way subsets** — a different domain *and* a different option-count construction from training, so this run no longer shares the easy-subset artifact that qualified 13a.5.

| Split | n | Accuracy | Raw ECE | Calibrated ECE | Raw Brier |
|---|---|---|---|---|---|
| eval_id (in-distribution: NLI + full 14-way DBpedia-14) | 2,100 | 93.29% | 0.0234 | 0.0150 | 0.1095 |
| **eval_ood (CLINC150, wide 15–26-way, zero training exposure)** | **3,000** | **96.10%** | **0.0671** | **0.0438** | **0.0712** |

Zero-shot accuracy on the wider, harder, genuinely unseen schema (96.10%) came in *above* in-distribution accuracy (93.29%) — real evidence the mechanism generalizes across both domain and option-count, not an artifact of construction this time. Total run: 24,000 train examples, 1 epoch, 73m10s. One honest weak spot in the in-distribution breakdown: ANLI/WaNLI subsets score 50–82% with ECE up to 0.34 — natural-language-inference is genuinely harder for this model than topic/intent classification, a task-difficulty finding, not a schema-width one.

This resolves 13a.5's open caveat: the decoder path reaching a general `choice` primitive through training data alone, without 5.1a's cross-attention head, now has real (not easy-subset) support. Per 13a.5's own framing, the more decisive next test is not another synthetic construction — it's this checkpoint against the real `benchmarks/jevbench` (139/231 schema-answerable) and `benchmarks/jabr_v2` (387/944 schema-answerable) harnesses, run next.

### 13a.7 The decisive test: real accuracy on real, third-party JevBench and jabr-v2 items (2026-09-23)

13a.5's schema-answerability counts (139/231 JevBench, 387/944 jabr-v2) were real, but accuracy on those items was not yet measured. The multischema-aware harness backend (`benchmarks/common/backends.DecoderMultischemaBackend`, wired into `benchmarks/common/harness.py` alongside `filter_supported_multischema`) was already built and committed earlier the same night -- verified against `--backend decoder_multischema_mock` and `--selftest` (see the `*_mock_*` result manifests timestamped ~04:16-04:19 UTC) -- but never actually run against a real checkpoint, since `ekvachan-decoder-qwen-wideschema` (13a.6) didn't exist yet at that point. This entry records the first real run of that already-built integration.

The `ekvachan-decoder-qwen-wideschema` checkpoint (13a.6) -- trained on DBpedia-14 + NLI + wide CLINC150 only, **zero exposure to JevBench or jabr-v2 data** -- run against both real, vendored, third-party benchmarks:

| Benchmark | n (schema-answerable) | Accuracy | Brier | ECE |
|---|---|---|---|---|
| JevBench (231 public items) | 139 | **83.45%** | 0.2610 | 0.0855 |
| jabr-v2 (944 items, v1+v2) | 387 | **88.89%** | 0.1508 | 0.0397 |

Context, not a direct comparison (different item subsets and scoring protocols, stated explicitly rather than implied): Von reports 72.0% macro-accuracy on jabr-v2's full 869-case v2 suite; this run only attempted the schema-answerable items (2-26 options, gold label present in the options list), not the full suite, and used forced per-item argmax rather than Von's own scoring protocol.


### 13a.8 First noul/score training run: the real unlock for the 92/231 + 557/944 items (2026-09-23)

STATUS.md's 2026-09-23 review found the remaining unattempted JevBench/jabr-v2 items (92 and 557 respectively) are 100% `noul`/`score` type, not blocked by option width -- the cross-attention head (5.1a) does nothing for a scalar score or a yes/no proposition, so training these two missing primitives is the actual next step, not an architecture change. `training/build_primitives_slice.py` (BoolQ -> `noul`, CC-BY-SA-3.0; Sp1786 3-level sentiment -> `score`, Apache-2.0, both license-checked directly against the HF Hub API) + `training/train_decoder_lora_primitives.py` (forked from the wideschema trainer; `score` rows use a fixed ascending scale order instead of the per-example letter shuffle, since ordinal position must stay consistent for the softmax's neighboring-letter mass to mean "landed between levels" -- see that script's docstring) mixed both into the existing NLI/DBpedia-14 `choice` data and trained one epoch, 24k examples, 71m54s.

| Question type | n | Accuracy | Brier | ECE |
|---|---|---|---|---|
| `choice` (continuity/regression check) | 539 | 92.02% | 0.1103 | 0.0230 |
| **`noul`** (BoolQ, new) | 259 | **88.80%** | 0.1615 | 0.0443 |
| **`score`** (Sp1786 sentiment, 3-level, new) | 252 | **75.40%** | 0.3547 | 0.0658 |

**`choice` generalization survived the primitive mix**: the CLINC150 wide zero-shot-schema regression check (reused unchanged from 13a.6) scored 95.73% here vs. 13a.6's 96.10% -- a 0.37pp difference, indistinguishable from noise at this scale. Mixing in two new primitives did not meaningfully hurt what was already working.

**`noul` looks real on a first pass** -- 88.80% on a genuine yes/no reading-comprehension task, well above chance, with the second-best ECE in the table.

**`score` is a real, honest weak spot, not swept under**: 75.40% accuracy and by far the worst Brier (0.355 vs 0.11-0.16 for everything else) on only a 3-way ordinal task -- worse than `choice`'s untrained-from-scratch baseline ever was on a comparably-sized schema. Two credible, not-yet-disentangled explanations: (1) 3-level sentiment is semantically subtler than a well-defined encyclopedia category or a passage-grounded yes/no -- genuine task difficulty, not a mechanism failure; (2) this is the first run ever to withhold the letter shuffle for an entire question type, and the fixed-order design is unverified beyond "it ran without crashing." Both need a follow-up to distinguish -- not concluded here.

**Follow-up run, same session -- distinguished**: a confusion matrix over the 252 `score` test examples (`negative`/`neutral`/`positive`) settles which of the two explanations above is right. 96.8% of errors (60/62) are adjacent-level confusions (negative<->neutral, neutral<->positive); only 3.2% (2/62) skip straight from negative to positive. This is the error pattern of a *working* ordinal mechanism meeting genuine semantic ambiguity at class boundaries, not a broken fixed-letter-order design -- a structurally-broken mechanism would produce far more uniform confusion across all cell pairs, not this strong concentration on adjacent levels. Reinforcing: mean probability mass on the true label even for misclassified examples is 0.62-0.66 (not near-zero), meaning the model is landing *near* the right answer on its errors rather than confidently wrong -- exactly the "can land between levels" behavior PRD 1.2 specifies for `score`. **Conclusion: `score`'s 75.40% accuracy is genuine task difficulty (3-level sentiment boundaries are inherently ambiguous), not a mechanism defect.** No design change needed; more/wider `score` training data (13a.8's own stated next step) is still the right lever for improving the number itself.

**What this run does not establish, stated as precisely as 13a.5/13a.6 did for their own caveats**: `eval_id_raw_by_question_type` still scores `noul`/`score` via the existing argmax-over-letters metric, a proxy -- not `noul`'s real served float output or `score`'s probability-weighted scale position (PRD 1.2's "can land between levels"), neither of which is wired into `serve/`/`benchmarks/` yet. Evidence bundle: `checkpoints/ekvachan-decoder-qwen-primitives/manifest.json` (`results/ekvachan-decoder-qwen-primitives.train-manifest.json` once copied per dev-guidelines rule 10).

Same night, in parallel: `serve/server.py` now defaults to the decoder (`serve.inference.DecoderChoiceModel`, wrapping `benchmarks.common.backends.DecoderMultischemaBackend`) instead of the original fixed-schema encoder -- closing the gap between "decoder is primary" (14 Q4) and what the live server actually answered. `score`/`noul` still return 501 from the wire contract, for the reason directly above.

**Live-server verification, same session**: `serve/server.py` started for real (`uvicorn serve.server:app`), hit over real HTTP. Model load: 10.7s. Real requests all correct -- a 2-way noul-shaped question ("Yes", 99.5% confidence), a 3-way NLI question, a 14-way DBpedia-style question. **Measured warm latency: 87-109ms per call over real HTTP** (first call after load includes ~1.5s of CUDA/kernel warmup, excluded from the warm range) -- well above the encoder's measured 27-36ms (13a.3), as expected for an unoptimized-Python-served 4B model with the SSM/linear-attention fallback kernels (13a.2/13a.4) versus a much smaller encoder. `score` correctly returns 501 (verified, not assumed). This is the real number the eventual Rust/ONNX port (7.1) and quantization (7.2) have to improve, same role 13a.3's encoder number already plays for that tier.
The honest remainder, unchanged from 13a.5/13a.6: this is not `is_complete_benchmark_score` on either dataset (92/231 and 557/944 items are still out of reach -- `noul`/`score` types and >26-option `choice` items, per the schema-filter's own unsupported-reason counts), and a fair head-to-head against Von's own published number requires either running the full unfiltered suite (not possible for this architecture without 5.1a's cross-attention head or a multi-token option scheme) or getting Von's own per-item schema-answerable subset for a like-for-like comparison. Evidence bundles: `results/jevbench-decoder_multischema-20260923T084856Z.manifest.json`, `results/jabr_v2-decoder_multischema-20260923T085009Z.manifest.json`.

### 13a.9 Resolving the 13a.2 calibration anomaly: it was never a bug, temperature scaling has nothing to fix (2026-09-23)

13a.2 found temperature scaling makes the decoder's ECE/Brier worse, not better, and hypothesized a mismatch between the calibration fit's `log(softmax_probs)` "logit surrogate" and true pre-softmax logits. That hypothesis doesn't survive a direct check -- `log(softmax(z))` differs from the true logits `z` by exactly a per-row additive constant (`z - LSE(z)`), and softmax is provably invariant to per-row constant shifts, so scaling the surrogate by `1/T` and re-softmaxing is mathematically identical to scaling the true logits by `1/T`, for any `T`. That equivalence isn't approximate; it's exact except for the negligible distortion introduced by clipping exact-zero padded-option probabilities to `1e-12` before taking the log (bounded, and tiny at the `T` values actually fitted here).

An oracle sweep on `ekvachan-decoder-qwen-primitives`'s eval_id test split (n=1,050, diagnostic only -- fitting to the test set itself is never a legitimate procedure, only used here to find the ceiling) settles it directly: **T=1.0 (raw, no scaling) is already the global minimum ECE and Brier on this split.** Every value tried away from 1.0 in either direction makes both metrics monotonically worse (T=0.5: ECE 0.064; T=2.0: ECE 0.132; T=1.0: ECE 0.0227). The real explanation is simpler than 13a.2's hypothesis: **the decoder's raw output is already close to optimally calibrated**, so there is nothing for temperature scaling to correct -- any adjustment can only add distortion.

This also explains the earlier fitted temperatures (0.86-1.15 across different runs) without invoking a math bug: fitting on a small calibration slice (450-900 examples) is enough sample noise to pull the 1-D gradient-descent fit slightly off 1.0 in either direction, even though 1.0 is the true optimum. Confirmed directly: refitting on a 3.3x larger calibration pool (n=1,500 vs 450) moved the fitted temperature from 0.9273 closer to the true optimum, 0.9663 -- consistent with a sample-size/noise story, not a structural one.

**Consequence**: `benchmarks/common/backends.DecoderMultischemaBackend`'s existing `apply_temperature=False` default was already the right call, now for a stronger, verified reason ("raw is provably optimal," not merely "calibrated made 13a.2's one run worse"). No code changes are needed to fix this -- there was nothing broken in the mechanism, only an incomplete explanation for a real, already-correctly-handled observation. Leaving the manifest's `temperature`/`*_calibrated_report` fields in place for transparency (so future runs can keep checking this holds), but they should keep being read as "for reference only," per 13a.2's original guidance, now on firmer footing.

### 13a.10 First retrain on the real better-jev-bench corpus: noul/score improve substantially, choice's raw number needs a real caveat (2026-09-23)

The owner's 2026-09-23 resequencing put better-jev-bench first, on the reasoning that the model wasn't actually done training -- 13a.8's `noul`/`score` were real but narrow (one dataset each). `training/build_benchcorpus_slice.py` combined the real, already-verified `bjb export` output (55k public / 4.4k held-out, 11 tasks, CLINC150 deliberately excluded to preserve the zero-shot regression check) with the existing NLI/DBpedia-14 continuity anchor: 62,000 train examples across 17 sources (up from 13a.8's 24,000/4), trained 4h40m (`checkpoints/ekvachan-decoder-qwen-benchcorpus`).

| Metric | 13a.8 (narrow) | This run (wide) | Change |
|---|---|---|---|
| `noul` | 88.80% | **95.48%** | **+6.68pp** |
| `score` | 75.40% | **80.95%** | **+5.55pp** |
| `choice` (aggregate) | 92.02% | 83.03% | -8.99pp (see caveat below) |
| CLINC150 zero-shot (unchanged regression check) | 95.73% | **96.77%** | +1.04pp |

**`noul` and `score` both improved substantially -- exactly what 13a.8 named as the right lever ("more/wider data, not a design change").** `noul` now draws from two real sources (BoolQ plus CUAD/civil-comments-adjacent text via the bench corpus's broader domain mix); `score` still has only `civil_comments/toxicity_level` as its real ordinal source, but at far higher volume and diversity of underlying text than the prior single sentiment dataset.

**The `choice` aggregate drop is not a real regression, and reporting it without this caveat would be dishonest**: 13a.8's `choice` eval was only NLI + DBpedia-14 -- comparatively clean-cut tasks. This run's `choice` eval also includes genuinely hard, never-before-tested real-world domains: CFPB complaint categorization (59.02%, n=266) and GoEmotions (57.88% on 28-way emotion classification, n=273, a task even specialized emotion classifiers find hard). The tasks common to *both* runs tell the real story: `dbpedia_14_fullwidth_test` 99.63%, NLI splits in the same 70-95% range history already established for ANLI/WaNLI's known difficulty, and -- the cleanest apples-to-apples check available, since it is the exact same held-out zero-shot-schema evaluation reused unmodified since 13a.6 -- **CLINC150 zero-shot held steady and marginally improved (95.73% -> 96.77%)**. Choice generalization did not get worse; the benchmark got harder and more honest by including domains nobody had tested before.

**New, real per-domain findings, reported rather than smoothed over**: BANKING77 (95.83%), CUAD clause-detection (97.13% noul / 87.36% choice), MASSIVE intent+scenario (92-93%) all look strong on a first pass. CFPB product categorization (59.02%) and GoEmotions (57.88%) are real, honest weak points -- both are wide (10-way and 28-way respectively), semantically ambiguous multi-label-ish tasks even for purpose-built classifiers, and each had only ~5,000 training examples in this pass. Worth a follow-up investigation (more data, or a confusion-matrix check mirroring 13a.8's `score` diagnosis) before concluding they're a real ceiling rather than an undertrained slice.

Evidence: `results/ekvachan-decoder-qwen-benchcorpus.train-manifest.json`. Next per the resequenced roadmap: vision data, then nano, then the game harnesses against this checkpoint for the real Von comparison.
**Follow-up, same session -- CFPB and GoEmotions diagnosed**: a confusion-matrix check (mirroring 13a.8's `score` methodology) on both weak spots. CFPB's dominant confusions are "Debt collection" <-> "Credit reporting or other personal consumer reports" (23 of 109 errors) and "Credit card" <-> "Credit reporting" (14 of 109) -- genuine real-world category overlap in CFPB's own taxonomy (a credit-report dispute very often *is* the debt-collection or credit-card issue being reported; "Credit reporting" functions as a near-catch-all category). GoEmotions' errors overwhelmingly collapse toward "neutral" (`approval`->`neutral` 14x, `annoyance`->`neutral` 9x, `curiosity`->`neutral` 6x, `disappointment`->`neutral` 6x) -- a property GoEmotions' own original paper documents directly (low inter-annotator agreement specifically for subtle emotions vs. neutral). Strong, lexically-distinctive emotions score high (`gratitude` 100%, `love`/`amusement` 87.5%); subtle or negative-adjacent ones score near zero (`disapproval`, `caring`, `disappointment` all ~0%) -- matching the dataset's own documented weak points, not something specific to this model. Weaker supporting signal than `score`'s clean adjacent-confusion result (mean P(true) on errors is 0.12-0.17, above zero but not as reassuring as `score`'s 0.62-0.66), so this is read as *consistent with* genuine task/taxonomy difficulty rather than a fully closed case -- worth revisiting if a future retrain doesn't move these numbers.

---

## 14. Open questions

Resolved by the project owner on 2026-09-22:
1. ~~GPU provider preference~~ → **Self-hosted on the owner's own Linux server** (Section 9 rewritten accordingly). Batch sizing is no longer an open question — Phase 1 settled it empirically (13a): the encoder trains fine at the script's default, the decoder arm is stable only at batch 4 with gradient checkpointing. The GPU model/VRAM is still not recorded anywhere in this repo, which is worth fixing not for sizing but for reproducibility — every evidence bundle Section 8.2 commits to shipping should name the hardware its timings were measured on.
2. ~~Hugging Face org / model-family name~~ → **ekVachan** (renamed throughout this document; HF org/repo naming to be created at Phase 1 kickoff).
3. ~~License check~~ → **Done** (Section 7.4): `browser-use/jev-ultrafast`, `phyous/tsai-sc`, and `AbdelStark/heist-one` are all MIT. Clear to fork/vendor with attribution.

Still open — needs a decision before Phase 1's architecture work is considered locked:
4. Section 3.1a flags a real fork in the road: this PRD's core architecture bet (5.1, encoder+heads) versus the family that currently dominates JevBench's own leaderboard (small decoder LLMs read via restricted-logit scoring). Recommendation stands to test both on JevBench during Phase 1 before locking 5.1 in — confirm that's an acceptable use of Phase 1 time, or say now if you want to commit to encoder-only and skip the comparison.

   **Update 2026-09-23 (13a.2)**: both arms now have real numbers on our own NLI split, and the decoder wins decisively — 92.25% accuracy vs. 86.32% (+5.93pp), and its raw ECE (0.0232) beats the encoder's best, calibrated ECE (0.0344), while needing 20× less training data. This is real, direct evidence toward the decoder family, on this eval — **but still not on JevBench** (3.1a's original recommendation), and **not proof the decoder generalizes to arbitrary option sets** (13a.3: neither arm can today — the decoder's LoRA training used the same fixed 3-class schema as the encoder, just read differently at inference time; its restricted-logit mechanism could in principle support variable option sets, but that hasn't been trained or tested). Partially resolved: **on accuracy and calibration, on this narrow task, the decoder wins outright.** Not yet resolved: whether that holds on a genuinely diverse-schema task, and whether it holds on JevBench specifically. The cost-to-iterate axis flips from a caution into a genuine tension once data-efficiency is measured (13a.2's revised framing) — recommend not locking 5.1 in as final until at least a schema-diversity test exists, since that's now the load-bearing open question, not raw accuracy.

   **Decided, 2026-09-23**: going forward with the **decoder (Qwen3.5-4B LoRA) as the primary architecture**, not the encoder — the margin was decisive enough (accuracy, calibration, *and* data-efficiency, per above and 13a.5) to act on before every last hedge (JevBench, true wide-schema) is closed out. 5.1's encoder-first framing is superseded; the encoder run stays in 13a.1 as a real, useful baseline, not as the direction.

---

## Sources
Jev / TypeSafe: [typesafe.ai launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [latent.space](https://www.latent.space/p/ainews-jev-a-system-one-model-that) · [MarkTechPost](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/) · [DataCamp](https://www.datacamp.com/blog/system-one-models-jev) · [flaviocopes](https://flaviocopes.com/jev/) · [jevai.net](https://jevai.net/articles/what-is-system-one-jev/) · [comprehensive gist](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965) · [lilting.ch](https://lilting.ch/en/articles/typesafe-ai-jev-system-one-model) · [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/typesafe-ais-jev-offers-an-alternative-to-llms-that-claims-to-be-193x-faster-and-445x-cheaper-system-one-type-model-is-bespoke-for-probabilistic-decision-making) · [LangChain](https://www.langchain.com/blog/building-a-harness-with-jev) · [archerhume architecture reverse-engineering](https://archerhume.com/posts/jevs-architecture-unmasked/) · [nextbigfuture](https://www.nextbigfuture.com/2026/09/typesafe-ai-jev-is-a-transformational-unlock-for-ai-decisions-classifications-and-huge-speed-and-cost-savings-for-ai-software.html) · [docs.typesafe.ai/llms.txt](https://docs.typesafe.ai/llms.txt)

Demos: [browser-use/jev-ultrafast repo](https://github.com/browser-use/jev-ultrafast) · [explainx.ai](https://www.explainx.ai/blog/jev-ultrafast-browser-use-typesafe-2026) · [phyous/tsai-sc](https://github.com/phyous/tsai-sc) · [MindStudio computer-use demos](https://www.mindstudio.ai/blog/jev-computer-use-minecraft-robotics-demos)

Open alternatives: [Von](https://github.com/wfzyx/von) · [Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) · [Kev](https://github.com/jaredpalmer/kev) · [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) · [systemonemodels.org landscape](https://systemonemodels.org/examples/alternatives/) · [awesome-typesafe-jev](https://github.com/AbdelStark/awesome-typesafe-jev)

Merged from a parallel research pass (Section 3.1a, 5.3a, 5.3b, 8.1a, 8.1b): [JevBench harness](https://github.com/fstandhartinger/jevbench) · [JevBench leaderboard](https://benchmarkheaven.com/jev-models) · [SemIf](https://github.com/TheoLeeCJ/SemIf) · [djev](https://github.com/mmastrac/djev) · [reflex](https://github.com/kshetrajna12/reflex) · [reflex-qwen3.5-4b-lora weights](https://huggingface.co/kshetrajna12/reflex-qwen3.5-4b-lora) · [PlayJev](https://github.com/OmniJev/PlayJev) · [PlayJev weights](https://huggingface.co/OmniJev/PlayJev-0.8B)
