# PRD: Argus — an open, self-hostable, faster System One model

Status: draft v0.1 · Owner: Niyati Singh · Date: 2026-09-22
Working codename for the model family: **Argus** (all-seeing watcher — fits a multimodal judge model; trivially renameable, not load-bearing).

---

## 0. TL;DR

Jev (TypeSafe AI, launched 2026-09-15) is a closed, hosted "System One" model: given a `state` and a set of typed `questions` (`choice` / `score` / `noul`), it returns calibrated probabilities in one non-autoregressive forward pass instead of generating text. It's fast (70–500ms, ~100ms typical) and cheap ($0.042/M input tokens, output free) compared to routing the same decision through a frontier chat LLM, and TypeSafe has shown it wired into browser agents (browser-use), StarCraft, a stealth game (HEIST//ONE), Minecraft, a robot arm, and voice/hand-tracking computer control.

**Jev is not open. No weights, no self-hosting, waitlist-gated API.** That gap is real and worth closing.

**It is also not novel anymore.** In the week since launch, ~30+ open reproductions and alternatives have shipped (`Von`, `Rizzo Flow`, `Kev`, `Laya`, `open-alternative-jev`, `NanoJev`, etc.), several of which already beat Jev on latency (trivial — local beats a multi-tenant cloud API) and at least one (`Von`) beats Jev outright on published game-decision benchmarks (ViZDoom `Defend the Center`: 9.00 vs 5.62 kills) using nothing more exotic than a 395M-parameter encoder with a classification head.

So the honest framing for this project is **not** "be first to open-source a faster Jev" — that's already done, multiple times, by small teams over a weekend. The bar is: **build the best one**, prove it on the same public, reproducible evidence trail this ecosystem already uses (frozen benchmarks + game harnesses + signed evidence traces), and make it radically easier to adopt and extend than anything that exists today. Section 3 lays out exactly what "best" has to beat, with numbers.

Everything the user asked for is achievable. Nothing below requires a capability that doesn't exist. The one thing to be upfront about: **training a genuinely better model is an empirical claim you earn by running the benchmarks, not something a PRD can promise in advance.** This document sets concrete, falsifiable targets (Section 8) instead of guaranteeing "we will beat X."

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
| Source heavy GPU compute if worth it | ✅ Worth it, but "heavy" is relative — this problem is encoder-scale (0.1B–9B params), not frontier-LLM-scale. A handful of rented GPU-hours per experiment, not a cluster. See Section 9 | |
| Ship an agent-usable setup/access skill in its own folder | ✅ Straightforward — scaffolded in this repo at `skills/argus-setup/` | |
| "Prove it's better than Jev" as the main focus | ⚠️ Achievable on specific, named benchmarks; **not** a claim we can make true by writing a PRD | Section 8 defines exactly what "prove" means and the concrete numbers we're accountable to |

**Nothing here is impossible.** The only adjustment from the brief as stated: "better than jev" needs to be read as "better than jev **and** better than the current open-source state of the art (Von)," because the open SOTA has already cleared "beats Jev" on at least one headline benchmark. Aiming only at Jev would ship something already obsolete on day one.

---

## 3. Competitive landscape (as of 2026-09-22 — this space is < 1 week old and moving fast)

### 3.1 The open alternatives that already exist
33+ projects catalogued at [systemonemodels.org/examples/alternatives](https://systemonemodels.org/examples/alternatives/). The ones that matter:

| Project | Approach | Base model | License | Key numbers | Gaps |
|---|---|---|---|---|---|
| **Von** (`wfzyx/von`) | Fine-tuned bidirectional encoder + classification heads | ModernBERT-Large, 395M, pretrained on 2T tokens | Apache-2.0, weights on HF (`wfzyx/von-1.0`) | jabr-v2 (49-task, 869-case OOD benchmark): **72.0% macro-acc**. ViZDoom *Defend the Center*: **9.00 kills** vs Jev's 5.62 (+60%). ViZDoom *Health Gathering*: **12.11s** vs Jev's **13.03s** — Jev is actually slightly ahead here, worth being precise about. Latency: **sub-18ms** local. | Degrades without explicit rubric text in `instructions`; no multimodal input; single dense model, no cascade; training corpus (~290k examples) is text-only |
| **Rizzo Flow** | LLM logit-reading (prefill once, branch questions off shared KV-cache, read answer-letter logits A–Z, softmax) | Spark-X2.5 4B / 1.7B, Apache-2.0 | Apache-2.0 | 49–52ms p50/p95 (RTX 5060 Ti). 81.2–84.8% on own fixtures — **authors explicitly claim no superiority over their own "SemIf" baseline**. Jev-compatible `/v1/systemone` endpoint. | Max 26 options (Jev/Von support 255). Confidently wrong on 6/36 missing-evidence cases (SemIf: 1/36). Single-platform tested (Windows/CUDA only). No rate limiting, single resident model, requests serialize. |
| **Kev** (`jaredpalmer/kev`) | Same encoder-classification pattern | Qwen3.5-based, 0.8B/4B/9B | Apache-2.0 | ~160ms for 6-question batch on Apple Silicon; trains in ~1h45m on Apple Silicon | Smaller/slower than Von at comparable accuracy tier; less benchmark disclosure |
| **open-alternative-jev** (`ikermoel`) | Logit-reading over *any* open-weights LLM via HF or vLLM (needs single-token option labels + ChatML) | Tested Qwen 0.5B–27B | Apache-2.0 | Qwen3.6-27B-8bit: 73.7% acc / ECE 0.020 / 582ms/case on a community benchmark — beats a reported Jev-1.13.0 number (72.7% acc, but ECE only 0.020 vs Jev's much worse calibration, KL 0.27 vs 1.44) **on that specific benchmark only**. Packed mode: 2.5× cheaper but 6–9% answer drift below 4B params. | 582ms is *slower* than both Jev and Von — this approach trades latency for "works on anything you already run." Not competitive on speed. |
| **Laya** | Fine-tuned encoder, calibration-focused | ModernBERT-large, 421M | — | Only alt publishing ECE directly: 0.081 post temp-scaling. ~16ms. 2.2k★ | Smaller ecosystem, less benchmark breadth than Von |
| **NanoJev** | Trained from scratch | 0.6B custom | — | Ships full training pipeline + dataset (1.2k★) — most reproducible of the field | Smaller model, weaker headline numbers, but most useful as a *training reference* |
| Wrappers (openjev-sglang, Decider, litjev) | Same logit-reading idea on Qwen3.5 | Qwen3.5 2B–35B | — | "Not calibrated estimates of correctness" per landscape page | Interface-only reproductions, not real competitors on quality |

General-purpose typed-output tooling people compare against (not System-One-specific, but adjacent): **DSPy** (38k★), **Outlines** (16k★), **Instructor** (14k★) — these do grammar-constrained/validated structured output over any LLM, not calibrated probability readouts. Different tool, same "avoid free-text parsing" instinct.

### 3.2 What this means for us
1. **The floor is already high.** A weekend project (`Von`) beats Jev's own headline game benchmark. Don't scope "beat Jev" as the finish line.
2. **Nobody has shipped multimodal input.** Every alternative above is text-only. Jev itself hints at "possibly image inputs... yet…" in its launch post but hasn't shipped it. For the computer-use / game-playing use case specifically (DOM state *and* a frame, a robot's depth map, a game's minimap), this is a real, currently-empty gap.
3. **Nobody ships a cascade/two-tier architecture as a product.** TypeSafe's own docs mention "confidence-gated branching" as a *usage pattern* you build yourself; nobody bakes a nano-tier (sub-5ms, handles the easy 80%) + escalation-to-base-tier (handles the hard, low-confidence 20%) into the *model serving layer itself*. This is a serving-architecture win, not a training win — cheaper to build than a bigger model, and directly reduces mean latency below Von's 18ms without touching accuracy on easy cases.
4. **Fine-tuning is undersold everywhere.** Kev and NanoJev ship training code; none ship a guided, one-command "point this at your own labeled taxonomy and get a calibrated head back" pipeline. Given the user explicitly wants "faster... by fine-tuning," this is where real, defensible differentiation lives — not in yet another slightly-larger encoder.
5. **`rizzo-flow` specifically is not a high bar.** Its own README disclaims superiority over its own baseline. Beating it is close to a given; it's not the benchmark that matters, `Von` is.

---

## 4. Goals and non-goals

### 4.1 Goals
- G1 — **API-compatible** with Jev's `/v1/systemone` contract (drop-in swap by changing a base URL), plus a richer native API.
- G2 — **Self-hostable, open-weight**, no waitlist, no vendor lock-in. Apache-2.0 throughout (code + weights), matching ecosystem norm and maximizing adoption.
- G3 — Beat **Von's published numbers** (jabr-v2 macro-acc, ViZDoom kills/survival, latency) on our own reproduction of the same benchmarks, with signed evidence artifacts. If we don't beat Von on a given axis, say so in the README — no oversell.
- G4 — Ship a **multimodal (vision-capable) tier** for computer-use/game-state decisions — the one gap nobody else has filled.
- G5 — Ship a **cascade serving architecture** (nano → base escalation) that beats Von's flat sub-18ms on *average* latency across a realistic confidence distribution, without regressing hard-case accuracy.
- G6 — Ship a **one-command fine-tuning pipeline**: bring your own labeled examples, get a calibrated custom head back, with an eval report (accuracy + ECE) automatically generated.
- G7 — Reproduce or extend the **public game/computer-use harnesses** (ViZDoom, StarCraft Strongarm, browser-use flights task) with our model swapped in, evidence-traced, and license-checked.
- G8 — Ship an **agent-usable setup skill** so any AI coding agent (Claude Code, etc.) can install, run, benchmark, and fine-tune Argus with minimal human hand-holding.

### 4.2 Non-goals (explicitly out of scope for v1)
- We are **not** building a general chat/reasoning LLM. Same philosophy as Jev: "code calculates, Argus judges, a real LLM reasons and generates." Anyone needing open-ended text stays on their existing LLM.
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

### 5.2 Multi-tier model family (mirrors Kev/Von's tiering, sized against their published numbers)

| Tier | Params | Target latency (local, batch=1) | Role |
|---|---|---|---|
| `argus-nano` | ~0.3–0.5B (ModernBERT-base scale) | <10ms | First-pass tier in the cascade (G5); handles the confidently-easy majority of decisions |
| `argus-base` | ~0.4B (ModernBERT-large scale, matches Von's weight class for a fair head-to-head) | <18ms, target <15ms | Primary tier; this is the model we benchmark against Von directly |
| `argus-vision` | base tier + a vision encoder fused before the pooling head | <30ms target (vision encoding is the added cost) | The multimodal differentiator (G4) — DOM-screenshot + text state, game frames, robot camera input |
| `argus-large` (stretch, phase 3+) | 4–9B | <150ms | For the hardest cases the cascade escalates to; optional, only if benchmarking shows the cascade needs a third tier |

Rationale for sizing at the small end (not chasing Kev's 9B or an even bigger model): every disclosed benchmark in Section 3 shows near-Jev or better accuracy from *sub-1B* encoders. Size isn't the bottleneck in this problem class — training data quality and calibration method are. Spending compute on a bigger dense model is the lowest-leverage lever available; spending it on data and the cascade/vision work is higher-leverage. Revisit only if `argus-base` benchmarks show a real accuracy ceiling.

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

### 5.4 What we are *not* speculating on
We will not claim to have reverse-engineered or replicated Jev's actual internals. Section 1.1's architecture hypothesis is explicitly someone else's medium-confidence guess. Our architecture is our own documented decision (5.1–5.3), not an attempt to clone an unknown black box.

---

## 6. API & SDK design

### 6.1 Compatibility layer (copy Jev's shape, per the brief)
`POST /v1/systemone` — byte-for-byte compatible request/response shape with Jev (Section 1.2), so existing Jev/Rizzo-Flow/Von client code works by changing a base URL. This is the "as easy as possible for people to use" lever the brief asked for — zero migration cost for anyone already integrated with Jev's ecosystem (and there already are 12+ language SDKs and a dozen framework integrations built against this exact shape — Section "ecosystem" findings, not reproduced here for brevity but see research dossier).

### 6.2 Native API (richer than the compatibility shim)
- Adds a 4th primitive TypeSafe doesn't have: **`vision_choice`** — same as `choice` but `state` may include an image/region reference, backed by `argus-vision`.
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
| `argus-nano` (cascade first pass) | <10ms |
| `argus-base` (escalation / direct call) | <15ms |
| `argus-vision` (with image encode) | <30ms |
| Mean latency across a realistic confidence distribution (cascade) | target: beat Von's flat 18ms on **average**, not necessarily on every single call |

### 7.4 Reusing public harnesses — license check required before any code reuse
Before forking or vendoring code from `browser-use/jev-ultrafast`, `phyous/tsai-sc`, the HEIST//ONE repo, or Von's ViZDoom harness: confirm each repo's actual license file (not assumed from the org's general reputation) before redistributing any of their code inside this repo. Where a harness's license is unclear or restrictive, reimplement the *task definition* (same seeds, same win condition, same scoring) independently rather than copying code — task definitions (which seed, which map, what counts as a win) are not copyrightable in the way source code is.

---

## 8. Benchmarking & "proof" strategy — what "better" is accountable to

This is the section that makes G3/G7 falsifiable instead of a slogan.

### 8.1 Standing benchmarks we commit to running and publishing, win or lose
1. **jabr-v2** (49-task, 869-case OOD benchmark Von reports on) — reproduce Von's exact eval harness/seeds if publicly available; if not, use the same task categories and disclose the difference. Target: beat Von's 72.0% macro-accuracy. Publish our number regardless of outcome.
2. **ViZDoom** `Defend the Center` and `Health Gathering`, same 8 shared seeds Von used. Targets: beat Von's 9.00 kills *and* its 12.11s survival (note: Von itself trails Jev's 13.03s on survival — so "beat Jev" and "beat Von" are two different bars here; report both).
3. **StarCraft Strongarm** mission via `tsai-sc`'s harness (license permitting) or a faithful reimplementation — measure win rate and attempts-to-first-win, compared against the "attempt 16" figure reported for Jev.
4. **browser-use/jev-ultrafast task** (Zurich→London Google Flights, plus its Wikipedia and hotel-search variants) — swap Argus in for Jev via the compatibility API, no other code changes, measure wall-clock and CDP call count against the disclosed 7.07s / 101-call baseline.
5. **Calibration**: publish ECE (expected calibration error) the way Laya and Von do — this metric is currently a differentiator few alternatives report; we report it by default, always, not just when favorable.

### 8.2 Evidence discipline
Every benchmark run ships as a signed evidence bundle (raw outputs, seeds, prompt/state hashes, weight hash, timestamp) in `results/` — matching the norm this ecosystem has already converged on (Rizzo Flow's evidence traces, jabr-v2's "frozen benchmark" design, HEIST//ONE's evidence traces). This is non-negotiable for credibility in a field this benchmark-literate.

### 8.3 What "we proved it" means in practice
A claim like "Argus beats Jev at StarCraft" is only true once section 8.1's harness has actually run and the evidence bundle is in the repo. Until then, it's a target, and the README must say so.

---

## 9. Compute plan

**This is not a frontier-pretraining project.** Every disclosed competitor operates at 0.4B–9B parameters and reports training/fine-tuning times in the range of "under 2 hours on a single consumer GPU" (Kev: ~1h45m on Apple Silicon for its full family) to a few A100/H100-hours for larger runs. Concretely:

| Workload | Scale | Suggested compute | Rough time |
|---|---|---|---|
| `argus-nano`/`argus-base` fine-tune from ModernBERT checkpoint | 0.3–0.5B | 1× A100 80GB or 1× H100 (rented) | Hours, not days |
| `argus-vision` fusion training | base + small vision encoder | 1× A100/H100 | Low single-digit hours per iteration |
| `argus-large` (stretch tier, only if the cascade proves it's needed) | 4–9B | 1× H100, possibly multi-GPU for larger batch sizes | Under a day per run |
| Full benchmark suite (Section 8) | inference only | Can run on a single consumer GPU or even CPU for the smaller tiers | Hours |

**Recommendation**: rent by the hour from a spot-priced GPU marketplace (RunPod, Lambda, Vast.ai, or similar) rather than committing to a reserved cluster. Verify current spot rates before budgeting a number — GPU spot pricing moves fast enough that any figure in this PRD would be stale by the time you read it; get a live quote from whichever provider you pick at kickoff. Total spend for a full v1 (all four tiers, several training iterations, full benchmark suite) is realistically **low-hundreds to low-thousands of dollars** in rented GPU time, not a "source a cluster" undertaking. This is a solo/small-team-affordable project, which is itself worth stating plainly since the brief asked whether heavy GPU spend was warranted — it isn't, at this model scale.

---

## 10. Open-source strategy & licensing
- **Code**: Apache-2.0 (matches Von, Rizzo Flow, Kev, open-alternative-jev — the de facto norm in this exact ecosystem; maximizes commercial + academic adoption, which directly serves the "easy for anyone to use" goal).
- **Weights**: same, published on Hugging Face, all tiers, full-precision + quantized.
- **Training data**: publish sources/mix (Section 5.3) and any harness-derived data we generate ourselves; do not redistribute any dataset whose license forbids it — audit before publishing.
- **API compatibility posture**: stated explicitly in the README, mirroring Rizzo Flow's own disclaimer language — "reproduces Jev's interface pattern for interoperability; does not reproduce TypeSafe's proprietary architecture, weights, or RLCD training." This is both accurate and legally clean (thin JSON interfaces for interoperability are well-trodden ground, and three other projects already operate this exact way in the open without incident).

---

## 11. Agent skill (the "internal skill folder" ask)

Scaffolded at `skills/argus-setup/`. Purpose: let a user hand this repo to their own coding agent (Claude Code or similar) and have the agent self-serve setup, without the human reading install docs.

v1 skill capabilities (see `skills/argus-setup/SKILL.md` for the actual instructions):
1. **Install & serve** — clone, pull the right weight tier for the host's hardware, start the local server, verify with a smoke-test call.
2. **Compat-check** — point an existing Jev-integrated codebase at the local server and flag any request shape it doesn't yet support.
3. **Benchmark** — run the Section 8 suite locally and produce the evidence bundle.
4. **Fine-tune** — take a user-supplied labeled dataset, run the one-command pipeline (G6), report back accuracy + ECE.

This mirrors the fact that TypeSafe itself already ships "an official agent skill for Claude Code integration" for Jev — so this is table stakes for adoption in this ecosystem, not a nice-to-have.

---

## 12. Repo structure (as created)

```
better-jev-for-all/
  README.md              — short public-facing overview, points to this PRD
  PRD.md                 — this document
  LICENSE                — Apache-2.0
  skills/
    argus-setup/
      SKILL.md            — agent-facing setup/benchmark/fine-tune skill
  docs/                   — research notes, landscape tracking, benchmark write-ups (grows over time)
```

Code (`server/`, `training/`, `eval/`, SDKs, etc.) is intentionally **not** scaffolded yet — per the brief, this pass is research + decisions + repo setup, not implementation. Section 13 sequences the build.

---

## 13. Roadmap

- **Phase 0 (this PRD)**: research, decisions, repo + skill scaffold. ✅ this document.
- **Phase 1**: `argus-base` encoder fine-tune off ModernBERT-large, compatibility-layer server, jabr-v2 + ECE benchmark reproduction. Ship when we have a real, evidence-backed number to compare against Von — not before.
- **Phase 2**: `argus-nano` + cascade serving, ViZDoom + StarCraft harness integration, latency benchmark publication.
- **Phase 3**: `argus-vision`, browser-use/jev-ultrafast swap-in benchmark, fine-tuning pipeline (G6), full SDK (Python + TS).
- **Phase 4 (stretch)**: `argus-large` third tier, if and only if cascade benchmarking shows a real accuracy ceiling the first two tiers can't clear.

---

## 14. Open questions for you (need a decision before Phase 1 starts)
1. GPU provider preference for Phase 1 (RunPod / Lambda / Vast.ai / other) — no strong reason to prefer one from research; pick based on whichever you already have billing set up with.
2. Hugging Face org name for published weights (affects branding/URLs — e.g. is "argus" available, or do you want a different model-family name entirely).
3. Confirm license file check (Section 7.4) before we fork any of `browser-use/jev-ultrafast`, `tsai-sc`, or HEIST//ONE's code — want me to do that check now, or hold until Phase 2 when we actually need the harness?

---

## Sources
Jev / TypeSafe: [typesafe.ai launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [latent.space](https://www.latent.space/p/ainews-jev-a-system-one-model-that) · [MarkTechPost](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/) · [DataCamp](https://www.datacamp.com/blog/system-one-models-jev) · [flaviocopes](https://flaviocopes.com/jev/) · [jevai.net](https://jevai.net/articles/what-is-system-one-jev/) · [comprehensive gist](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965) · [lilting.ch](https://lilting.ch/en/articles/typesafe-ai-jev-system-one-model) · [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/typesafe-ais-jev-offers-an-alternative-to-llms-that-claims-to-be-193x-faster-and-445x-cheaper-system-one-type-model-is-bespoke-for-probabilistic-decision-making) · [LangChain](https://www.langchain.com/blog/building-a-harness-with-jev) · [archerhume architecture reverse-engineering](https://archerhume.com/posts/jevs-architecture-unmasked/) · [nextbigfuture](https://www.nextbigfuture.com/2026/09/typesafe-ai-jev-is-a-transformational-unlock-for-ai-decisions-classifications-and-huge-speed-and-cost-savings-for-ai-software.html) · [docs.typesafe.ai/llms.txt](https://docs.typesafe.ai/llms.txt)

Demos: [browser-use/jev-ultrafast repo](https://github.com/browser-use/jev-ultrafast) · [explainx.ai](https://www.explainx.ai/blog/jev-ultrafast-browser-use-typesafe-2026) · [phyous/tsai-sc](https://github.com/phyous/tsai-sc) · [MindStudio computer-use demos](https://www.mindstudio.ai/blog/jev-computer-use-minecraft-robotics-demos)

Open alternatives: [Von](https://github.com/wfzyx/von) · [Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) · [Kev](https://github.com/jaredpalmer/kev) · [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) · [systemonemodels.org landscape](https://systemonemodels.org/examples/alternatives/) · [awesome-typesafe-jev](https://github.com/AbdelStark/awesome-typesafe-jev)
