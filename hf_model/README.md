---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: peft
tags:
  - lora
  - decision-model
  - jev
  - system-one
  - text-classification
pipeline_tag: text-classification
---

# ekVachan-decoder

LoRA adapters (r=16, alpha=32, 128 target modules) for
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), the decoder arm of
**ekVachan** -- an open, self-hostable alternative to TypeSafe AI's
[Jev System-One](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
decision model. Given a `state` and a set of typed `questions` (`choice`,
`score`, `noul`), it returns calibrated probabilities in one non-autoregressive
forward pass (a restricted-logit read over single-token letter codes), not a
generated response.

Full source, training pipeline, and evidence trail:
[github.com/asp616848/better-jev-for-all](https://github.com/asp616848/better-jev-for-all)
(`README.md` for the summary, `PRD.md` for every measured number and how it
was produced).

## What's in this repo

Two named LoRA adapters over the same base model, hot-swapped at serving time
by `serve.inference.RoutingDecoderModel`:

- `ekvachan-decoder-qwen-benchcorpus/` -- text-only, `max_options=26`
- `ekvachan-decoder-qwen-vision/` -- the strongest (`stage3`) checkpoint,
  vision-capable, `max_options=588`; also answers text-only requests by
  default in this project's reference server (`EKVACHAN_TEXT_ADAPTER=vision`)

Each subfolder carries its own `manifest.json` (training config, real
measured accuracy/Brier/ECE) alongside the adapter weights -- treat that file
as the source of truth over this card if they ever disagree.

## Real, verified numbers

Third-party benchmarks, zero training exposure, measured against the actual
serving class (`RoutingDecoderModel`, eager and CUDA-graph modes identical
within noise -- see PRD.md 13a.33):

| Benchmark | Accuracy |
|---|---|
| JevBench (231/231 items) | **70.99%** |
| jabr-v2 (944/944 items) | **85.49%** |

Latency, on our own dedicated reference server (CUDA graphs on by default,
PRD.md 13a.34): **~112ms p50 / ~131ms p95** on realistic (100-250 token)
request shapes; ~45ms on short (~15-token) synthetic prompts. This model is
not yet competitive with the fastest field entries (Von <18ms, Laya ~16ms)
on raw latency -- see the main repo's README for the full comparison and an
honest accounting of what CUDA graphs did and didn't fix.

## Usage

Not a plain `AutoModelForCausalLM.generate()` model -- it requires the
restricted-logit read mechanism (build a single-letter code-table prompt,
take the last token's logits, restrict to the option codes, softmax). Use
`serve.inference.RoutingDecoderModel` from the main repo directly rather than
reimplementing this:

```python
from serve.inference import RoutingDecoderModel

model = RoutingDecoderModel(checkpoints_dir="path/to/this/repo's/local/clone")
result = model.predict_choice(
    "The customer says the package never arrived.",
    ["refund", "replace", "escalate"],
)
print(result["choice"], result["probabilities"])
```

Or try it with zero setup in the
[ekVachan Space](https://huggingface.co/spaces/abhi6168/ekvachan) (note: Space
latency includes ZeroGPU queueing, not representative of the numbers above).

## License

Apache-2.0, same as the base model and the main repo.
