# ekvachan-decoder-qwen-fullcorpus — 0.8B

**Metadata record only. No weights are in this repository, and this checkpoint is not
published to Hugging Face.** See [Status](#status-read-this-first) below.

A LoRA adapter for [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B),
trained by `training/train_decoder_lora_general.py` on the full public corpus
(316,156 rows, 1 epoch, 14.4 h on one L40S). Every number below is read from
[`manifest.json`](manifest.json) in this directory or from a run manifest in
[`results/`](../../results) — nothing here is hand-entered.

## Status: read this first

- **The weights are not published.** The 25.6 MB `adapter_model.safetensors` is
  deliberately excluded from git by `.gitignore` (`checkpoints/`, `*.safetensors`) —
  this repo has never committed model weights, and this checkpoint does not change that.
- **This is not a new release.** The published 0.8B-family weights remain
  [`abhi6168/ekvachan-decoder-flash`](https://huggingface.co/abhi6168/ekvachan-decoder-flash).
  What lands here is the reproducible metadata: the training manifest, the adapter
  configuration, the tokenizer/processor configuration, and the results.
- **To load these weights you need `adapter_model.safetensors`**, which this repository
  does not contain. Everything else needed to reconstruct the run is here.
- The adapter adds **no new tokens** (`added_tokens_decoder` is empty), so the base
  model's tokenizer is used unchanged and `tokenizer.json` is not vendored here.

## Configuration

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` |
| Model class | `AutoModelForImageTextToText` (`Qwen3_5ForConditionalGeneration`) |
| Architecture | `decoder-lora-restricted-logit-general` |
| Total params | 859,375,680 |
| Trainable params | 6,389,760 (0.74%) |
| LoRA | r=16, alpha=32, dropout=0.05, 96 modules |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| Max options | 588 |
| Max length / max pixels | 1024 / 200,704 |
| Vision rows | yes (text + image mixed batches) |
| Init adapter | none (fresh LoRA) |
| Skipped OOM batches | 0 |
| Seed / epochs | 42 / 1 |
| Calibration temperature | 0.9407 (fit on the `eval_id` calib slice only, n=750) |
| Train size | 316,156 |
| Train wall-clock | 51,829 s (14.4 h) |
| Completed | 2026-09-28T00:57:25Z |

## Held-out results (in-training eval)

Two disjoint evaluations. `eval_id` is held-out rows from sources the adapter trained
on; `eval_ood` is the **zero-training-exposure** set, where no example or schema from
the evaluated source appeared anywhere in training.

| | n | Accuracy | Brier | ECE |
|---|---|---|---|---|
| `eval_id` (raw) | 1,751 | **87.09%** | 0.1856 | 0.0361 |
| `eval_id` (calibrated) | 1,751 | **87.09%** | 0.1842 | 0.0257 |
| `eval_ood` (raw) | 2,996 | **86.65%** | 0.1960 | 0.0330 |

Calibration moves Brier/ECE, not accuracy — the temperature is a single scalar and
cannot reorder a greedy argmax.

### Per-source, `eval_ood` — zero training exposure

| Source | Accuracy | n |
|---|---|---|
| `bjb:atari_head/action` | 100.00% | 85 |
| `bjb:os_atlas/target_element` | 97.81% | 319 |
| `bjb:massive/scenario` | 96.95% | 131 |
| `bjb:cuad/clause_present` | 96.88% | 128 |
| `bjb:civil_comments/is_toxic` | 94.91% | 373 |
| `bjb:clinc150/intent` | 94.18% | 189 |
| `bjb:massive/intent` | 90.84% | 131 |
| `bjb:cuad/clause_type` | 89.06% | 64 |
| `bjb:banking77/intent` | 85.44% | 64 |
| `bjb:cfpb_complaints/product` | 84.18% | 373 |
| `bjb:ledgar/provision_type` | 81.30% | 369 |
| `bjb:civil_comments/toxicity_level` | 80.70% | 373 |
| `bjb:go_emotions/emotion` | 66.20% | 358 |

### Per-source, `eval_id` — held out, but trained-on distribution

| Source | Accuracy | n |
|---|---|---|
| `bjb:atari_head/action` | 100.00% | 46 |
| `bjb:clinc150/intent` | 99.17% | 120 |
| `bjb:os_atlas/target_element` | 97.79% | 181 |
| `bjb:cuad/clause_present` | 97.06% | 68 |
| `bjb:banking77/intent` | 96.88% | 64 |
| `bjb:civil_comments/is_toxic` | 96.36% | 220 |
| `bjb:cuad/clause_type` | 92.31% | 39 |
| `bjb:massive/scenario` | 92.21% | 77 |
| `bjb:massive/intent` | 87.18% | 78 |
| `bjb:ledgar/provision_type` | 84.47% | 219 |
| `bjb:cfpb_complaints/product` | 83.25% | 209 |
| `bjb:civil_comments/toxicity_level` | 82.35% | 221 |
| `bjb:go_emotions/emotion` | 60.77% | 209 |

By question type (`eval_id`): `noul` 96.53% (n=288), `choice` 85.75% (n=1,242),
`score` 82.35% (n=221).

**The `eval_id` and `eval_ood` columns disagree per source, and that is the useful
signal, not noise.** CLINC150 is 99.17% in-distribution and 94.18% out — a 5 pp drop on
the same task with training exposure removed. `banking77` drops 96.88 → 85.44. Several of
these per-source columns are small (n<100), so they should not be read as precise; the
aggregate `eval_ood` figure of 86.65% over n=2,996 is the load-bearing number.

`go_emotions` (66.20% `eval_ood`, 28-way) and `cfpb_complaints` (84.18%, 10-way) remain
the weak points, consistent with the confusion analysis in PRD.md 13a.8.

## `bjb evaluate` — the headline per-source numbers

The properly-powered measurement. Served over HTTP against the real endpoint
(`fullcorpus-08b@full`), 25,233 items attempted, **25,233 answered, 0 out-of-schema,
0 declined** (coverage 1.0), 884.9 s. Evidence bundle:
[`better-jev-bench`](https://github.com/asp616848/better-jev-bench) →
`results_fullcorpus_08b/run_571d107c43724a4e9358/`.

| Task | Accuracy | n | Chance |
|---|---|---|---|
| `clinc150/intent` | **97.05%** | 2,000 | 0.66% |
| `cuad/clause_present` | **95.85%** | 2,000 | 50.0% |
| `civil_comments/is_toxic` | 95.35% | 2,000 | 92.1% |
| `massive/scenario` | **92.65%** | 2,000 | — |
| `banking77/intent` | **90.50%** | 2,000 | 1.30% |
| `os_atlas/target_element` | **88.25%** | 2,000 | — |
| `massive/intent` | **88.20%** | 2,000 | — |
| `cfpb_complaints/product` | **87.10%** | 2,000 | 54.1% |
| `cuad/clause_type` | **85.09%** | 1,174 | 2.44% |
| `ledgar/provision_type` | **82.45%** | 2,000 | 1.00% |
| `civil_comments/toxicity_level` | 81.25% | 2,000 | 79.3% |
| `go_emotions/emotion` | 61.95% | 2,000 | 35.3% |
| `screenspot_v2/target_element` | 55.83% | 858 | — |
| `atari_head/action` | 11.57% | 1,201 | 5.56% |

Axes: intelligence 64.37, calibration 89.45, generality 63.88, speed 76.36,
speed_multimodal 85.78, `score_no_cost` 72.80.

Two caveats the run's own manifest states, carried forward rather than dropped:
it marks itself **`provisional: true`**, and it flags that **finance is 1 of 11
scored datasets carrying 25% of Intelligence** (PRD §14.1) — so the aggregate is
domain-concentrated, not a balanced 11-domain mean.

### Where the two evaluations disagree, and why that matters

The `bjb evaluate` table and the training-time `eval_ood` table above measure
overlapping tasks with very different n, and **they disagree in both directions**:

| Task | `bjb evaluate` (n≈2,000) | training `eval_ood` (n=64–373) | Direction |
|---|---|---|---|
| `clinc150/intent` | 97.05% | 94.18% | bjb higher |
| `banking77/intent` | 90.50% | 85.44% | bjb higher |
| `cfpb_complaints/product` | 87.10% | 84.18% | bjb higher |
| `ledgar/provision_type` | 82.45% | 81.30% | bjb higher |
| `massive/intent` | 88.20% | 90.84% | bjb **lower** |
| `cuad/clause_type` | 85.09% | 89.06% | bjb **lower** |
| `os_atlas/target_element` | 88.25% | 97.81% | bjb much **lower** |
| `atari_head/action` | 11.57% | 100.00% | bjb **88× lower** |

The first four are explainable: this run trained on all 316,156 public rows, so
`bjb evaluate` over the public corpus is largely in-distribution and reads higher.

**The last row is not explainable that way, and it is the most important line in
this document.** `atari_head/action` scores 100.00% on the training-time held-out
slice (n=85) and 11.57% on `bjb evaluate` (n=1,201, chance 5.56%) — a gap far too
large to be sampling noise on either side. A 100% held-out score on a task the real
evaluation puts barely above chance indicates the training-time slice is not
measuring the same task the benchmark measures, or that those eval rows are near
duplicates of training rows. Until that is diagnosed, **the training-time per-source
tables should not be quoted as generalization evidence for the image-bearing
tasks** — `atari_head` and `screenspot_v2` (55.83%) are exactly the multimodal
claims, and they are the weakest measured numbers in the whole table.

## Real third-party benchmarks

Run against the actual production serving object (`--backend routing-decoder`), not a
separate model-loading path, on 2026-09-28. Manifests in `results/`:

| Benchmark | Coverage | Accuracy | Brier | ECE |
|---|---|---|---|---|
| JevBench | 231/231 | **53.68%** | 0.5961 | 0.1702 |
| jabr-v2 | 944/944 | **65.25%** | 0.4462 | 0.0386 |

| Embodied (ViZDoom, 8 episodes/seed, tics 4) | Rubric `none` | Rubric `von` |
|---|---|---|
| `defend_the_center` — kills, mean ± sd | 1.88 ± 0.78 | 1.88 ± 0.78 |
| `health_gathering` — survival s, mean ± sd | 12.11 ± 1.00 | 13.26 ± 2.04 |

**These scores are below the already-published 0.8B model.** ekVachan-flash measures
56.71% on JevBench and 67.37% on jabr-v2; this full-corpus run measures 53.68% and
65.25% — 3.03 pp and 2.12 pp lower. Training on the full corpus at 0.8B scale did not
improve the third-party benchmarks over the existing checkpoint, whatever the
in-training held-out numbers suggest. Treat this checkpoint as an experiment that
landed below its predecessor, not as a release candidate.

JevBench ECE of 0.170 is also markedly worse than the published model's; on a 231-item
set the model's confidence is poorly calibrated even where the argmax is often right.

## Latency

Measured locally on Apple MPS (`results/latency_08b_local.json`, 15 requests per point,
single stream, 5 options) — an indicative floor, not a GPU number:

| Output tokens | p50 | p95 |
|---|---|---|
| 131 | 352 ms | 385 ms |
| 512 | 984 ms | 986 ms |
| 1,025 | 1,891 ms | 2,082 ms |
| 4,099 | 7,661 ms | 8,138 ms |

Generation is ~1.85 ms per token at 1k context on MPS. The published flash model's
~24 ms p50 production figure is L40S + CUDA graphs; these numbers are not comparable to
it and should not be presented as a regression.

## Reproducing

```bash
# 1. build the slice from a `bjb export` (--export is required; defaults are the
#    0.8B / 1024-token / 200704-pixel settings this run used)
python training/build_fullcorpus_slice.py \
    --export <path-to-bjb-export> \
    --out-dir data/processed/fullcorpus_08b

# 2. train -- batch 2 x accum 32 for wide-row OOM headroom, as the run note records
python training/train_decoder_lora_general.py \
    --data-dir data/processed/fullcorpus_08b \
    --base-model Qwen/Qwen3.5-0.8B \
    --output-dir checkpoints-0.5b/ekvachan-decoder-qwen-fullcorpus \
    --batch-size 2 --grad-accum-steps 32 --epochs 1 \
    --max-length 1024 --max-pixels 200704 --seed 42 \
    --run-note "full-corpus 0.8B: 316156 train rows (all public, 1024 cap), fresh LoRA, 1 epoch, batch2x32 for wide-row OOM headroom"
```

The manifest carries the seed, slice sizes, LoRA geometry, and the calibration
temperature's fitting set, so a re-run is comparable. The training slice itself
(`data/processed/fullcorpus_08b`) is regenerable and is not committed.
