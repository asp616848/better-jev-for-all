"""
The general decoder-LoRA training entrypoint for `ekvachan-base` /
`ekvachan-vision` going forward. Replaces forking a sixth near-duplicate
script (`train_decoder_lora_vision.py`, as PRD 5.2b originally sketched it)
with one script that handles text-only and vision-bearing data through the
same code path, by explicit owner instruction on 2026-09-24: "try to keep
things general to vision or non-vision model for your changes."

**Why this is a genuine architecture decision, not just a rename.** The five
scripts before this one (`train_decoder_lora.py` -> `_multischema` ->
`_wideschema` -> `_primitives` -> `_benchcorpus`) are ~90% byte-identical --
each fork existed to keep an already-published result (13a.1-13a.10)
reproducible from its own frozen code while one real change was made. That
policy was right for scripts that were each proving a specific thing.
PRD 5.2b's probe closed the open question this lineage was working through:
Qwen3.5-4B's vision path is mechanically proven (config, chat template,
processor, LoRA target_modules, the restricted-logit read itself -- all
verified end to end, see `training/probe_vision_path.py`), so there is no
longer a "does this work" question left to fork over. What's left is a
software-engineering question -- text and vision data, one training script --
and forking a sixth copy to answer it would just be more of the same 90%
duplication the owner is asking to stop accumulating. The shared mechanism
(masking, OOM-skip, temperature fit, `_report_by_key`, the letter-token
assertion, left-padding index math) now lives in `training/decoder_lora_lib.py`;
this script is the orchestration layer on top of it.

**What "general" means concretely, four points**:

1. **One model class for both cases, always.** `AutoModelForImageTextToText`
   (-> `Qwen3_5ForConditionalGeneration`), never `AutoModelForCausalLM`. This
   is not a vision-only special case: PRD 5.2b measured the fixed cost of the
   VL class on a text-only request at 333.5M frozen vision-tower params
   (0.67 GB bf16, never touched -- a text-only row emits no `pixel_values` and
   the vision tower is frozen by LoRA target_modules name-mismatch, asserted
   in `decoder_lora_lib.assert_vision_tower_frozen()`). Paying that once,
   always, is cheaper and simpler than branching model classes depending on
   what's in a given data slice -- see PRD 5.2b's own reasoning for why
   `ekvachan-vision` is "a capability of the base tier, not a fourth model."
2. **One processor, not tokenizer-vs-processor branching.** `AutoProcessor`
   handles a text-only row identically to how `AutoTokenizer` did (verified in
   the probe: no `pixel_values` emitted, same input_ids for the same string).
3. **Auto-detected `max-length`/`max-pixels` defaults, not a vision flag the
   caller must remember.** If any row in the loaded train split carries an
   `images` field, defaults shift to PRD 5.2b's recommended
   `--max-length 768` / `max_pixels=256*28*28` (caps any screenshot at 180
   tokens) and the shared-GPU free-VRAM guard raises from 10 GB to 14 GB.
   Purely text data (no `images` column at all, or an empty one on every row --
   e.g. `data/processed/benchcorpus_slice`, unchanged) keeps the prior
   384/10 GB defaults. Every default is still a `--flag` the caller can
   override explicitly.
4. **No truncation, ever** -- `decoder_lora_lib.make_collate()` pads to the
   batch's longest row and raises loudly if that exceeds `--max-length`,
   rather than truncating silently. PRD 5.2b names silent truncation as "the
   failure mode most likely to waste a night of GPU time on this task."

**The regression check this script owes the project** (dev-guidelines rule 3
-- verify, don't assume): run this script against the exact same
`benchcorpus_slice` data `train_decoder_lora_benchcorpus.py` trains on, with
no vision rows present, and confirm the numbers land in the same neighborhood
as 13a.10's real published report. That run and its real manifest are
referenced from PRD 13a (search "13a.11") -- re-derive from the manifest
path there, don't trust this paragraph.

Everything else -- the shuffled-vs-ordinal option order, `labels[:,
last_col]` masking, the OOM-skip loop, temperature fitting on a calibration
split, `_report_by_key` -- is unchanged, imported from `decoder_lora_lib`.

Run (text-only data, e.g. the benchcorpus slice):
    uv run python3 -u -m training.train_decoder_lora_general \\
        --data-dir data/processed/benchcorpus_slice \\
        --output-dir checkpoints/<name> \\
        --grad-checkpointing --batch-size 4 --grad-accum-steps 16

Run (vision-bearing data, e.g. a slice from build_vision_slice.py):
    uv run python3 -u -m training.train_decoder_lora_general \\
        --data-dir data/processed/<vision_slice> \\
        --output-dir checkpoints/<name> \\
        --grad-checkpointing --batch-size 4 --grad-accum-steps 16
    # --max-length/--max-pixels/--min-free-vram-gb auto-shift to the vision
    # defaults the moment the loaded train split has any `images` rows.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForImageTextToText, AutoProcessor, get_linear_schedule_with_warmup

from training.decoder_lora_lib import (
    MAX_OPTIONS,
    PromptDataset,
    assert_vision_tower_frozen,
    build_code_table,
    evaluate,
    fit_temperature,
    full_report,
    make_collate,
    report_by_key,
    strip_batch_extras,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_ONLY_MAX_LENGTH = 384
TEXT_ONLY_MIN_FREE_VRAM_GB = 10
VISION_MAX_LENGTH = 768
VISION_MAX_PIXELS = 256 * 28 * 28  # PRD 5.2b: caps any screenshot at 180 image tokens
VISION_MIN_FREE_VRAM_GB = 24  # PRD 5.1b: measured peak 20.35 GB at width 151/batch 4; was 14 pre-5.1b


def _split_has_images(ds) -> bool:
    if "images" not in ds.column_names:
        return False
    # A column scan, not a full materialize -- cheap even at 62k rows, and it's
    # the one honest way to answer "does this slice carry any vision rows" without
    # trusting a filename or a flag the caller might forget to pass.
    return any(bool(row) for row in ds["images"])


def load_raw_split(data_dir: Path, split: str, subset: int, seed: int):
    """Loads the raw HF dataset only -- codes aren't known yet at this point
    (they need a live tokenizer), so `PromptDataset` wrapping happens later,
    once `build_code_table()` has run."""
    ds = load_from_disk(str(data_dir / split))
    if subset:
        ds = ds.shuffle(seed=seed).select(range(min(subset, len(ds))))
    return ds, _split_has_images(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data-dir", required=True, help="a slice dir with train/eval_id/eval_ood subdirs")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=None, help="default: 384 text-only, 768 if the "
                     "loaded train split carries any vision rows (auto-detected)")
    ap.add_argument("--max-pixels", type=int, default=None, help="AutoProcessor cap on image area; "
                     "default: unset for text-only data, 256*28*28 if vision rows are present")
    ap.add_argument("--min-free-vram-gb", type=float, default=None, help="default: 10 text-only, 24 vision "
                     "(PRD 5.1b: raised from 14 -- measured peak 20.35 GB at width 151/batch 4)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--eval-id-subset", type=int, default=0)
    ap.add_argument("--eval-ood-subset", type=int, default=0)
    ap.add_argument("--calib-fraction", type=float, default=0.3)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--truncate", action="store_true", help="reproduce the five frozen scripts' old "
                     "truncation=True mechanism instead of this script's own no-truncate-fail-loud "
                     "default. Exists for exactly one caller: a regression check that has to hold the "
                     "truncation policy fixed so the only variable that changes is the refactor itself.")
    ap.add_argument("--init-adapter", default=None, help="path to an existing checkpoint dir (with an adapter/ subdir) to continue-train from, instead of a fresh LoRA off --base-model. PRD 5.1b continue-train fallback: reuses an already-trained adapter's weights as the starting point rather than retraining from scratch.")
    ap.add_argument("--run-note", default="", help="free-text note written into the manifest, e.g. "
                     "'text-only regression check against 13a.10'")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    train_ds, train_has_vision = load_raw_split(data_dir, "train", args.train_subset, args.seed)
    eval_id_ds, eval_id_has_vision = load_raw_split(data_dir, "eval_id", args.eval_id_subset, args.seed)
    eval_ood_ds, eval_ood_has_vision = load_raw_split(data_dir, "eval_ood", args.eval_ood_subset, args.seed)
    has_vision = train_has_vision or eval_id_has_vision or eval_ood_has_vision

    max_length = args.max_length or (VISION_MAX_LENGTH if has_vision else TEXT_ONLY_MAX_LENGTH)
    max_pixels = args.max_pixels or (VISION_MAX_PIXELS if has_vision else None)
    min_free_vram_gb = args.min_free_vram_gb or (VISION_MIN_FREE_VRAM_GB if has_vision else TEXT_ONLY_MIN_FREE_VRAM_GB)
    print(f"auto-detected vision rows: {has_vision} -- max_length={max_length}, "
          f"max_pixels={max_pixels}, min_free_vram_gb={min_free_vram_gb}")

    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1e9
        print(f"free VRAM right now: {free_gb:.1f} GB / {total_bytes/1e9:.1f} GB total")
        if free_gb < min_free_vram_gb:
            raise RuntimeError(f"Only {free_gb:.1f} GB free -- refusing to start on a shared GPU this tight "
                                f"(guard: {min_free_vram_gb} GB).")

    processor_kwargs = {"max_pixels": max_pixels} if max_pixels else {}
    processor = AutoProcessor.from_pretrained(args.base_model, **processor_kwargs)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    codes, letter_ids = build_code_table(processor.tokenizer)
    print(f"code table: {len(codes)} codes, e.g. {dict(zip(codes[:3], letter_ids[:3]))} ... "
          f"{dict(zip(codes[-3:], letter_ids[-3:]))}")

    train_pd = PromptDataset(train_ds, codes)
    eval_id_full = PromptDataset(eval_id_ds, codes)
    eval_ood_pd = PromptDataset(eval_ood_ds, codes)

    for name, pd_ds in [("train", train_pd), ("eval_id", eval_id_full), ("eval_ood", eval_ood_pd)]:
        widths = [len(o) for o in pd_ds.ds["options"]]
        assert min(widths) >= 2 and max(widths) <= MAX_OPTIONS, f"{name}: option-count range violates [2, {MAX_OPTIONS}]"

    n_calib = int(len(eval_id_full) * args.calib_fraction)
    calib_pd = Subset(eval_id_full, range(n_calib))
    test_id_pd = Subset(eval_id_full, range(n_calib, len(eval_id_full)))
    print(f"train: {len(train_pd)} | eval_id calib: {len(calib_pd)} | eval_id test: {len(test_id_pd)} | "
          f"eval_ood: {len(eval_ood_pd)}")

    model = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    if args.init_adapter:
        init_adapter_dir = Path(args.init_adapter) / "adapter"
        assert init_adapter_dir.exists(), f"--init-adapter {args.init_adapter}: no adapter/ subdir found"
        model = PeftModel.from_pretrained(model, str(init_adapter_dir), is_trainable=True)
        print(f"continue-training from existing adapter: {init_adapter_dir}")
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    lora_mods = assert_vision_tower_frozen(model)
    print(f"vision-tower-freeze assertion passed: {len(lora_mods)} LoRA modules, none in model.visual")

    train_collate = make_collate(processor, max_length, letter_ids, train=True, truncate=args.truncate)
    eval_collate = make_collate(processor, max_length, letter_ids, train=False, truncate=args.truncate)
    train_loader = DataLoader(train_pd, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate, num_workers=2)
    calib_loader = DataLoader(calib_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)
    test_id_loader = DataLoader(test_id_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)
    ood_loader = DataLoader(eval_ood_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    optim_steps_per_epoch = -(-len(train_loader) // args.grad_accum_steps)
    total_steps = optim_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    start_time = time.time()
    step, skipped_oom = 0, 0
    running_loss, running_loss_count = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            strip_batch_extras(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                out = model(**batch)
                (out.loss / args.grad_accum_steps).backward()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                print(f"WARNING: OOM on batch {i}, skipping (total: {skipped_oom})")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            running_loss += out.loss.item()
            running_loss_count += 1
            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 50 == 0 or step == 1 or step == total_steps:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / max(1, running_loss_count)
                    print(f"epoch {epoch} step {step}/{total_steps} | loss {avg_loss:.4f} | {elapsed:.0f}s")
                    running_loss, running_loss_count = 0.0, 0
                if step % 25 == 0:
                    # unattended-overnight safety net: an unfinished run should still leave a
                    # usable adapter behind. Separate dir from the final output -- this one
                    # skips calibration/eval, it's a recovery artifact, not the validated result.
                    inprogress_dir = Path(args.output_dir) / "adapter_inprogress"
                    inprogress_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(inprogress_dir))
                    processor.save_pretrained(str(inprogress_dir))
                    print(f"[periodic checkpoint] saved adapter_inprogress at step {step}/{total_steps}")
        print(f"=== epoch {epoch} done, {time.time()-start_time:.0f}s elapsed ===")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_seconds = time.time() - start_time

    calib_probs, calib_labels, _, _ = evaluate(model, calib_loader, device, letter_ids)
    calib_logit_surrogate = np.log(np.clip(calib_probs, 1e-12, 1.0))
    temperature = fit_temperature(calib_logit_surrogate, calib_labels)
    print(f"fitted temperature (on eval_id calib slice only): {temperature:.4f}")

    id_probs, id_labels, id_sources, id_qtypes = evaluate(model, test_id_loader, device, letter_ids)
    id_logit_surrogate = np.log(np.clip(id_probs, 1e-12, 1.0))
    id_calibrated_probs = F.softmax(torch.tensor(id_logit_surrogate) / temperature, dim=-1).numpy()

    ood_probs, ood_labels, ood_sources, ood_qtypes = evaluate(model, ood_loader, device, letter_ids)

    id_raw_report = full_report(id_probs, id_labels, MAX_OPTIONS)
    id_calibrated_report = full_report(id_calibrated_probs, id_labels, MAX_OPTIONS)
    ood_raw_report = full_report(ood_probs, ood_labels, MAX_OPTIONS)

    id_raw_by_source = report_by_key(id_probs, id_labels, id_sources)
    id_raw_by_qtype = report_by_key(id_probs, id_labels, id_qtypes)
    ood_raw_by_source = report_by_key(ood_probs, ood_labels, ood_sources)

    print("eval_id raw:", json.dumps(id_raw_report, indent=2))
    print("eval_id calibrated:", json.dumps(id_calibrated_report, indent=2))
    print("eval_id raw by source:", json.dumps(id_raw_by_source, indent=2))
    print("eval_id raw BY QUESTION_TYPE (choice/noul/score):", json.dumps(id_raw_by_qtype, indent=2))
    print("eval_ood raw (CLINC150 wide, choice-regression-check):", json.dumps(ood_raw_report, indent=2))
    print("eval_ood raw by source:", json.dumps(ood_raw_by_source, indent=2))

    model.save_pretrained(output_dir / "adapter")
    processor.save_pretrained(output_dir / "adapter")

    manifest = {
        "base_model": args.base_model,
        "architecture": "decoder-lora-restricted-logit-general",
        "script": "training/train_decoder_lora_general.py",
        "model_class": "AutoModelForImageTextToText (Qwen3_5ForConditionalGeneration)",
        "vision_rows_detected": has_vision,
        "max_options": MAX_OPTIONS,
        "max_length": max_length,
        "max_pixels": max_pixels,
        "truncate": args.truncate,
        "min_free_vram_gb": min_free_vram_gb,
        "seed": args.seed,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "n_lora_modules": len(lora_mods),
        "train_size": len(train_pd),
        "calib_size": len(calib_pd),
        "eval_id_test_size": len(test_id_pd),
        "eval_ood_size": len(eval_ood_pd),
        "temperature": temperature,
        "temperature_fit_on": "eval_id calib slice only (not eval_ood)",
        "eval_id_raw_report": id_raw_report,
        "eval_id_calibrated_report": id_calibrated_report,
        "eval_id_raw_by_source": id_raw_by_source,
        "eval_id_raw_by_question_type": id_raw_by_qtype,
        "eval_ood_raw_report": ood_raw_report,
        "eval_ood_raw_by_source": ood_raw_by_source,
        "eval_ood_note": "CLINC150 wide zero-shot-schema choice eval -- regression check: does this data mix "
                          "hurt choice generalization relative to the prior lineage's own numbers?",
        "run_note": args.run_note,
        "init_adapter": args.init_adapter,
        "skipped_oom_batches": skipped_oom,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_seconds": train_seconds,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
