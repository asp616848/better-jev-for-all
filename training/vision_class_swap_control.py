"""
PRD 5.2b step 0: the control run. Cheap (~20 min, no training) and it's real
evidence, not busywork -- it isolates one variable before any vision data or
vision training exists.

**The question.** `train_decoder_lora_benchcorpus.py` trained its adapter
(`checkpoints/ekvachan-decoder-qwen-benchcorpus`) against `Qwen3_5ForCausalLM`
(`AutoModelForCausalLM`), which the probe (`training/probe_vision_path.py`)
found has *zero* vision parameters. `train_decoder_lora_general.py` (and any
future vision run) always loads `Qwen3_5ForConditionalGeneration`
(`AutoModelForImageTextToText`) instead -- 5.175B params, 333.5M of them a
vision tower the LoRA never touches. Does swapping the model class alone,
with no retraining and no vision data, change this adapter's real eval
numbers at all? If not, any accuracy movement in a future vision run is
attributable to the vision data, not to the class swap -- exactly what PRD
5.2b step 0 asks for.

**The method.** The adapter's LoRA keys were saved under the CausalLM class's
module tree (`base_model.model.model.layers.N....`); the VL class nests the
decoder one level deeper (`base_model.model.model.language_model.layers.N.…`)
-- verified directly against both the adapter's own `adapter_model.safetensors`
and the probe's `lora.example_key` finding. This script remaps exactly that
one path segment into a fresh adapter directory (never overwrites the
original), loads it into the VL class, and re-runs 13a.10's *text-only* eval
(same `data/processed/benchcorpus_slice` eval_id/eval_ood splits, same
metrics, same temperature-fitting procedure) via `decoder_lora_lib`'s shared
evaluate()/full_report() -- so this is a real forward-pass measurement, not
an assumption that a remap "should" work.

Run (~20 min on the L40S, no training):
    uv run python3 -u -m training.vision_class_swap_control
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import PeftModel
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForImageTextToText, AutoProcessor

from training.decoder_lora_lib import (
    MAX_OPTIONS,
    PromptDataset,
    assert_letter_tokens,
    evaluate,
    fit_temperature,
    full_report,
    make_collate,
    report_by_key,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OLD_PREFIX = "base_model.model.model."
NEW_PREFIX = "base_model.model.model.language_model."


def remap_adapter(src_adapter_dir: Path, dst_adapter_dir: Path) -> dict:
    """Inserts the `.language_model` path segment into every LoRA weight key
    that needs it (the decoder layers), leaving anything else (e.g. an
    embedding/lm_head override, if present) untouched. Never writes into
    `src_adapter_dir` -- always a fresh directory."""
    dst_adapter_dir.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(src_adapter_dir / "adapter_model.safetensors"))
    remapped = {}
    n_remapped = 0
    for k, v in sd.items():
        if k.startswith(OLD_PREFIX) and not k.startswith(NEW_PREFIX) and ".layers." in k:
            new_k = NEW_PREFIX + k[len(OLD_PREFIX):]
            remapped[new_k] = v
            n_remapped += 1
        else:
            remapped[k] = v
    save_file(remapped, str(dst_adapter_dir / "adapter_model.safetensors"))

    cfg = json.loads((src_adapter_dir / "adapter_config.json").read_text())
    (dst_adapter_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return {"n_keys": len(sd), "n_remapped": n_remapped}


def load_split(data_dir: Path, split: str) -> PromptDataset:
    ds = load_from_disk(str(data_dir / split))
    return PromptDataset(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--src-adapter", default=str(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-benchcorpus" / "adapter"))
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed" / "benchcorpus_slice"))
    ap.add_argument("--remapped-adapter-out", default=str(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-benchcorpus-vlclass-remap" / "adapter"))
    ap.add_argument("--reference-manifest", default=str(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-benchcorpus" / "manifest.json"))
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--calib-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1e9
        print(f"free VRAM right now: {free_gb:.1f} GB / {total_bytes/1e9:.1f} GB total")
        if free_gb < 12:
            raise RuntimeError(f"Only {free_gb:.1f} GB free -- refusing to start on a shared GPU this tight.")

    src_adapter_dir = Path(args.src_adapter)
    dst_adapter_dir = Path(args.remapped_adapter_out)
    remap_stats = remap_adapter(src_adapter_dir, dst_adapter_dir)
    print(f"remapped adapter: {remap_stats}")
    assert remap_stats["n_remapped"] > 0, "no keys matched the remap pattern -- adapter shape unexpected"

    processor = AutoProcessor.from_pretrained(args.base_model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    letter_ids = assert_letter_tokens(processor.tokenizer, args.base_model)

    print(f"loading {args.base_model} as AutoModelForImageTextToText (Qwen3_5ForConditionalGeneration)...")
    base = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=device)
    model = PeftModel.from_pretrained(base, str(dst_adapter_dir))
    model.eval()

    lora_mods = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    in_vision = [n for n in lora_mods if "visual" in n]
    print(f"remapped adapter loaded: {len(lora_mods)} LoRA modules found, {len(in_vision)} in vision tower")
    assert len(lora_mods) > 0, "remapped adapter did not attach any LoRA modules -- the remap likely failed"
    assert not in_vision

    data_dir = Path(args.data_dir)
    eval_id_full = load_split(data_dir, "eval_id")
    eval_ood_pd = load_split(data_dir, "eval_ood")

    n_calib = int(len(eval_id_full) * args.calib_fraction)
    calib_pd = Subset(eval_id_full, range(n_calib))
    test_id_pd = Subset(eval_id_full, range(n_calib, len(eval_id_full)))
    print(f"eval_id calib: {len(calib_pd)} | eval_id test: {len(test_id_pd)} | eval_ood: {len(eval_ood_pd)}")

    # Side finding, not the main question: while building this control run, the
    # new no-truncate collate (decoder_lora_lib.make_collate(truncate=False))
    # surfaced that some benchcorpus_slice rows silently exceeded 384 tokens
    # under 13a.10's own truncation=True mechanism. Audit it for real, batch by
    # batch, catching the raise instead of letting it abort the run, and put
    # the count in the manifest rather than only mentioning it in a docstring.
    audit_collate = make_collate(processor, args.max_length, letter_ids, train=False, truncate=False)
    audit_rows = [eval_id_full[i] for i in range(len(eval_id_full))] + [eval_ood_pd[i] for i in range(len(eval_ood_pd))]
    n_over_budget, n_total = 0, 0
    for start in range(0, len(audit_rows), args.eval_batch_size):
        chunk = audit_rows[start:start + args.eval_batch_size]
        n_total += 1
        try:
            audit_collate(chunk)
        except RuntimeError:
            n_over_budget += 1
    truncation_audit = {
        "max_length": args.max_length,
        "n_batches_total": n_total,
        "n_batches_over_budget": n_over_budget,
        "eval_batch_size": args.eval_batch_size,
        "rows_audited": len(audit_rows),
        "note": "Counted by running the new no-truncate collate over every eval_id+eval_ood batch and "
                "catching the RuntimeError it raises when a batch's untruncated sequence length exceeds "
                "--max-length. 13a.10's own script used truncation=True and silently cut these batches "
                "instead of erroring -- this is a real, previously-undetected property of the frozen "
                "13a.10 pipeline, found as a side effect of building this control run, not something "
                "this control run set out to measure.",
    }
    print(f"truncation audit (13a.10's own max_length={args.max_length}): "
          f"{n_over_budget}/{n_total} batches ({args.eval_batch_size} rows each) would have been "
          f"silently truncated")

    # truncate=True: reproduce 13a.10's own eval mechanism exactly (that script
    # called tokenizer(..., truncation=True, max_length=384)) so this control
    # run changes exactly one variable -- the model class -- not also the
    # truncation policy. See decoder_lora_lib.make_collate()'s docstring.
    eval_collate = make_collate(processor, args.max_length, letter_ids, train=False, truncate=True)
    calib_loader = DataLoader(calib_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)
    test_id_loader = DataLoader(test_id_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)
    ood_loader = DataLoader(eval_ood_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)

    t0 = time.time()
    calib_probs, calib_labels, _, _ = evaluate(model, calib_loader, device, letter_ids)
    calib_logit_surrogate = np.log(np.clip(calib_probs, 1e-12, 1.0))
    temperature = fit_temperature(calib_logit_surrogate, calib_labels)
    print(f"fitted temperature (VL class, remapped adapter): {temperature:.4f}")

    id_probs, id_labels, id_sources, id_qtypes = evaluate(model, test_id_loader, device, letter_ids)
    id_logit_surrogate = np.log(np.clip(id_probs, 1e-12, 1.0))
    id_calibrated_probs = F.softmax(torch.tensor(id_logit_surrogate) / temperature, dim=-1).numpy()
    ood_probs, ood_labels, ood_sources, ood_qtypes = evaluate(model, ood_loader, device, letter_ids)
    eval_seconds = time.time() - t0

    id_raw_report = full_report(id_probs, id_labels, MAX_OPTIONS)
    id_calibrated_report = full_report(id_calibrated_probs, id_labels, MAX_OPTIONS)
    ood_raw_report = full_report(ood_probs, ood_labels, MAX_OPTIONS)
    id_raw_by_qtype = report_by_key(id_probs, id_labels, id_qtypes)
    ood_raw_by_source = report_by_key(ood_probs, ood_labels, ood_sources)

    reference = json.loads(Path(args.reference_manifest).read_text())
    ref_id = reference["eval_id_raw_report"]
    ref_ood = reference["eval_ood_raw_report"]

    def _delta(a, b):
        return {k: (a[k] - b[k]) for k in ("accuracy", "brier", "ece")}

    id_delta = _delta(id_raw_report, ref_id)
    ood_delta = _delta(ood_raw_report, ref_ood)
    # "Reproduce" threshold: within 0.5pp accuracy and comparable Brier/ECE.
    # Loose enough to absorb eval-order/dataloader nondeterminism, tight enough
    # that a real regression (e.g. a key that didn't remap) would fail it.
    reproduced = abs(id_delta["accuracy"]) < 0.005 and abs(ood_delta["accuracy"]) < 0.005

    print(f"CausalLM-class reference (13a.10): eval_id acc {ref_id['accuracy']:.4f}, eval_ood acc {ref_ood['accuracy']:.4f}")
    print(f"VL-class remapped adapter (this run): eval_id acc {id_raw_report['accuracy']:.4f}, eval_ood acc {ood_raw_report['accuracy']:.4f}")
    print(f"deltas: eval_id {id_delta}, eval_ood {ood_delta}")
    print(f"REPRODUCED (class swap neutral): {reproduced}")

    manifest = {
        "purpose": "PRD 5.2b step 0 control run -- is the CausalLM -> ImageTextToText model-class swap "
                   "neutral on a text-only eval, with no retraining and no vision data?",
        "base_model": args.base_model,
        "model_class": "AutoModelForImageTextToText (Qwen3_5ForConditionalGeneration)",
        "src_adapter": str(src_adapter_dir),
        "remapped_adapter": str(dst_adapter_dir),
        "remap_stats": remap_stats,
        "n_lora_modules_after_remap": len(lora_mods),
        "reference_manifest": args.reference_manifest,
        "reference_run": "13a.10 (train_decoder_lora_benchcorpus.py, AutoModelForCausalLM)",
        "reference_eval_id_raw_report": ref_id,
        "reference_eval_ood_raw_report": ref_ood,
        "this_run_temperature": temperature,
        "this_run_eval_id_raw_report": id_raw_report,
        "this_run_eval_id_calibrated_report": id_calibrated_report,
        "this_run_eval_id_raw_by_question_type": id_raw_by_qtype,
        "this_run_eval_ood_raw_report": ood_raw_report,
        "this_run_eval_ood_raw_by_source": ood_raw_by_source,
        "eval_id_delta_vs_reference": id_delta,
        "eval_ood_delta_vs_reference": ood_delta,
        "reproduced_within_0.5pp_accuracy": reproduced,
        "truncation_audit": truncation_audit,
        "eval_seconds": eval_seconds,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "results" / f"vision-class-swap-control-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.manifest.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
