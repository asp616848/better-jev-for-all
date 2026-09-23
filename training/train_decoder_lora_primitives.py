"""
Fork of `train_decoder_lora_wideschema.py`, built to run
`training/build_primitives_slice.py`'s data -- the first training pass that
mixes `noul` and `score` primitives in alongside `choice`. See that data
builder's module docstring for the full rationale (STATUS.md's 2026-09-23
review found the 92/231 + 557/944 unattempted JevBench/jabr-v2 items are
100% `noul`/`score`, not option-width -- this is the actual unlock, not the
cross-attention head 5.1a was originally scoped for).

**Why fork again rather than edit `train_decoder_lora_wideschema.py` in
place**: same policy this project has applied at every step so far
(`train_decoder_lora.py` -> `_multischema` -> `_wideschema` -> this).
`train_decoder_lora_wideschema.py`'s output is the exact artifact behind
PRD.md 13a.6's published 96.10% zero-shot-schema number. Editing it in place
to change letter-ordering behavior for a new question_type would make that
already-published, already-reused-as-a-regression-check result silently
unreproducible from its own generating script.

**The one real, load-bearing mechanism change** (everything else below is
carried over byte-identical): `PromptDataset.__getitem__` no longer always
calls `shuffled_letter_order` -- rows with `question_type == "score"` use
the identity order (`list(range(n_options))`) instead, so a score's options
are always presented in the same fixed ascending-scale order the data
builder wrote them in. See `build_primitives_slice.py`'s docstring for why:
an ordinal scale's neighboring-letter probability mass is only meaningful as
"landed between levels" if letter position consistently tracks scale
position across every example. `choice` and `noul` rows are unaffected --
both still get the per-example shuffle, exactly as before, since neither
has an ordinal structure a fixed position could (correctly) exploit.

Nothing else changed: the masking/variable-width machinery, the OOM-skip
training loop, the two-held-out-eval-set design, the id-slice-only
temperature fitting, left-padding index logic -- all carried over unchanged
from `train_decoder_lora_wideschema.py`, which already proved these correct
at width 26.

**What this run does and does not establish** -- see `build_primitives_slice.py`'s
docstring for the full statement; repeated briefly here since it governs how
to read this script's own output: this reuses `eval/metrics.py`'s existing
accuracy/Brier/ECE (argmax against `label_idx`) as a first proxy for whether
`noul`/`score` CAN be learned through this mechanism at all. It does not
build or validate `noul`'s eventual float output or `score`'s
probability-weighted scale-position output -- that is `serve/`/`benchmarks/`
integration work, tracked separately.

Run: `uv run python3 -m training.train_decoder_lora_primitives --grad-checkpointing --batch-size 4 --grad-accum-steps 16`
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from eval.metrics import fit_temperature, full_report

REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_OPTIONS = 26  # A-Z -- unchanged from train_decoder_lora_wideschema.py
LETTERS = [chr(ord("A") + i) for i in range(MAX_OPTIONS)]


def shuffled_letter_order(pair_seed: str, n_options: int) -> list[int]:
    """Unchanged from train_decoder_lora_wideschema.py -- still used for choice/noul rows."""
    rng = np.random.default_rng(int(hashlib.sha1(pair_seed.encode()).hexdigest()[:8], 16))
    order = list(range(n_options))
    rng.shuffle(order)
    return order


def build_prompt(state: str, options: list[str], order: list[int], instructions: str) -> str:
    n = len(options)
    option_lines = "\n".join(f"{LETTERS[i]}) {options[order[i]]}" for i in range(n))
    return (
        f"{instructions}\n\n{state}\n\nOptions:\n{option_lines}\n\n"
        f"Answer with a single letter ({'/'.join(LETTERS[:n])})."
    )


class PromptDataset(TorchDataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.ds = hf_dataset
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        options = ex["options"]
        n_options = len(options)
        if not (2 <= n_options <= MAX_OPTIONS):
            raise ValueError(
                f"example at idx {idx} (source={ex.get('source')!r}) has {n_options} options, "
                f"outside the supported [2, {MAX_OPTIONS}] range -- this should have been caught "
                f"by build_primitives_slice.py's validation; refusing to silently truncate."
            )
        # THE one real change from train_decoder_lora_wideschema.py -- see module docstring.
        if ex.get("question_type") == "score":
            order = list(range(n_options))  # fixed ascending scale order, never shuffled
        else:
            order = shuffled_letter_order(ex["state"] + str(ex["label_idx"]) + str(n_options), n_options)
        prompt_text = build_prompt(ex["state"], options, order, ex["instructions"])
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        target_letter_idx = order.index(ex["label_idx"])
        return {
            "prompt": prompt,
            "target_letter_idx": target_letter_idx,
            "label_idx": ex["label_idx"],
            "n_options": n_options,
            "order": order,
            "source": ex.get("source", "unknown"),
            "question_type": ex.get("question_type", "choice"),
        }


def make_collate(tokenizer, max_length: int, letter_ids: list[int], train: bool):
    def collate(batch):
        prompts = [b["prompt"] for b in batch]
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        target_letter_idx = torch.tensor([b["target_letter_idx"] for b in batch], dtype=torch.long)
        label_idx = torch.tensor([b["label_idx"] for b in batch], dtype=torch.long)
        n_options = torch.tensor([b["n_options"] for b in batch], dtype=torch.long)

        order_padded = torch.full((len(batch), MAX_OPTIONS), -1, dtype=torch.long)
        for i, b in enumerate(batch):
            order_padded[i, : b["n_options"]] = torch.tensor(b["order"], dtype=torch.long)

        if train:
            labels = torch.full_like(enc["input_ids"], -100)
            last_col = enc["input_ids"].shape[1] - 1
            answer_token_ids = torch.tensor(letter_ids, dtype=torch.long)[target_letter_idx]
            labels[:, last_col] = answer_token_ids
            enc["labels"] = labels

        enc["target_letter_idx"] = target_letter_idx
        enc["label_idx"] = label_idx
        enc["n_options"] = n_options
        enc["order"] = order_padded
        enc["_sources"] = [b["source"] for b in batch]
        enc["_question_types"] = [b["question_type"] for b in batch]
        return enc

    return collate


def load_split(data_dir: Path, tokenizer, max_length: int, split: str, subset: int, seed: int) -> PromptDataset:
    ds = load_from_disk(str(data_dir / split))
    if subset:
        ds = ds.shuffle(seed=seed).select(range(min(subset, len(ds))))
    return PromptDataset(ds, tokenizer, max_length)


@torch.no_grad()
def evaluate(model, loader, device, letter_ids: list[int]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    all_probs, all_labels, all_sources, all_qtypes = [], [], [], []
    letter_ids_t = torch.tensor(letter_ids, device=device)
    for batch in loader:
        batch.pop("target_letter_idx")
        label_idx = batch.pop("label_idx")
        n_options = batch.pop("n_options")
        order = batch.pop("order")
        sources = batch.pop("_sources")
        qtypes = batch.pop("_question_types")
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)

        last_logits = out.logits[:, -1, :]
        restricted = last_logits[:, letter_ids_t]

        col_idx = torch.arange(MAX_OPTIONS, device=restricted.device).unsqueeze(0)
        pad_mask = col_idx >= n_options.to(restricted.device).unsqueeze(1)
        restricted = restricted.masked_fill(pad_mask, float("-inf"))
        restricted_probs = F.softmax(restricted.float(), dim=-1).cpu().numpy()

        order_np = order.numpy()
        n_options_np = n_options.numpy()
        canonical_probs = np.zeros_like(restricted_probs)
        for row in range(restricted_probs.shape[0]):
            n_i = int(n_options_np[row])
            valid_order = order_np[row, :n_i]
            inverse_perm = np.argsort(valid_order)
            canonical_probs[row, :n_i] = restricted_probs[row, :n_i][inverse_perm]

        all_probs.append(canonical_probs)
        all_labels.append(label_idx.numpy())
        all_sources.extend(sources)
        all_qtypes.extend(qtypes)
    return np.concatenate(all_probs), np.concatenate(all_labels), all_sources, all_qtypes


def _report_by_key(probs: np.ndarray, labels: np.ndarray, keys: list[str]) -> dict:
    by_key = {}
    keys_arr = np.array(keys)
    for k in sorted(set(keys)):
        m = keys_arr == k
        by_key[k] = full_report(probs[m], labels[m], MAX_OPTIONS)
    return by_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed" / "primitives_slice"))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-primitives"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--eval-id-subset", type=int, default=0)
    ap.add_argument("--eval-ood-subset", type=int, default=0)
    ap.add_argument("--calib-fraction", type=float, default=0.3)
    ap.add_argument("--grad-checkpointing", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1e9
        print(f"free VRAM right now: {free_gb:.1f} GB / {total_bytes/1e9:.1f} GB total")
        if free_gb < 10:
            raise RuntimeError(f"Only {free_gb:.1f} GB free -- refusing to start on a shared GPU this tight.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    letter_token_ids_full = [tokenizer.encode(l, add_special_tokens=False) for l in LETTERS]
    for l, ids in zip(LETTERS, letter_token_ids_full):
        assert len(ids) == 1, f"letter {l!r} is not a single token under {args.base_model}'s tokenizer: {ids}"
    letter_ids = [ids[0] for ids in letter_token_ids_full]
    assert len(set(letter_ids)) == len(letter_ids)
    print(f"letter token ids: {dict(zip(LETTERS, letter_ids))}")

    data_dir = Path(args.data_dir)
    train_pd = load_split(data_dir, tokenizer, args.max_length, "train", args.train_subset, args.seed)
    eval_id_full = load_split(data_dir, tokenizer, args.max_length, "eval_id", args.eval_id_subset, args.seed)
    eval_ood_pd = load_split(data_dir, tokenizer, args.max_length, "eval_ood", args.eval_ood_subset, args.seed)

    for name, pd_ds in [("train", train_pd), ("eval_id", eval_id_full), ("eval_ood", eval_ood_pd)]:
        widths = [len(o) for o in pd_ds.ds["options"]]
        assert min(widths) >= 2 and max(widths) <= MAX_OPTIONS, f"{name}: option-count range violates [2, {MAX_OPTIONS}]"

    n_calib = int(len(eval_id_full) * args.calib_fraction)
    calib_pd = Subset(eval_id_full, range(n_calib))
    test_id_pd = Subset(eval_id_full, range(n_calib, len(eval_id_full)))
    print(f"train: {len(train_pd)} | eval_id calib: {len(calib_pd)} | eval_id test: {len(test_id_pd)} | eval_ood: {len(eval_ood_pd)}")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_collate = make_collate(tokenizer, args.max_length, letter_ids, train=True)
    eval_collate = make_collate(tokenizer, args.max_length, letter_ids, train=False)
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
            batch.pop("target_letter_idx")
            batch.pop("label_idx")
            batch.pop("n_options")
            batch.pop("order")
            batch.pop("_sources")
            batch.pop("_question_types")
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

    id_raw_by_source = _report_by_key(id_probs, id_labels, id_sources)
    id_raw_by_qtype = _report_by_key(id_probs, id_labels, id_qtypes)
    ood_raw_by_source = _report_by_key(ood_probs, ood_labels, ood_sources)

    print("eval_id raw:", json.dumps(id_raw_report, indent=2))
    print("eval_id calibrated:", json.dumps(id_calibrated_report, indent=2))
    print("eval_id raw by source:", json.dumps(id_raw_by_source, indent=2))
    print("eval_id raw BY QUESTION_TYPE (choice/noul/score):", json.dumps(id_raw_by_qtype, indent=2))
    print("eval_ood raw (CLINC150 wide, choice-regression-check):", json.dumps(ood_raw_report, indent=2))
    print("eval_ood raw by source:", json.dumps(ood_raw_by_source, indent=2))

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")

    manifest = {
        "base_model": args.base_model,
        "architecture": "decoder-lora-restricted-logit-primitives",
        "max_options": MAX_OPTIONS,
        "seed": args.seed,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
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
        "eval_ood_note": "CLINC150 wide zero-shot-schema choice eval, reused unchanged from 13a.6 as a regression "
                          "check -- does mixing in noul/score hurt choice generalization?",
        "primitives_note": "First run mixing noul (BoolQ) and score (Sp1786 sentiment, 3-level) primitives into "
                            "training. eval_id_raw_by_question_type is the load-bearing new number -- accuracy/ECE/Brier "
                            "per primitive type, still measured via the existing choice-style argmax metric as a proxy "
                            "(see build_primitives_slice.py docstring for what this does and does not establish).",
        "skipped_oom_batches": skipped_oom,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_seconds": train_seconds,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
