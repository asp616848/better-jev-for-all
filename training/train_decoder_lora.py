"""
Phase 1 decoder-variant training: LoRA-tune Qwen3.5-4B to read a restricted
set of next-token logits (the pattern SemIf/reflex use, per PRD.md Section
5.1/5.2's "test both" decision) instead of full autoregressive generation.

Key correctness details, verified against the real tokenizer/chat template
before being assumed (see commit history):
- Qwen3.5-4B's chat template defaults to inserting a `<think>...</think>`
  block; `enable_thinking=False` collapses it to empty so the very next
  token after the prompt is the actual answer position, not a reasoning
  chain. Without this the restricted-logit read would be reading the wrong
  position entirely.
- Raw label words ("entailment", "contradiction") are NOT single tokens in
  this tokenizer (3 tokens each) — only single letters (A/B/C) are, so
  options are presented as a lettered list, not the raw words.
- Option-letter <-> class mapping is shuffled per-example (seeded off the
  example's own pair hash, so it's deterministic but not fixed-position) to
  avoid the positional bias PRD.md Section 3.1 documents in Rizzo Flow
  ("confidently wrong on missing-evidence cases", "residual position bias
  despite shuffling") — if we always put entailment at "A", the model can
  learn a positional shortcut instead of reading content.

Uses the exact same eval/metrics.py as train_encoder.py so the two
architectures' numbers are directly, fairly comparable — that's the entire
point of the Phase 1 comparison.
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
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from eval.metrics import fit_temperature, full_report

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = ["entailment", "neutral", "contradiction"]
LETTERS = ["A", "B", "C"]

INSTRUCTIONS = (
    "Given a premise and a hypothesis, determine whether the hypothesis is "
    "entailed by the premise (must be true if the premise is true), neutral "
    "(could be true or false), or contradicted by the premise (cannot be true "
    "if the premise is true)."
)


def shuffled_letter_order(pair_seed: str) -> list[int]:
    """Deterministic per-example shuffle of which LABELS index sits at which letter position."""
    rng = np.random.default_rng(int(hashlib.sha1(pair_seed.encode()).hexdigest()[:8], 16))
    order = list(range(len(LABELS)))
    rng.shuffle(order)
    return order  # order[letter_idx] = label_idx shown at that letter


def build_prompt(state: str, order: list[int]) -> str:
    option_lines = "\n".join(f"{LETTERS[i]}) {LABELS[order[i]]}" for i in range(len(LABELS)))
    return (
        f"{INSTRUCTIONS}\n\n{state}\n\nOptions:\n{option_lines}\n\n"
        f"Answer with a single letter ({'/'.join(LETTERS)})."
    )


class PromptDataset(TorchDataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.ds = hf_dataset
        self.tok = tokenizer
        self.max_length = max_length
        self.letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        order = shuffled_letter_order(ex["state"] + str(ex["label_idx"]))
        prompt_text = build_prompt(ex["state"], order)
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        target_letter_idx = order.index(ex["label_idx"])  # which letter shows the true label
        return {
            "prompt": prompt,
            "target_letter_idx": target_letter_idx,  # index into LETTERS, 0/1/2
            "label_idx": ex["label_idx"],  # canonical class index, for eval reporting
            "order": order,  # order[letter_idx] = label_idx shown at that letter, for this example
        }


def make_collate(tokenizer, max_length: int, letter_ids: list[int], train: bool):
    def collate(batch):
        prompts = [b["prompt"] for b in batch]
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        target_letter_idx = torch.tensor([b["target_letter_idx"] for b in batch], dtype=torch.long)
        label_idx = torch.tensor([b["label_idx"] for b in batch], dtype=torch.long)
        order = torch.tensor([b["order"] for b in batch], dtype=torch.long)  # (B, n_labels)

        if train:
            # supervise exactly the answer-token position: labels = -100 everywhere except
            # the last real (non-pad) token position, where the target is the correct letter id.
            # Tokenizer uses left-padding (padding_side="left"), so every row's last real token
            # sits at the same fixed final column regardless of that row's own real length —
            # NOT at attention_mask.sum()-1, which is the right-padding indexing scheme.
            labels = torch.full_like(enc["input_ids"], -100)
            last_col = enc["input_ids"].shape[1] - 1
            answer_token_ids = torch.tensor(letter_ids, dtype=torch.long)[target_letter_idx]
            labels[:, last_col] = answer_token_ids
            enc["labels"] = labels

        enc["target_letter_idx"] = target_letter_idx
        enc["label_idx"] = label_idx
        enc["order"] = order
        return enc

    return collate


def load_split(data_dir: Path, tokenizer, max_length: int, split: str, subset: int, seed: int) -> PromptDataset:
    ds = load_from_disk(str(data_dir / split))
    if subset:
        ds = ds.shuffle(seed=seed).select(range(min(subset, len(ds))))
    return PromptDataset(ds, tokenizer, max_length)


@torch.no_grad()
def evaluate(model, loader, device, letter_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs over canonical LABELS order, canonical label indices)."""
    model.eval()
    all_probs, all_labels = [], []
    letter_ids_t = torch.tensor(letter_ids, device=device)
    for batch in loader:
        batch.pop("target_letter_idx")
        label_idx = batch.pop("label_idx")
        order = batch.pop("order")  # (B, n_labels): order[b, letter_idx] = label_idx shown at that letter
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)

        # Left-padding: every row's last real token is at the same fixed final column.
        last_logits = out.logits[:, -1, :]  # (B, vocab)
        restricted = last_logits[:, letter_ids_t]  # (B, n_labels), columns in LETTERS order
        restricted_probs = F.softmax(restricted.float(), dim=-1).cpu().numpy()  # (B, n_labels), letter-order

        # Remap letter-order probs -> canonical LABELS order: for row b, letter position
        # `letter_idx` held class `order[b, letter_idx]`, so canonical[label] = letter_probs[letter_idx]
        # wherever order[b, letter_idx] == label. Equivalently: canonical = letter_probs[inverse(order[b])].
        order_np = order.numpy()
        canonical_probs = np.zeros_like(restricted_probs)
        for row in range(restricted_probs.shape[0]):
            inverse_perm = np.argsort(order_np[row])  # label_idx -> letter_idx that displayed it
            canonical_probs[row] = restricted_probs[row][inverse_perm]

        all_probs.append(canonical_probs)
        all_labels.append(label_idx.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed" / "nli_slice"))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen"))
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
    ap.add_argument("--eval-subset", type=int, default=0)
    ap.add_argument("--calib-fraction", type=float, default=0.3)
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
            raise RuntimeError(
                f"Only {free_gb:.1f} GB free — a 4B model in bf16 needs ~8GB just for weights, "
                "refusing to start on a shared GPU this tight. Check nvidia-smi and retry."
            )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # so "last real token" is always the final position pre-generation

    letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    print(f"letter token ids: {dict(zip(LETTERS, letter_ids))}")

    data_dir = Path(args.data_dir)
    train_pd = load_split(data_dir, tokenizer, args.max_length, "train", args.train_subset, args.seed)
    eval_pd_full = load_split(data_dir, tokenizer, args.max_length, "eval", args.eval_subset, args.seed)

    n_calib = int(len(eval_pd_full) * args.calib_fraction)
    calib_pd = torch.utils.data.Subset(eval_pd_full, range(n_calib))
    test_pd = torch.utils.data.Subset(eval_pd_full, range(n_calib, len(eval_pd_full)))
    print(f"train: {len(train_pd)} | calib: {len(calib_pd)} | test: {len(test_pd)}")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_collate = make_collate(tokenizer, args.max_length, letter_ids, train=True)
    eval_collate = make_collate(tokenizer, args.max_length, letter_ids, train=False)
    train_loader = DataLoader(train_pd, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate, num_workers=2)
    calib_loader = DataLoader(calib_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)
    test_loader = DataLoader(test_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    optim_steps_per_epoch = -(-len(train_loader) // args.grad_accum_steps)
    total_steps = optim_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    start_time = time.time()
    step, skipped_oom = 0, 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch.pop("target_letter_idx")
            batch.pop("label_idx")
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
            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"epoch {epoch} step {step}/{total_steps} | loss {running_loss/(50*args.grad_accum_steps):.4f} | {elapsed:.0f}s")
                    running_loss = 0.0
        print(f"=== epoch {epoch} done, {time.time()-start_time:.0f}s elapsed ===")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_probs, calib_labels = evaluate(model, calib_loader, device, letter_ids)
    # temperature fitting expects logits, not probs — refit on log(probs) is not equivalent
    # to true logits, but since we've already restricted+softmaxed, we fit T on those 3-way
    # scores directly via their log as a logit surrogate (monotonic reparam of the same softmax).
    calib_logit_surrogate = np.log(np.clip(calib_probs, 1e-12, 1.0))
    temperature = fit_temperature(calib_logit_surrogate, calib_labels)
    print(f"fitted temperature: {temperature:.4f}")

    test_probs, test_labels = evaluate(model, test_loader, device, letter_ids)
    test_logit_surrogate = np.log(np.clip(test_probs, 1e-12, 1.0))
    calibrated_probs = F.softmax(torch.tensor(test_logit_surrogate) / temperature, dim=-1).numpy()

    raw_report = full_report(test_probs, test_labels, len(LABELS))
    calibrated_report = full_report(calibrated_probs, test_labels, len(LABELS))
    print("raw:", json.dumps(raw_report, indent=2))
    print("calibrated:", json.dumps(calibrated_report, indent=2))

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")

    manifest = {
        "base_model": args.base_model,
        "architecture": "decoder-lora-restricted-logit",
        "seed": args.seed,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_size": len(train_pd),
        "calib_size": len(calib_pd),
        "test_size": len(test_pd),
        "temperature": temperature,
        "raw_report": raw_report,
        "calibrated_report": calibrated_report,
        "skipped_oom_batches": skipped_oom,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_seconds": time.time() - start_time,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
