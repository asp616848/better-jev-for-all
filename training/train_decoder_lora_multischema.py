"""
Width-aware fork of `train_decoder_lora.py`, built to answer a real open
question from PRD.md Section 14 Q4 (2026-09-23 update): the decoder's
restricted-logit read mechanism -- read next-token logits at a few
designated "letter" positions, softmax over just those -- is architecturally
NOT fixed-width in the way the encoder's classification head is. But it has
only ever been *trained* on one fixed 3-class schema. This script trains it
on a genuinely mixed-schema, variable-option-count dataset
(`training/build_multischema_slice.py`'s output) to find out whether that
architectural slack is real or theoretical.

**Why a fork, not an in-place edit of `train_decoder_lora.py`**: that
script's output is the exact artifact backing PRD.md Section 13a.2's
published, reviewed numbers (92.25% accuracy / 0.0232 raw ECE on the fixed
3-way NLI comparison). Mutating its letter-index/position-bias logic in
place -- the same logic this docstring is about to extend -- risks quietly
changing what that already-published number was measured on, or
introducing a regression that's hard to attribute. Forking keeps the
validated 3-way script frozen and byte-for-byte reproducible, and isolates
the new variable-width data contract (per-example `options`, a padded/
masked restricted-softmax, no shared global `LABELS`) so a bug in the new
path can't silently touch the trusted one. The two scripts are meant to
diverge further over time, not be reconciled back into one.

**What's actually different from the original, and why each change is
safe** (read this before trusting any change to the letter-index math --
that logic took real debugging effort to get right the first time, per the
original script's own docstring):

1. No more global `LABELS`/module-level 3-way assumption. Every example
   carries its own `options: list[str]` (2-10 entries) and `label_idx`
   (index into that example's own list) -- exactly the shape
   `training/data.py` already writes, so this is a drop-in extension of an
   existing convention, not a new one.

2. `MAX_OPTIONS = 10` is a hard width cap, not a suggestion. It exists
   because: (a) `LETTERS` must stay single uppercase letters to stay
   single-token in Qwen3.5-4B's tokenizer -- verified directly on this
   server before writing this file: `A` through `J` (10 letters) each
   encode to exactly one, mutually distinct token id. Going past `J` was
   NOT verified and this script will refuse to start if any example
   exceeds the cap (see the assertion in `main()`) rather than silently
   reading a wrong or colliding letter token. Extending past 10 (e.g. to
   BANKING77's 77-way) needs a real answer to "what identifies option #11
   as a single token" (two-letter codes? digits? a different read
   mechanism entirely?) -- a deliberate design decision for a later
   experiment, not something to improvise here.

3. `shuffled_letter_order(pair_seed, n_options)` -- same RNG-seeded
   anti-position-bias shuffle as the original `shuffled_letter_order`,
   just parameterized on the example's own option count instead of a
   hardcoded `len(LABELS) == 3`. The original's intent (never let the
   model learn "the answer is always letter B" instead of reading content)
   is unchanged and, if anything, matters more here: with option counts
   varying per-example, a position-only shortcut would generalize even
   worse than in the fixed-3-way case, so this is the single most
   important piece of logic to have gotten right in this fork.

4. Batches now mix different option counts. The collate function pads
   `order` to `MAX_OPTIONS` with a sentinel (unused downstream -- eval only
   reads the first `n_options` real entries per row) and carries an
   explicit `n_options` tensor through to `evaluate()`, which masks the
   restricted logits for a row's unused letter columns to `-inf` *before*
   softmax (not after) so they get exactly zero probability, not a small
   nonzero one. This matters for a reason worth spelling out: with exact
   zero padding, `eval/metrics.py`'s `accuracy`/`brier_score`/
   `expected_calibration_error` need NO changes to work correctly on a
   ragged (variable option-count) eval set represented as one padded
   `(N, MAX_OPTIONS)` array. Proof sketch: argmax is unaffected (a real
   option's softmax probability is always > 0 = padded probability, since
   softmax over >=1 real logits is a valid distribution that sums to 1
   across only the real columns). Brier's per-row squared error over
   padded columns is `(0 - 0)^2 = 0` (padded prediction 0, padded one-hot
   target 0, since the true label is always a real column) -- contributes
   nothing, so the padded-array Brier score is IDENTICAL to computing each
   row's Brier score at its own true width and averaging. ECE only uses
   `max(axis=-1)` (a real column, same argument as accuracy) and
   correctness-of-argmax -- also unaffected. This was checked by hand
   against `eval/metrics.py`'s actual implementation before writing this
   file, not assumed. The one place this is an *approximation* rather than
   exact: the calibration temperature is fit on `log(probs)` as a logit
   surrogate (same approach and same caveat as the original script,
   PRD.md Section 13a.2's noted anomaly), and `log(0)` is clipped to
   `log(1e-12)` for padded columns -- after temperature scaling those
   columns get a tiny nonzero probability (~1e-12 scale) instead of exactly
   0. This is negligible at reported precision but is a real, disclosed
   departure from exactness in the *calibrated* (not raw) numbers, exactly
   analogous to why the original script's raw numbers are the trustworthy
   column per 13a.2.

5. Prompts now read `instructions` from each example's own row (every
   schema needs its own task description) instead of a single hardcoded
   NLI-specific string. `training/data.py`'s NLI records already carry this
   field with the same text the original script hardcoded, so NLI-only
   behavior is unchanged; DBpedia-14/CLINC150 records carry their own.

6. Two held-out eval sets, not one: `eval_id/` (in-distribution -- same two
   trained-on schemas, NLI + DBpedia-14, disjoint examples from train) and
   `eval_ood/` (out-of-distribution -- CLINC150, a schema with ZERO
   examples in train/ at all). Calibration temperature is fit only on a
   slice of `eval_id/` (fitting it on OOD data would leak information the
   model never had at train time into the "did it generalize" question).
   Both `eval_id` and `eval_ood` get raw AND id-fit-temperature-calibrated
   reports, so the OOD calibration number's degradation (or lack of it) is
   itself informative, not just accuracy on unseen schemas.

Everything not discussed above (left-padding index logic, the OOM-skip
training loop, the manifest shape, the preflight VRAM check) is carried
over unchanged from `train_decoder_lora.py`, which already got it right.
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

MAX_OPTIONS = 10  # keep in sync with training/build_multischema_slice.py's MAX_OPTIONS; see docstring point 2
LETTERS = [chr(ord("A") + i) for i in range(MAX_OPTIONS)]  # A..J -- verified single-token, mutually distinct (see docstring)


def shuffled_letter_order(pair_seed: str, n_options: int) -> list[int]:
    """Deterministic per-example shuffle of which option index sits at which letter position.

    Width-aware generalization of the original 3-way `shuffled_letter_order`:
    same RNG construction, same intent (kill the positional shortcut), just
    permuting `range(n_options)` instead of a hardcoded `range(3)`.
    """
    rng = np.random.default_rng(int(hashlib.sha1(pair_seed.encode()).hexdigest()[:8], 16))
    order = list(range(n_options))
    rng.shuffle(order)
    return order  # order[letter_idx] = option_idx shown at that letter, for THIS example's own options list


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
                f"by build_multischema_slice.py's validation; refusing to silently truncate."
            )
        order = shuffled_letter_order(ex["state"] + str(ex["label_idx"]) + str(n_options), n_options)
        prompt_text = build_prompt(ex["state"], options, order, ex["instructions"])
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        target_letter_idx = order.index(ex["label_idx"])  # which letter shows the true option, in THIS example
        return {
            "prompt": prompt,
            "target_letter_idx": target_letter_idx,
            "label_idx": ex["label_idx"],  # index into this example's OWN options list -- not a shared cross-schema space
            "n_options": n_options,
            "order": order,  # length n_options; order[letter_idx] = option_idx shown at that letter
            "source": ex.get("source", "unknown"),
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
            # Unchanged from the original script: left-padding means every row's last real
            # token sits at the same fixed final column regardless of that row's own length.
            labels = torch.full_like(enc["input_ids"], -100)
            last_col = enc["input_ids"].shape[1] - 1
            answer_token_ids = torch.tensor(letter_ids, dtype=torch.long)[target_letter_idx]
            labels[:, last_col] = answer_token_ids
            enc["labels"] = labels

        enc["target_letter_idx"] = target_letter_idx
        enc["label_idx"] = label_idx
        enc["n_options"] = n_options
        enc["order"] = order_padded
        enc["_sources"] = [b["source"] for b in batch]  # not a tensor -- popped before .to(device), used only for reporting
        return enc

    return collate


def load_split(data_dir: Path, tokenizer, max_length: int, split: str, subset: int, seed: int) -> PromptDataset:
    ds = load_from_disk(str(data_dir / split))
    if subset:
        ds = ds.shuffle(seed=seed).select(range(min(subset, len(ds))))
    return PromptDataset(ds, tokenizer, max_length)


@torch.no_grad()
def evaluate(model, loader, device, letter_ids: list[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (probs padded to MAX_OPTIONS columns [true zeros beyond each row's n_options],
    per-row label_idx [index into that row's OWN options list], per-row source strings)."""
    model.eval()
    all_probs, all_labels, all_sources = [], [], []
    letter_ids_t = torch.tensor(letter_ids, device=device)
    for batch in loader:
        batch.pop("target_letter_idx")
        label_idx = batch.pop("label_idx")
        n_options = batch.pop("n_options")
        order = batch.pop("order")
        sources = batch.pop("_sources")
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)

        last_logits = out.logits[:, -1, :]  # (B, vocab) -- left-padding, last real token is always the final column
        restricted = last_logits[:, letter_ids_t]  # (B, MAX_OPTIONS), columns in LETTERS order

        col_idx = torch.arange(MAX_OPTIONS, device=restricted.device).unsqueeze(0)  # (1, MAX_OPTIONS)
        pad_mask = col_idx >= n_options.to(restricted.device).unsqueeze(1)  # (B, MAX_OPTIONS), True at unused letters
        restricted = restricted.masked_fill(pad_mask, float("-inf"))
        restricted_probs = F.softmax(restricted.float(), dim=-1).cpu().numpy()  # padded cols are exact 0.0

        order_np = order.numpy()  # (B, MAX_OPTIONS), -1 sentinel past each row's own n_options
        n_options_np = n_options.numpy()
        canonical_probs = np.zeros_like(restricted_probs)
        for row in range(restricted_probs.shape[0]):
            n_i = int(n_options_np[row])
            valid_order = order_np[row, :n_i]  # permutation of this row's own option indices
            inverse_perm = np.argsort(valid_order)  # option_idx -> letter_idx that displayed it
            canonical_probs[row, :n_i] = restricted_probs[row, :n_i][inverse_perm]
            # columns [n_i:MAX_OPTIONS] stay exactly 0.0 from np.zeros_like -- correct, no real option there

        all_probs.append(canonical_probs)
        all_labels.append(label_idx.numpy())
        all_sources.extend(sources)
    return np.concatenate(all_probs), np.concatenate(all_labels), all_sources


def _report_by_source(probs: np.ndarray, labels: np.ndarray, sources: list[str]) -> dict:
    by_source = {}
    sources_arr = np.array(sources)
    for src in sorted(set(sources)):
        m = sources_arr == src
        by_source[src] = full_report(probs[m], labels[m], MAX_OPTIONS)
    return by_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed" / "multischema_slice"))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-multischema"))
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
    ap.add_argument("--calib-fraction", type=float, default=0.3,
                     help="fraction of eval_id/ (NOT eval_ood/) used to fit the calibration temperature")
    ap.add_argument("--grad-checkpointing", action="store_true",
                     help="off by default: trades ~30-40% more compute for memory headroom; "
                          "only worth it if a real run OOMs without it")
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
    tokenizer.padding_side = "left"

    letter_token_ids_full = [tokenizer.encode(l, add_special_tokens=False) for l in LETTERS]
    for l, ids in zip(LETTERS, letter_token_ids_full):
        assert len(ids) == 1, f"letter {l!r} is not a single token under {args.base_model}'s tokenizer: {ids}"
    letter_ids = [ids[0] for ids in letter_token_ids_full]
    assert len(set(letter_ids)) == len(letter_ids), f"letter token ids are not mutually distinct: {letter_ids}"
    print(f"letter token ids: {dict(zip(LETTERS, letter_ids))}")

    data_dir = Path(args.data_dir)
    train_pd = load_split(data_dir, tokenizer, args.max_length, "train", args.train_subset, args.seed)
    eval_id_full = load_split(data_dir, tokenizer, args.max_length, "eval_id", args.eval_id_subset, args.seed)
    eval_ood_pd = load_split(data_dir, tokenizer, args.max_length, "eval_ood", args.eval_ood_subset, args.seed)

    # Defensive width check -- fail loudly before spending GPU time if the data violates the cap
    # this script's letter/masking logic assumes.
    for name, pd_ds in [("train", train_pd), ("eval_id", eval_id_full), ("eval_ood", eval_ood_pd)]:
        widths = [len(o) for o in pd_ds.ds["options"]]  # single batched column read, not a per-row Python loop
        assert min(widths) >= 2 and max(widths) <= MAX_OPTIONS, (
            f"{name}: option-count range [{min(widths)}, {max(widths)}] violates [2, {MAX_OPTIONS}]"
        )

    n_calib = int(len(eval_id_full) * args.calib_fraction)
    calib_pd = Subset(eval_id_full, range(n_calib))
    test_id_pd = Subset(eval_id_full, range(n_calib, len(eval_id_full)))
    print(f"train: {len(train_pd)} | eval_id calib: {len(calib_pd)} | eval_id test: {len(test_id_pd)} | eval_ood: {len(eval_ood_pd)}")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=device)
    if args.grad_checkpointing:
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
    test_id_loader = DataLoader(test_id_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)
    ood_loader = DataLoader(eval_ood_pd, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_collate, num_workers=2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    optim_steps_per_epoch = -(-len(train_loader) // args.grad_accum_steps)
    total_steps = optim_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    start_time = time.time()
    step, skipped_oom = 0, 0
    # running_loss/running_loss_count track exactly the micro-batches accumulated since the
    # last print (reset together), rather than inferring a denominator from the step number --
    # that inference is wrong on the first window and on any run shorter than 50 optimizer
    # steps (i.e. any smoke test), which is why it's an explicit counter instead.
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

    calib_probs, calib_labels, _ = evaluate(model, calib_loader, device, letter_ids)
    calib_logit_surrogate = np.log(np.clip(calib_probs, 1e-12, 1.0))
    temperature = fit_temperature(calib_logit_surrogate, calib_labels)
    print(f"fitted temperature (on eval_id calib slice only): {temperature:.4f}")

    id_probs, id_labels, id_sources = evaluate(model, test_id_loader, device, letter_ids)
    id_logit_surrogate = np.log(np.clip(id_probs, 1e-12, 1.0))
    id_calibrated_probs = F.softmax(torch.tensor(id_logit_surrogate) / temperature, dim=-1).numpy()

    ood_probs, ood_labels, ood_sources = evaluate(model, ood_loader, device, letter_ids)
    ood_logit_surrogate = np.log(np.clip(ood_probs, 1e-12, 1.0))
    ood_calibrated_probs = F.softmax(torch.tensor(ood_logit_surrogate) / temperature, dim=-1).numpy()

    id_raw_report = full_report(id_probs, id_labels, MAX_OPTIONS)
    id_calibrated_report = full_report(id_calibrated_probs, id_labels, MAX_OPTIONS)
    ood_raw_report = full_report(ood_probs, ood_labels, MAX_OPTIONS)
    ood_calibrated_report = full_report(ood_calibrated_probs, ood_labels, MAX_OPTIONS)

    id_raw_by_source = _report_by_source(id_probs, id_labels, id_sources)
    ood_raw_by_source = _report_by_source(ood_probs, ood_labels, ood_sources)

    print("eval_id raw:", json.dumps(id_raw_report, indent=2))
    print("eval_id calibrated:", json.dumps(id_calibrated_report, indent=2))
    print("eval_id raw by source:", json.dumps(id_raw_by_source, indent=2))
    print("eval_ood raw (ZERO-SHOT SCHEMA):", json.dumps(ood_raw_report, indent=2))
    print("eval_ood calibrated (id-fit temperature applied to an unseen schema):", json.dumps(ood_calibrated_report, indent=2))
    print("eval_ood raw by source:", json.dumps(ood_raw_by_source, indent=2))

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")

    manifest = {
        "base_model": args.base_model,
        "architecture": "decoder-lora-restricted-logit-multischema",
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
        "eval_ood_raw_report": ood_raw_report,
        "eval_ood_calibrated_report": ood_calibrated_report,
        "eval_ood_raw_by_source": ood_raw_by_source,
        "eval_ood_note": "zero training examples from this schema/source were seen during training -- this is the generalization test",
        "skipped_oom_batches": skipped_oom,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_seconds": train_seconds,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
