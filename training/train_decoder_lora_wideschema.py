"""
Fork of `train_decoder_lora_multischema.py`, built to run the harder, honest
follow-up PRD.md Section 13a.5 (2026-09-23) explicitly calls out as "the
natural next experiment": genuine wide-schema classification -- full-width
DBpedia-14 (all 14 real categories, no subsetting) in training, and a wider
(15-26 option, still not full 151-way) CLINC150 held-out eval -- instead of
13a.5's random-4-8-option-subset-always-containing-the-answer construction,
which both its training and held-out schemas used and which made that run's
98.5% zero-shot-schema number easier to achieve than genuine wide
classification would be.

**Why a new fork, not editing `train_decoder_lora_multischema.py` in
place**: exactly the reasoning that script's own docstring gives for why IT
forked from `train_decoder_lora.py`, applied one experiment later.
`train_decoder_lora_multischema.py`'s output is the exact artifact behind
PRD.md Section 13a.5's published, reviewed numbers (92.14% / 98.50% accuracy,
0.0131 / 0.0339 raw ECE on eval_id / eval_ood). Widening its `MAX_OPTIONS` or
changing its data contract in place would make that already-published run
silently unreproducible from its own script. Forking again keeps 13a.5's
script and numbers frozen and isolates the new, harder task so a bug here
can't retroactively touch the trusted 13a.5 artifact -- the same policy that
script itself already established, applied consistently rather than
special-cased away because "we're just widening a constant this time."

**What's actually different from `train_decoder_lora_multischema.py`, and
why each change is safe** (read this before trusting it -- letter-index/
position-bias math is exactly the class of subtle bug both this script's and
the original 3-way script's docstrings warn took real debugging effort the
first time):

1. **`MAX_OPTIONS`: 10 -> 26.** This required auditing every place the old
   cap was load-bearing, not just bumping the constant -- done here, in
   full, before trusting the result:
   - `LETTERS = [chr(ord("A") + i) for i in range(MAX_OPTIONS)]` is already
     parameterized on `MAX_OPTIONS`, not hardcoded to 10 -- generating A-Z
     instead of A-J needed no code change, only the constant.
   - `shuffled_letter_order(pair_seed, n_options)` permutes
     `range(n_options)` via a seeded RNG -- already fully width-agnostic,
     no reference to 10 or 26 anywhere in it. No off-by-one risk: it
     produces a permutation of exactly the row's own option count,
     whatever that is.
   - `build_prompt` writes `LETTERS[i] for i in range(n)` and
     `LETTERS[:n]` for the "answer with" line -- both already index by the
     row's own `n`, not by `MAX_OPTIONS`. An example with n_options=5 always
     uses letters A-E in that presentation (never skips to, say, "K") --
     this was true at MAX_OPTIONS=10 and remains true at 26; widening the
     cap does not change which letters a narrow-option example uses.
   - The collate function's `order_padded` tensor and `evaluate()`'s
     `pad_mask = col_idx >= n_options` masking are shaped by `MAX_OPTIONS`
     (now 26 columns instead of 10) but the masking logic itself -- mask
     unused letter-logit columns to exact `-inf` *before* softmax, so they
     get exactly zero probability, not a small nonzero one -- has no
     hardcoded width dependence. The `eval/metrics.py` correctness proof
     sketch in the multischema script's own docstring (argmax/Brier/ECE all
     unaffected by zero-padding to a wider column count) holds unchanged at
     width 26; it never depended on the specific value 10, only on padded
     columns being exact zeros, which they still are.
   - **Conclusion: this is genuinely a one-constant change, verified by
     reading the logic, not just re-run at the old cap and assumed fine.**
     No position-bias or off-by-one fix was needed because none of the
     above code paths encoded 10 as anything other than "whatever
     `MAX_OPTIONS` is."
   - Independently re-verified (not just trusted from the multischema
     script's A-J check) on this server on 2026-09-23: all 26 letters A-Z
     are single, mutually distinct tokens under Qwen3.5-4B's tokenizer
     (ids 32-57, contiguous). `main()`'s existing runtime assertion
     (`assert len(ids) == 1` for every letter) re-checks this live on every
     run regardless, so a tokenizer change on a future base model would
     still fail loudly rather than silently misread a letter position.

2. **Data contract is unchanged** (`options`, `label_idx`, `instructions`
   per-row, ragged option counts) -- this script reads whatever
   `training/build_wideschema_slice.py` writes, at up to 26 options instead
   of up to 10. No new fields were needed: DBpedia-14's full-width records
   just carry 14-length `options` lists instead of 4-8-length ones, and
   CLINC150's wide records carry 15-26-length lists instead of 4-8-length
   ones. The training loop, collate function, and eval function are
   otherwise byte-identical to `train_decoder_lora_multischema.py`.

3. **Default `--data-dir`/`--output-dir` point at the new wideschema slice**
   (`data/processed/wideschema_slice`, `checkpoints/ekvachan-decoder-qwen-wideschema`)
   instead of the multischema ones, so this script can't accidentally read
   13a.5's data or overwrite its checkpoint.

**The honest limit, stated here too, not just in the data-builder**: CLINC150
has 151 real classes (150 intents + `oos`). 151 > 26, so it is structurally
out of reach for this single-uppercase-letter restricted-logit mechanism --
not a bug, not something this script works around, a real ceiling. Reaching
genuine 151-way CLINC150 classification would need either a multi-token
option-identifier scheme (unexplored, real design work) or PRD.md Section
5.1a's cross-attention decision head (a different mechanism entirely, scores
by encoding option text rather than reading one output token). Neither is
attempted here. The wide (15-26 way) CLINC150 eval this script runs is a
meaningfully harder test than 13a.5's 4-8-way version, but it is NOT evidence
about the full 151-way task, and should never be reported as if it were.

Everything else (left-padding index logic, the OOM-skip training loop, the
manifest shape, the preflight VRAM check, the two-held-out-eval-set design,
the id-slice-only temperature fitting) is carried over unchanged from
`train_decoder_lora_multischema.py`, which already got it right.
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

MAX_OPTIONS = 26  # A-Z -- the full single-uppercase-letter budget; see docstring point 1
LETTERS = [chr(ord("A") + i) for i in range(MAX_OPTIONS)]  # A..Z -- verified single-token, mutually distinct (see docstring)


def shuffled_letter_order(pair_seed: str, n_options: int) -> list[int]:
    """Deterministic per-example shuffle of which option index sits at which letter position.

    Unchanged from `train_decoder_lora_multischema.py`: already fully
    width-agnostic (permutes `range(n_options)`, never references
    `MAX_OPTIONS`), so this needed no edit to support widths up to 26.
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
                f"by build_wideschema_slice.py's validation; refusing to silently truncate."
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
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed" / "wideschema_slice"))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-wideschema"))
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
    print("eval_ood raw (ZERO-SHOT SCHEMA, WIDE 15-26 WAY):", json.dumps(ood_raw_report, indent=2))
    print("eval_ood calibrated (id-fit temperature applied to an unseen schema):", json.dumps(ood_calibrated_report, indent=2))
    print("eval_ood raw by source:", json.dumps(ood_raw_by_source, indent=2))

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")

    manifest = {
        "base_model": args.base_model,
        "architecture": "decoder-lora-restricted-logit-wideschema",
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
        "eval_ood_note": "zero training examples from this schema/source were seen during training -- this is the generalization test. "
                          "Options are wide (15-26 way) CLINC150 subsets, NOT the full 151-way task -- 151 > 26-letter budget, "
                          "structurally out of reach for this mechanism (see module docstring).",
        "skipped_oom_batches": skipped_oom,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_seconds": train_seconds,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
