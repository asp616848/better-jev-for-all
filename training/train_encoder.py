"""
Phase 1 `ekvachan-base` training: fine-tune ModernBERT-large with a
classification head for the `choice` primitive, using the CE + Brier
calibration loss from PRD.md Section 5.3 (lambda~=0.5), followed by
post-hoc temperature scaling on a held-out calibration split.

Deliberately a plain PyTorch loop (no HF Trainer) so the CE+Brier loss and
the evidence-bundle output are fully transparent and match what Section 10
commits to publishing (the whole recipe, not just a "trainer.train()" call).
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
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from eval.metrics import fit_temperature, full_report

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = ["entailment", "neutral", "contradiction"]


def brier_ce_loss(logits: torch.Tensor, labels: torch.Tensor, brier_lambda: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean()
    return ce + brier_lambda * brier, ce.item(), brier.item()


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        states = [b["state"] for b in batch]
        labels = torch.tensor([b["label_idx"] for b in batch], dtype=torch.long)
        enc = tokenizer(states, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc["labels"] = labels
        return enc

    return collate


@torch.no_grad()
def get_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out.logits.float().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-large")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed" / "nli_slice"))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "checkpoints" / "ekvachan-base"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--brier-lambda", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-subset", type=int, default=0, help="0 = full dataset; >0 = quick dev run")
    ap.add_argument("--eval-subset", type=int, default=0, help="0 = full eval set; >0 = quick dev run")
    ap.add_argument("--calib-fraction", type=float, default=0.3, help="fraction of eval set used to fit temperature")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds = load_from_disk(str(Path(args.data_dir) / "train"))
    eval_ds = load_from_disk(str(Path(args.data_dir) / "eval"))
    if args.eval_subset:
        eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.eval_subset, len(eval_ds))))
    if args.train_subset:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.train_subset, len(train_ds))))

    # split eval -> calibration (fit T) / test (final report) — never fit T on the set we report on
    eval_ds = eval_ds.shuffle(seed=args.seed)
    n_calib = int(len(eval_ds) * args.calib_fraction)
    calib_ds = eval_ds.select(range(n_calib))
    test_ds = eval_ds.select(range(n_calib, len(eval_ds)))

    print(f"train: {len(train_ds)} | calib: {len(calib_ds)} | test: {len(test_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=len(LABELS))
    model.to(device)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=4)
    calib_loader = DataLoader(calib_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"bf16: {use_bf16}")

    start_time = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss, running_ce, running_brier = 0.0, 0.0, 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(**batch)
                loss, ce_val, brier_val = brier_ce_loss(out.logits, labels, args.brier_lambda)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            running_ce += ce_val
            running_brier += brier_val
            step += 1
            if step % 200 == 0:
                elapsed = time.time() - start_time
                print(f"epoch {epoch} step {step}/{total_steps} | loss {running_loss/200:.4f} "
                      f"(ce {running_ce/200:.4f} brier {running_brier/200:.4f}) | {elapsed:.0f}s elapsed")
                running_loss = running_ce = running_brier = 0.0

        print(f"=== epoch {epoch} done, {time.time()-start_time:.0f}s elapsed ===")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_logits, calib_labels = get_logits(model, calib_loader, device)
    temperature = fit_temperature(calib_logits, calib_labels)
    print(f"fitted temperature: {temperature:.4f}")

    test_logits, test_labels = get_logits(model, test_loader, device)
    raw_probs = F.softmax(torch.tensor(test_logits), dim=-1).numpy()
    calibrated_probs = F.softmax(torch.tensor(test_logits) / temperature, dim=-1).numpy()

    raw_report = full_report(raw_probs, test_labels, len(LABELS))
    calibrated_report = full_report(calibrated_probs, test_labels, len(LABELS))

    print("raw (T=1):", json.dumps(raw_report, indent=2))
    print("calibrated:", json.dumps(calibrated_report, indent=2))

    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")

    weight_hash = hashlib.sha256()
    for p in sorted((output_dir / "model").glob("*.safetensors")):
        weight_hash.update(p.read_bytes())

    manifest = {
        "base_model": args.base_model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "brier_lambda": args.brier_lambda,
        "train_size": len(train_ds),
        "calib_size": len(calib_ds),
        "test_size": len(test_ds),
        "temperature": temperature,
        "raw_report": raw_report,
        "calibrated_report": calibrated_report,
        "weight_sha256": weight_hash.hexdigest(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_seconds": time.time() - start_time,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
