"""
Build a FULL-corpus training slice from a `bjb export` public JSONL.

Unlike `build_vision_slice.py` (which passes through 13a.10's frozen 62k
benchcorpus rows), this builder consumes the *entire* public slice --
`bjb export --out exports/full --tiers A --max-options 0` -- so the 0.8B
model trains on all of better-jev-bench's public data, not the staged
curriculum subsets. General to vision-or-not: text records keep the exact
8-key shape, image-bearing records carry `images: [<local path>]` (resolved
+ hash-verified by the export itself; existence re-asserted here).

What it does:
  1. Stream the export JSONL, validate every row (option count 2..588,
     label_idx points at label, no duplicate options, known question_type,
     at most 1 image with an existing file). `score` option order is never
     touched -- the export's canonical ascending scale is final.
  2. Deterministic shuffle (seed) + stratified carve-out of `eval_id`
     (default 2,500) and `eval_ood` (default 3,000), proportional by source
     with a per-source minimum. Both are EXCLUDED from train. Note: these
     are in-public samples, not zero-shot -- CLINC150 etc. are in-train now,
     so the old "CLINC zero-shot" gate is superseded by the bjb held-out
     sweep + JevBench/jabr-v2, which stay untouched (different files).
  3. Token-budget audit against the real tokenizer/processor for --base-model
     (default Qwen/Qwen3.5-0.8B). Text rows: exact tokenizer count. Vision
     rows: exact text-token count + a sampled, measured image-token overhead
     (full processor encode on a stratified sample of 60); only rows within
     the danger band get an exact full encode. Over-budget rows are EXCLUDED
     and reported per source (same policy as PRD 5.1b item 5's 496-row
     exclusion) -- never truncated, never silent.
  4. Write HF datasets {train, eval_id, eval_ood} + stats.json with counts,
     exclusions, eval composition, export sha256 and seed.

Run:
    uv run python3 -m training.build_fullcorpus_slice \\
        --export ../bench-repo/exports/full/public.jsonl \\
        --out-dir data/processed/fullcorpus_08b \\
        --base-model Qwen/Qwen3.5-0.8B --max-length 850
"""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_OPTIONS = 588  # PRD 5.1b -- full-width rows (clinc150 151-way) must pass
SEED = 42
VISION_MAX_LENGTH = 850  # stage4's proven value for mixed text+vision
VISION_MAX_PIXELS = 256 * 28 * 28
EVAL_ID_N = 2500
EVAL_OOD_N = 3000
MIN_PER_SOURCE_EVAL = 10
VISION_SAMPLE_N = 60
MARGIN = 32


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_export(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            imgs = rec.get("images") or []
            assert len(imgs) <= 1, f"{path}:{line_no}: {len(imgs)} images, v1 supports at most 1"
            img_paths = []
            for im in imgs:
                p = Path(im["path"])
                assert p.exists(), f"{path}:{line_no}: image missing: {p}"
                actual = _sha256_file(p)
                assert actual == im["sha256"], f"{path}:{line_no}: sha256 mismatch for {p}"
                img_paths.append(str(p))
            r = {
                "state": rec["state"],
                "question_key": rec["question_key"],
                "question_type": rec["question_type"],
                "instructions": rec["instructions"],
                "options": list(rec["options"]),
                "label": rec["label"],
                "label_idx": rec["label_idx"],
                "source": rec["source"],
                "images": img_paths,
            }
            records.append(r)
    return records


def _validate(records: list[dict], name: str) -> None:
    for r in records:
        n = len(r["options"])
        assert 2 <= n <= MAX_OPTIONS, f"{name}: option count {n} out of [2, {MAX_OPTIONS}]"
        assert 0 <= r["label_idx"] < n, f"{name}: label_idx out of range"
        assert r["options"][r["label_idx"]] == r["label"], f"{name}: label_idx mismatch"
        assert len(set(r["options"])) == n, f"{name}: duplicate options"
        assert r["question_type"] in ("choice", "noul", "score"), f"{name}: bad type"
        assert len(r.get("images") or []) <= 1, f"{name}: >1 image"


def _stratified_split(records: list[dict], n_eval_id: int, n_eval_ood: int,
                      seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    for rows in by_source.values():
        rng.shuffle(rows)
    total = len(records)
    eval_id, eval_ood, train = [], [], []
    for source, rows in sorted(by_source.items()):
        frac = len(rows) / total
        k_id = max(MIN_PER_SOURCE_EVAL, round(frac * n_eval_id)) if len(rows) > MIN_PER_SOURCE_EVAL else 0
        k_ood = max(MIN_PER_SOURCE_EVAL, round(frac * n_eval_ood)) if len(rows) > 2 * MIN_PER_SOURCE_EVAL else 0
        k_id = min(k_id, len(rows) // 3)
        k_ood = min(k_ood, (len(rows) - k_id) // 3)
        eval_id.extend(rows[:k_id])
        eval_ood.extend(rows[k_id:k_id + k_ood])
        train.extend(rows[k_id + k_ood:])
    rng.shuffle(train)
    rng.shuffle(eval_id)
    rng.shuffle(eval_ood)
    return train, eval_id, eval_ood


def _row_text(r: dict, codes: list[str]) -> str:
    options = r["options"]
    n = len(options)
    option_lines = "\n".join(f"{codes[j]}) {options[j]}" for j in range(n))
    if n <= 26:
        tail = f"Answer with a single letter ({'/'.join(codes[:n])})."
    else:
        tail = f"Answer with a single option code from the list above ({codes[0]} .. {codes[n-1]})."
    return f"{r['instructions']}\n\n{r['state']}\n\nOptions:\n{option_lines}\n\n{tail}"


def _audit_token_budget(train, eval_id, eval_ood, base_model, max_length, max_pixels, seed):
    """Exact tokenizer audit for every row; exact processor audit for vision
    rows in the danger band; sampled overhead for the rest. Returns
    (kept_train, kept_eval_id, kept_eval_ood, exclusions)."""
    from transformers import AutoTokenizer, AutoProcessor
    from training.decoder_lora_lib import build_code_table

    all_rows = [("train", r) for r in train] + [("eval_id", r) for r in eval_id] + [("eval_ood", r) for r in eval_ood]
    tok = AutoTokenizer.from_pretrained(base_model)
    codes, _ = build_code_table(tok)

    def text_tokens(r):
        text = _row_text(r, codes)
        messages = [{"role": "user", "content": text}]
        rendered = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        return len(tok(rendered)["input_ids"])

    vision_rows = [(split, r) for split, r in all_rows if r.get("images")]
    text_rows = [(split, r) for split, r in all_rows if not r.get("images")]
    print(f"auditing {len(text_rows)} text + {len(vision_rows)} vision rows vs max_length={max_length}...")

    # 1. Measure image-token overhead on a stratified sample (full encode).
    from PIL import Image
    proc = AutoProcessor.from_pretrained(base_model, max_pixels=max_pixels)
    pcodes, _ = build_code_table(proc.tokenizer)
    rng = random.Random(seed)
    sample = rng.sample(vision_rows, min(VISION_SAMPLE_N, len(vision_rows)))
    overheads = []
    for _, r in sample:
        text = _row_text(r, pcodes)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        rendered = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        img = Image.open(r["images"][0]).convert("RGB")
        enc = proc(text=[rendered], images=[img], return_tensors="pt")
        total = enc["input_ids"].shape[1]
        tt = text_tokens(r)
        overheads.append(total - tt)
    max_over = max(overheads) if overheads else 0
    print(f"measured image-token overhead on {len(sample)} sampled vision rows: "
          f"min={min(overheads) if overheads else 0}, max={max_over}")

    # 2. Exact text-token count for every row; danger-band vision rows get exact encode.
    excluded = []
    kept = {"train": [], "eval_id": [], "eval_ood": []}

    def exact_vision_total(r):
        text = _row_text(r, pcodes)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        rendered = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        img = Image.open(r["images"][0]).convert("RGB")
        enc = proc(text=[rendered], images=[img], return_tensors="pt")
        return enc["input_ids"].shape[1]

    for split, r in all_rows:
        tt = text_tokens(r)
        if not r.get("images"):
            if tt >= max_length:
                excluded.append((split, r["source"], tt))
            else:
                kept[split].append(r)
        else:
            if tt + max_over + MARGIN < max_length:
                kept[split].append(r)
            else:
                total = exact_vision_total(r)
                if total >= max_length:
                    excluded.append((split, r["source"], total))
                else:
                    kept[split].append(r)
    return kept["train"], kept["eval_id"], kept["eval_ood"], excluded


def build(export: Path, out_dir: Path, *, base_model: str, max_length: int,
          max_pixels: int, seed: int, n_eval_id: int, n_eval_ood: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading export {export}...")
    records = load_export(export)
    print(f"loaded {len(records)} records "
          f"({sum(1 for r in records if r['images'])} with images)")
    _validate(records, "export")

    train, eval_id, eval_ood = _stratified_split(records, n_eval_id, n_eval_ood, seed)
    print(f"split: train={len(train)} eval_id={len(eval_id)} eval_ood={len(eval_ood)}")

    train, eval_id, eval_ood, excluded = _audit_token_budget(
        train, eval_id, eval_ood, base_model, max_length, max_pixels, seed)
    if excluded:
        by_src = collections.Counter(f"{s}/{src}" for s, src, _ in excluded)
        print(f"excluded {len(excluded)} over-budget rows (no truncation, all logged): {dict(by_src)}")
    for name, rows in [("train", train), ("eval_id", eval_id), ("eval_ood", eval_ood)]:
        _validate(rows, name)

    Dataset.from_list(train).save_to_disk(str(out_dir / "train"))
    Dataset.from_list(eval_id).save_to_disk(str(out_dir / "eval_id"))
    Dataset.from_list(eval_ood).save_to_disk(str(out_dir / "eval_ood"))

    def summ(rows):
        return {"n": len(rows),
                "by_question_type": dict(collections.Counter(r["question_type"] for r in rows)),
                "n_sources": len(set(r["source"] for r in rows)),
                "n_with_images": sum(1 for r in rows if r.get("images"))}

    stats = {
        "export": str(export),
        "export_sha256": _sha256_file(export),
        "base_model": base_model,
        "max_length": max_length,
        "max_pixels": max_pixels,
        "seed": seed,
        "train": summ(train),
        "eval_id": summ(eval_id),
        "eval_ood": {**summ(eval_ood),
                     "note": "in-public stratified sample, NOT zero-shot -- every source is in-train; "
                             "true zero-shot gates are the bjb held-out sweep + JevBench/jabr-v2."},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--max-length", type=int, default=VISION_MAX_LENGTH)
    ap.add_argument("--max-pixels", type=int, default=VISION_MAX_PIXELS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-id-n", type=int, default=EVAL_ID_N)
    ap.add_argument("--eval-ood-n", type=int, default=EVAL_OOD_N)
    args = ap.parse_args()
    build(Path(args.export), Path(args.out_dir), base_model=args.base_model,
          max_length=args.max_length, max_pixels=args.max_pixels, seed=args.seed,
          n_eval_id=args.eval_id_n, n_eval_ood=args.eval_ood_n)


if __name__ == "__main__":
    main()
