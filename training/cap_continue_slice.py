"""
Fix for the v3 continue-train stall: measure_lengths.py (run on-server,
2026-09-25) showed only 496/26,250 rows (1.9%) of data/processed/
wide_slice_continue/train exceed 768 rendered prompt tokens, and they are
overwhelmingly one source -- bjb:ledgar/provision_type (478/3179 rows) --
plus 18 bjb:cfpb_complaints/product rows. mean=312, median=245, p90=602,
p99=849, max=1159 tokens.

With --batch-size 1 --grad-accum-steps 64, each optimizer step draws 64
rows; P(a step contains >=1 outlier) = 1-(1-0.019)^64 ~= 68%. Each such row,
even alone in its accumulation window, was pushing a step past 10-12+
minutes (observed live: step 2 hit 12:42 elapsed with no sign of finishing)
-- consistent with `causal_conv1d`'s reference-PyTorch fallback (no
optimized kernel, no CUDA toolkit on this box to build one) scaling badly
with sequence length, most likely a per-position Python loop whose overhead
compounds across ~30+ layers x 2 (grad-checkpoint recompute).

decoder_lora_lib.py's make_collate() deliberately refuses to silently
truncate (PRD 5.2b) -- so the fix is to exclude these rows from *this*
continue-train run, not lower --max-length and let training crash on the
first excluded-but-present row. This filters the already-built
wide_slice_continue split (no need to touch the frozen wide_slice or
vision_slice sources); LEDGAR keeps ~2700/3179 rows (85%), still a
meaningful sample of that task.
"""

from pathlib import Path

from datasets import load_from_disk
from transformers import AutoProcessor

from training.decoder_lora_lib import build_code_table, build_prompt_text

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen3.5-4B"
CAP = 768


def filter_split(ds, processor, codes):
    keep = []
    n_over = 0
    for r in ds:
        text = build_prompt_text(
            r["state"], r["options"], r.get("order") or list(range(len(r["options"]))),
            r.get("instructions", ""), codes,
        )
        n_tok = len(processor.tokenizer(text, add_special_tokens=True)["input_ids"])
        if n_tok <= CAP:
            keep.append(True)
        else:
            keep.append(False)
            n_over += 1
    return ds.select([i for i, k in enumerate(keep) if k]), n_over


def main():
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    codes, _ = build_code_table(processor.tokenizer)

    src_dir = REPO_ROOT / "data/processed/wide_slice_continue"
    out_dir = REPO_ROOT / "data/processed/wide_slice_continue_capped"

    for split in ("train", "eval_id", "eval_ood"):
        ds = load_from_disk(str(src_dir / split))
        filtered, n_over = filter_split(ds, processor, codes)
        filtered.save_to_disk(str(out_dir / split))
        print(f"{split}: {len(ds)} -> {len(filtered)} rows ({n_over} excluded, >{CAP} tokens)")


if __name__ == "__main__":
    main()
