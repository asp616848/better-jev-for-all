"""
The general vision-aware data slice builder for the `train_decoder_lora_general.py`
lineage (PRD 5.2b). Same owner instruction as that script's docstring applies
here: keep this general to vision-or-not, don't fork a vision-only builder on
top of an already-forked vision-only trainer.

**What this does, concretely.** Starts from `data/processed/benchcorpus_slice`
(13a.10's frozen 62,000 train / 3,325 eval_id-test / 3,000 eval_ood rows) and
copies it through byte-for-byte except for one additive change: every row
gains an `images: []` field (empty for every text row), so the dataset has a
uniform schema whether or not any vision rows are ever added. It then
optionally adds vision rows to `train`/`eval_id` from one of two sources:

- `--vision-source <path.jsonl>`: real image-bearing records. The expected
  contract, one record per line, is the natural 9-key extension of the
  8-key `{state, question_key, question_type, instructions, options, label,
  label_idx, source}` shape this project's builders already use:
  add `"images": ["<local resolved path>"]` (a list of exactly one local
  file path -- see `decoder_lora_lib.PromptDataset`'s v1 scope note --
  already downloaded and sha256-verified by the producer) and optionally
  `"images_sha256": ["<hash>"]` and `"split": "train"|"eval_id"` for the
  producer to control placement. **This contract is provisional.** As of
  2026-09-24 the sibling bench-repo's own `Item`/`to_ekvachan_record()` does
  not yet emit an `images` key at all -- verified directly against
  `better-jev-bench/better_jev_bench/types.py` and `export.py` on that date;
  its own PRD §13.4 *commits* to adding one shaped almost exactly like this
  (`Item.images: tuple[ImageRef, ...]`, resolved to local paths in
  `to_ekvachan_record()`), but the code doesn't exist yet. When it lands,
  expect a thin adapter here, not a rewrite -- read (never edit)
  `abhijeet-labgpu:~/ekvachan/bench-repo` to confirm the exact shape before
  wiring `--vision-source` at a real `bjb export` output.
- `--smoke`: no real vision source needed. Generates a small set of
  synthetic geometric-shape images with PIL (not sourced from any dataset,
  so there is no licensing question at all) and a `choice` question over
  each ("which shape is drawn in this image?"). This is **explicitly a
  mechanism-only smoke test, not a claim about real vision data or real
  vision accuracy** -- its only job is to exercise the real pipeline
  (processor call, pixel_values batching, the image-token-budget assertion
  below, the vision-tower-freeze assertion in the trainer) end to end before
  any real corpus is ready. Labeled `source="synthetic_smoke_shapes"`
  everywhere so it can never be mistaken for real eval evidence.

**The per-row truncation-budget assertion PRD 5.2b requires** ("the builder
must also assert per-row that image_tokens + text_tokens < max_length, and
fail loudly rather than truncate") is enforced here with the real
`AutoProcessor` for every vision row added, at build time -- not deferred to
the trainer's runtime check in `decoder_lora_lib.make_collate()` (which stays
as defense in depth, not the only line of defense).

`score` rows are never narrowed or reordered here (there are none in the
vision additions this script currently supports; the assertion is inherited
from `_validate()` regardless, matching every other builder in this lineage).

Run (byte-through, no vision rows -- proves the pass-through is really a
no-op on the frozen 13a.10 data):
    uv run python3 -m training.build_vision_slice --out-dir data/processed/vision_slice_textonly

Run (mechanism-only smoke, ~2,000 rows, ~1,500 text / ~500 synthetic vision):
    uv run python3 -m training.build_vision_slice --smoke \\
        --out-dir data/processed/vision_slice_smoke --n-smoke-images 500 --n-smoke-text 1500
"""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_OPTIONS = 26
SEED = 42

BENCHCORPUS_DIR = REPO_ROOT / "data" / "processed" / "benchcorpus_slice"

# Must match train_decoder_lora_general.py's vision defaults (PRD 5.2b) --
# duplicated here as literals, not imported, because the builder and the
# trainer are independently runnable tools and neither should require the
# other to be importable just to know a token budget.
VISION_MAX_LENGTH = 768
VISION_MAX_PIXELS = 256 * 28 * 28

SHAPES = ["circle", "square", "triangle", "star", "cross"]
SHAPE_INSTRUCTIONS = (
    "You are looking at a simple synthetic test image used to validate a "
    "vision training pipeline mechanically -- it is not a claim about a real "
    "visual reasoning task. Given the image, choose which shape it shows."
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _passthrough_records(split_dir: Path) -> list[dict]:
    """13a.10's frozen data, copied through with `images: []` added. Deliberately
    does NOT touch state/options/label_idx/source/question_type -- see module
    docstring for why the text side must not move."""
    ds = load_from_disk(str(split_dir))
    out = []
    for row in ds:
        r = dict(row)
        r["images"] = []
        out.append(r)
    return out


def _gen_synthetic_shape_image(rng: random.Random, out_path: Path, shape: str, size=(320, 240)) -> None:
    from PIL import Image, ImageDraw

    bg = tuple(rng.randint(180, 255) for _ in range(3))
    fg = tuple(rng.randint(0, 120) for _ in range(3))
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    w, h = size
    cx, cy = rng.randint(w // 4, 3 * w // 4), rng.randint(h // 4, 3 * h // 4)
    r = min(w, h) // 5

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
    elif shape == "star":
        import math
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r // 2
            ang = math.pi / 5 * i - math.pi / 2
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        draw.polygon(pts, fill=fg)
    elif shape == "cross":
        thickness = r // 2
        draw.rectangle([cx - r, cy - thickness, cx + r, cy + thickness], fill=fg)
        draw.rectangle([cx - thickness, cy - r, cx + thickness, cy + r], fill=fg)
    else:
        raise ValueError(shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def build_smoke_vision_records(n: int, images_dir: Path, rng: random.Random) -> list[dict]:
    images_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        true_shape = rng.choice(SHAPES)
        options = list(SHAPES)
        rng.shuffle(options)
        label_idx = options.index(true_shape)
        img_path = images_dir / f"shape_{i:05d}_{true_shape}.png"
        _gen_synthetic_shape_image(rng, img_path, true_shape)
        out.append({
            "state": "A single shape is drawn on a plain background.",
            "question_key": "shape",
            "question_type": "choice",
            "instructions": SHAPE_INSTRUCTIONS,
            "options": options,
            "label": true_shape,
            "label_idx": label_idx,
            "source": "synthetic_smoke_shapes",
            "images": [str(img_path.resolve())],
        })
    return out


def load_vision_source_records(path: Path) -> tuple[list[dict], list[dict]]:
    """Reads the provisional `--vision-source` JSONL contract described in the
    module docstring. Returns (train_records, eval_id_records)."""
    train, eval_id = [], []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            images = rec.get("images") or []
            if len(images) != 1:
                raise ValueError(f"{path}:{line_no}: expected exactly 1 image path, got {len(images)}")
            img_path = Path(images[0])
            if not img_path.is_absolute():
                raise ValueError(f"{path}:{line_no}: image path must already be resolved to an absolute "
                                  f"local path by the producer, got {images[0]!r}")
            if not img_path.exists():
                raise FileNotFoundError(f"{path}:{line_no}: {img_path} does not exist")
            expected_sha = (rec.get("images_sha256") or [None])[0]
            if expected_sha is not None:
                actual = _sha256_file(img_path)
                if actual != expected_sha:
                    raise ValueError(f"{path}:{line_no}: sha256 mismatch for {img_path}: "
                                      f"expected {expected_sha}, got {actual}")
            split = rec.get("split", "train")
            (train if split == "train" else eval_id).append(rec)
    return train, eval_id


def _assert_within_token_budget(vision_records: list[dict], text_records: list[dict],
                                 base_model: str, max_length: int, max_pixels: int) -> None:
    """PRD 5.2b's required per-row assertion, run for real against the actual
    processor/tokenizer at build time -- fail loudly here, before a single GPU
    cycle is spent, rather than truncating silently mid-run.

    **Fixed 2026-09-24, found by the first real full-scale run crashing on it**:
    this originally checked vision rows only. That missed the actual failure
    mode -- 13a.10's inherited 62,000 text rows were never audited against the
    *new* (768, up from 384) budget at all, and 86 of them (0.14%, max 1,171
    tokens, from `bjb:cfpb_complaints/product`'s longer complaint narratives)
    exceed it. `decoder_lora_lib.py`'s collate-time assertion caught this at
    the first mixed batch that happened to draw one -- correctly, since
    fail-loud-at-runtime is the designed fallback, but this build-time check
    should have caught it first, which is the whole point of having it. Now
    checks every row, vision or not, with the tokenizer for text-only rows
    (cheap, no image I/O) and the processor for vision rows (as before)."""
    from transformers import AutoTokenizer, AutoProcessor

    def _row_text(r: dict) -> str:
        options = r["options"]
        n = len(options)
        letters = [chr(ord("A") + j) for j in range(n)]
        option_lines = "\n".join(f"{letters[j]}) {options[j]}" for j in range(n))
        return (f"{r['instructions']}\n\n{r['state']}\n\nOptions:\n{option_lines}\n\n"
                f"Answer with a single letter ({'/'.join(letters)}).")

    violations = []

    if text_records:
        tok = AutoTokenizer.from_pretrained(base_model)
        for r in text_records:
            text = _row_text(r)
            messages = [{"role": "user", "content": text}]
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            n_tokens = len(tok(rendered)["input_ids"])
            if n_tokens >= max_length:
                violations.append((r.get("source"), None, n_tokens))

    if vision_records:
        from PIL import Image
        proc = AutoProcessor.from_pretrained(base_model, max_pixels=max_pixels)
        for r in vision_records:
            text = _row_text(r)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            rendered = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            img = Image.open(r["images"][0]).convert("RGB")
            enc = proc(text=[rendered], images=[img], return_tensors="pt")
            n_tokens = enc["input_ids"].shape[1]
            if n_tokens >= max_length:
                violations.append((r.get("source"), r["images"][0], n_tokens))

    if violations:
        raise ValueError(
            f"{len(violations)} row(s) (text and/or vision) exceed --max-length {max_length} tokens: "
            f"{violations[:5]} ... refusing to emit a slice that would silently truncate. "
            f"Raise --max-length (a text outlier needs headroom independent of --max-pixels), or exclude "
            f"the offending source explicitly and say so."
        )
    print(f"token-budget assertion passed: {len(vision_records)} vision + {len(text_records)} text row(s), "
          f"all < {max_length} tokens (max_pixels={max_pixels})")


def _validate(records: list[dict], name: str) -> None:
    for r in records:
        n = len(r["options"])
        assert 2 <= n <= MAX_OPTIONS, f"{name}: option count {n} out of [2, {MAX_OPTIONS}]"
        assert 0 <= r["label_idx"] < n, f"{name}: label_idx {r['label_idx']} out of range for {n} options"
        assert r["options"][r["label_idx"]] == r["label"], f"{name}: label_idx doesn't point at label text"
        assert len(set(r["options"])) == n, f"{name}: duplicate option text in {r['options']}"
        assert r["question_type"] in ("choice", "noul", "score"), f"{name}: unexpected question_type {r['question_type']!r}"
        images = r.get("images") or []
        assert len(images) <= 1, f"{name}: {len(images)} images on one row -- v1 scope supports at most 1"
        # Inherited invariant from build_primitives_slice.py / export.py's own
        # narrow_options(): an ordinal `score` scale's option order is its scale
        # position, and must never be resampled or reshuffled. Nothing in this
        # script currently narrows or reshuffles *any* row's options after they
        # arrive (vision rows included) -- record loaders build `options` once,
        # in final order, and this function only validates. Stated here so a
        # future edit that adds narrowing can't silently skip the check every
        # other builder in this lineage enforces.


def build(out_dir: Path, *, smoke: bool, n_smoke_images: int, n_smoke_text: int,
          vision_source: Path | None, base_model: str, max_length: int, max_pixels: int, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print("loading 13a.10's frozen benchcorpus_slice (unchanged pass-through)...")
    train_records = _passthrough_records(BENCHCORPUS_DIR / "train")
    eval_id_records = _passthrough_records(BENCHCORPUS_DIR / "eval_id")
    eval_ood_records = _passthrough_records(BENCHCORPUS_DIR / "eval_ood")  # CLINC150 -- untouched, no vision added

    vision_train, vision_eval_id = [], []
    vision_mode = "none"
    if vision_source is not None:
        print(f"loading real vision-source records from {vision_source}...")
        vision_train, vision_eval_id = load_vision_source_records(vision_source)
        vision_mode = "real_vision_source"
    elif smoke:
        print(f"generating {n_smoke_images} synthetic smoke-test images (mechanism-only, not real eval data)...")
        images_dir = out_dir / "smoke_images"
        n_train_img = int(n_smoke_images * 0.75)
        n_eval_img = n_smoke_images - n_train_img
        vision_train = build_smoke_vision_records(n_train_img, images_dir, rng)
        vision_eval_id = build_smoke_vision_records(n_eval_img, images_dir, rng)
        vision_mode = "synthetic_smoke"
        if n_smoke_text:
            # Trim the text side down for a fast, cheap smoke run too -- this is
            # NOT the "keep 62,000 rows unchanged" invariant (that invariant is
            # about the REAL run's data, built with no --smoke/--n-smoke-text
            # flags at all; see the top-level run recipe with no --smoke).
            train_records = rng.sample(train_records, min(n_smoke_text, len(train_records)))
            eval_id_records = rng.sample(eval_id_records, min(max(1, n_smoke_text // 10), len(eval_id_records)))

    all_vision = vision_train + vision_eval_id
    all_text = [r for r in (train_records + eval_id_records) if not r.get("images")]
    _assert_within_token_budget(all_vision, all_text, base_model, max_length, max_pixels)

    train_records = train_records + vision_train
    eval_id_records = eval_id_records + vision_eval_id
    rng.shuffle(train_records)
    rng.shuffle(eval_id_records)

    for name, records in [("train", train_records), ("eval_id", eval_id_records), ("eval_ood", eval_ood_records)]:
        _validate(records, name)

    Dataset.from_list(train_records).save_to_disk(str(out_dir / "train"))
    Dataset.from_list(eval_id_records).save_to_disk(str(out_dir / "eval_id"))
    Dataset.from_list(eval_ood_records).save_to_disk(str(out_dir / "eval_ood"))

    def by_type_and_source(records):
        return {
            "by_question_type": dict(collections.Counter(r["question_type"] for r in records)),
            "n_sources": len(set(r["source"] for r in records)),
            "n_with_images": sum(1 for r in records if r.get("images")),
        }

    stats = {
        "vision_mode": vision_mode,
        "vision_source": str(vision_source) if vision_source else None,
        "smoke": smoke,
        "train": {"n": len(train_records), **by_type_and_source(train_records)},
        "eval_id": {"n": len(eval_id_records), **by_type_and_source(eval_id_records)},
        "eval_ood": {
            "n": len(eval_ood_records), **by_type_and_source(eval_ood_records),
            "note": "CLINC150 wide zero-shot-schema choice eval, byte-identical pass-through from "
                    "benchcorpus_slice -- no vision rows are ever added here, by design (PRD 5.2b).",
        },
        "benchcorpus_source_dir": str(BENCHCORPUS_DIR),
        "base_model_used_for_token_budget_check": base_model if all_vision else None,
        "max_length_used_for_token_budget_check": max_length if all_vision else None,
        "max_pixels_used_for_token_budget_check": max_pixels if all_vision else None,
        "seed": seed,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="add synthetic mechanism-only vision rows")
    ap.add_argument("--n-smoke-images", type=int, default=500)
    ap.add_argument("--n-smoke-text", type=int, default=0, help="if set, also subsample the text side to this "
                     "many rows for a fast smoke run (0 = keep all 62,000, the real-run default)")
    ap.add_argument("--vision-source", default=None, help="path to a real vision-source JSONL (see module docstring "
                     "for the expected contract)")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-length", type=int, default=VISION_MAX_LENGTH)
    ap.add_argument("--max-pixels", type=int, default=VISION_MAX_PIXELS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    stats = build(
        Path(args.out_dir),
        smoke=args.smoke,
        n_smoke_images=args.n_smoke_images,
        n_smoke_text=args.n_smoke_text,
        vision_source=Path(args.vision_source) if args.vision_source else None,
        base_model=args.base_model,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
