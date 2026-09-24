"""
Shared library for the decoder-LoRA restricted-logit training lineage.

Every script from `train_decoder_lora.py` through `train_decoder_lora_benchcorpus.py`
(five scripts, 13a.1-13a.10) re-implemented the same ~250 lines: prompt
construction, the shuffled-vs-ordinal option order, the collate function, the
`labels[:, last_col]` masking trick, the OOM-skip training loop, temperature
fitting, and `_report_by_key`. That was the right call *while the mechanism
was still being proven* -- PRD.md's own fork-don't-edit policy exists so each
published result (13a.1-13a.10) stays reproducible from its own frozen script,
forever, and none of those five files are touched by this one.

PRD 5.2b changes what's true going forward: the mechanism is proven (see
`training/probe_vision_path.py` and `results/vision-path-probe-20260923T201758Z
.manifest.json`), and the owner gave an explicit standing instruction for all
*future* work: "try to keep things general to vision or non-vision model for
your changes" -- i.e. stop forking a sixth near-duplicate script. This module
is the extraction that makes that possible: the genuinely shared logic lives
here once, and `train_decoder_lora_general.py` (the new entrypoint that
replaces forking for all future training work) imports it. The vision-specific
bits -- processor vs. tokenizer, model class, pixel_values passthrough, the
vision-tower-freeze assertion -- are NOT here; they're thin and belong next to
the orchestration code that uses them.

One deliberate mechanism change from the five frozen scripts, made here and
explained once: those scripts called `tokenizer(prompts, truncation=True,
max_length=...)`, which truncates a too-long row *silently*. PRD 5.2b names
silent truncation as "the failure mode most likely to waste a night of GPU
time on this task." `make_collate()` below never truncates -- it pads to the
longest row in the batch and raises loudly if that exceeds `max_length`. This
is strictly safer than what it replaces and is exercised by the regression
check documented in `train_decoder_lora_general.py`'s docstring.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from eval.metrics import full_report  # noqa: F401  (re-exported for callers)
from eval.metrics import fit_temperature  # noqa: F401  (re-exported for callers)

REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_OPTIONS = 26  # A-Z -- unchanged since train_decoder_lora_wideschema.py
LETTERS = [chr(ord("A") + i) for i in range(MAX_OPTIONS)]


def shuffled_letter_order(pair_seed: str, n_options: int) -> list[int]:
    """Unchanged since train_decoder_lora_wideschema.py -- still used for choice/noul rows."""
    rng = np.random.default_rng(int(hashlib.sha1(pair_seed.encode()).hexdigest()[:8], 16))
    order = list(range(n_options))
    rng.shuffle(order)
    return order


def option_order_for(ex: dict) -> list[int]:
    """`score` rows keep the fixed ascending scale order (never shuffled -- the
    letter position *is* the scale position, see build_primitives_slice.py);
    every other question type gets a seeded shuffle. Unchanged logic, factored
    out of train_decoder_lora_benchcorpus.PromptDataset.__getitem__."""
    n_options = len(ex["options"])
    if ex.get("question_type") == "score":
        return list(range(n_options))
    return shuffled_letter_order(ex["state"] + str(ex["label_idx"]) + str(n_options), n_options)


def build_prompt_text(state: str, options: list[str], order: list[int], instructions: str) -> str:
    """Byte-identical to every prior script's `build_prompt()`. Returns the
    plain instruction text; the image block (if any) is prepended by the
    caller when constructing chat `messages`, not baked in here, so this stays
    usable for both a bare tokenizer and a processor's chat template."""
    n = len(options)
    option_lines = "\n".join(f"{LETTERS[i]}) {options[order[i]]}" for i in range(n))
    return (
        f"{instructions}\n\n{state}\n\nOptions:\n{option_lines}\n\n"
        f"Answer with a single letter ({'/'.join(LETTERS[:n])})."
    )


def build_prompt_messages(state: str, options: list[str], order: list[int], instructions: str,
                           image_path: str | None) -> list[dict]:
    """The one genuine generalization over the text-only lineage's build_prompt():
    when `image_path` is set, content becomes a list with an image block
    prepended, exactly the shape `probe_vision_path.py` verified end to end
    (`{"type": "image"}` renders to `<|vision_start|><|image_pad|><|vision_end|>`
    and the rest of the rendered prompt is byte-identical to the text-only
    case). When `image_path` is None, content is the plain string every prior
    script already produced -- verified in the same probe to render and tokenize
    identically through `AutoProcessor` as through `AutoTokenizer`."""
    text = build_prompt_text(state, options, order, instructions)
    if image_path is None:
        content: Any = text
    else:
        content = [{"type": "image"}, {"type": "text", "text": text}]
    return [{"role": "user", "content": content}]


class PromptDataset(TorchDataset):
    """Generalized over every prior script's PromptDataset: a row may carry an
    optional `images` field (a list of local file paths; the schema
    `better-jev-bench` PRD §13.4 commits to for its eventual `bjb export`
    image-bearing records). v1 scope, stated rather than silently assumed:
    at most one image per row -- our prompt shape only ever emits one image
    block. A row with more than one path is a hard error, not a silent drop.
    """

    def __init__(self, hf_dataset, resolve_image_root: Path | None = None):
        self.ds = hf_dataset
        self.resolve_image_root = resolve_image_root

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
                f"by the slice builder's validation; refusing to silently truncate."
            )
        images = ex.get("images") or []
        if len(images) > 1:
            raise ValueError(
                f"example at idx {idx} (source={ex.get('source')!r}) carries {len(images)} images; "
                f"this lineage's prompt shape supports at most one image per row (v1 scope)."
            )
        image_path = None
        if images:
            p = Path(images[0])
            if self.resolve_image_root is not None and not p.is_absolute():
                p = self.resolve_image_root / p
            if not p.exists():
                raise FileNotFoundError(
                    f"example at idx {idx} (source={ex.get('source')!r}) references image {p}, "
                    f"which does not exist -- refusing to train on a row whose image can't be resolved."
                )
            image_path = str(p)

        order = option_order_for(ex)
        messages = build_prompt_messages(ex["state"], options, order, ex["instructions"], image_path)
        target_letter_idx = order.index(ex["label_idx"])
        return {
            "messages": messages,
            "image_path": image_path,
            "target_letter_idx": target_letter_idx,
            "label_idx": ex["label_idx"],
            "n_options": n_options,
            "order": order,
            "source": ex.get("source", "unknown"),
            "question_type": ex.get("question_type", "choice"),
        }


def make_collate(processor, max_length: int, letter_ids: list[int], train: bool, truncate: bool = False):
    """Generalized collate: always goes through `processor` (an `AutoProcessor`,
    never a bare `AutoTokenizer` -- see the module docstring in
    train_decoder_lora_general.py for why unifying on the processor is the
    actual generalization, not a vision-only special case). A text-only row
    naturally emits no `pixel_values` (verified in probe_vision_path.py); a
    row with an image contributes exactly one PIL image, in the same order its
    text appears in the batch, matching the processor's own mixed-batch
    contract (`images=[...]` is a flat list matched against the batch's total
    `<|image_pad|>` placeholder count, not a per-row list).

    Default (`truncate=False`): never truncates. Pads to the longest row in
    the batch and raises loudly (naming the offending sources) if that
    exceeds `max_length`, instead of silently truncating -- see module
    docstring. `train_decoder_lora_general.py` always uses this default; it
    is a deliberate improvement over the five frozen scripts before it, which
    called `tokenizer(..., truncation=True)` and silently cut off whatever
    didn't fit. That silent truncation was real, not hypothetical: this
    collate's assertion is what *found* it, on 13a.10's own eval_id split
    (~380/4750 rows tokenize past 384) -- see
    `results/vision-class-swap-control-*.manifest.json`.

    `truncate=True`: reproduces the old scripts' exact mechanism
    (`processor(..., truncation=True, max_length=max_length)`). This exists
    for exactly one caller, `vision_class_swap_control.py` -- the control run
    has to reproduce 13a.10's *own* eval mechanism byte-for-mechanism so the
    only variable that changes is the model class, not also the truncation
    policy.
    """
    from PIL import Image

    def collate(batch):
        texts = [
            processor.apply_chat_template(b["messages"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            for b in batch
        ]
        pil_images = []
        for b in batch:
            if b["image_path"] is not None:
                pil_images.append(Image.open(b["image_path"]).convert("RGB"))

        enc = processor(
            text=texts,
            images=pil_images if pil_images else None,
            padding=True,
            truncation=truncate,
            max_length=max_length if truncate else None,
            return_tensors="pt",
        )

        seq_len = enc["input_ids"].shape[1]
        if not truncate and seq_len > max_length:
            offending = [(b["source"], b["question_type"]) for b in batch]
            raise RuntimeError(
                f"batch sequence length {seq_len} exceeds --max-length {max_length} -- refusing to "
                f"silently truncate (PRD 5.2b: this is the failure mode most likely to waste GPU "
                f"time). Sources in this batch: {offending}. Raise --max-length, or check the slice "
                f"builder's per-row image_tokens + text_tokens assertion."
            )

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


_BATCH_EXTRA_KEYS = ("target_letter_idx", "label_idx", "n_options", "order", "_sources", "_question_types")


def strip_batch_extras(batch: dict) -> dict:
    """Pop everything `make_collate` added beyond what `model(**batch)` accepts,
    *except* `labels` -- deliberately, since the training loop needs `labels`
    to stay in `batch` for `model(**batch)` to compute a loss. Returns the
    popped extras. Shared by the train loop and `evaluate()` so the set of
    extra keys is defined in exactly one place; `evaluate()` pops `labels`
    itself afterward (eval batches never carry one in the first place --
    `make_collate(..., train=False)` never sets it -- but the pop is
    defensive, matching every prior script's evaluate())."""
    extras = {}
    for k in _BATCH_EXTRA_KEYS:
        if k in batch:
            extras[k] = batch.pop(k)
    return extras


@torch.no_grad()
def evaluate(model, loader, device, letter_ids: list[int]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Unchanged mechanism from train_decoder_lora_benchcorpus.py: read
    `logits[:, -1, :]` restricted to the letter-token ids, mask columns past
    each row's real option count, and undo the shuffled order to report
    probabilities in canonical option order. Verified unchanged under a real
    mixed image+text batch in probe_vision_path.py."""
    model.eval()
    all_probs, all_labels, all_sources, all_qtypes = [], [], [], []
    letter_ids_t = torch.tensor(letter_ids, device=device)
    for batch in loader:
        extras = strip_batch_extras(batch)
        label_idx = extras["label_idx"]
        n_options = extras["n_options"]
        order = extras["order"]
        sources = extras["_sources"]
        qtypes = extras["_question_types"]
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


def report_by_key(probs: np.ndarray, labels: np.ndarray, keys: list[str]) -> dict:
    """Unchanged from train_decoder_lora_benchcorpus._report_by_key."""
    by_key = {}
    keys_arr = np.array(keys)
    for k in sorted(set(keys)):
        m = keys_arr == k
        by_key[k] = full_report(probs[m], labels[m], MAX_OPTIONS)
    return by_key


def assert_letter_tokens(tokenizer, base_model: str) -> list[int]:
    """Unchanged assertion from every prior script: every A-Z letter must be
    exactly one token under this tokenizer, or the restricted-logit mechanism
    (`logits[:, -1, letter_ids]`) is reading the wrong thing."""
    letter_token_ids_full = [tokenizer.encode(l, add_special_tokens=False) for l in LETTERS]
    for l, ids in zip(LETTERS, letter_token_ids_full):
        assert len(ids) == 1, f"letter {l!r} is not a single token under {base_model}'s tokenizer: {ids}"
    letter_ids = [ids[0] for ids in letter_token_ids_full]
    assert len(set(letter_ids)) == len(letter_ids)
    return letter_ids


def assert_vision_tower_frozen(peft_model) -> list[str]:
    """PRD 5.2b: the existing LoRA `target_modules` freeze the vision tower by
    *name mismatch*, not by an explicit freeze (the ViT blocks use
    `attn.qkv` / `mlp.linear_fc{1,2}`, which match nothing in
    `target_modules`). That was an accident worth making deliberate. Returns
    the list of LoRA module names for logging; raises if any of them landed
    in `model.visual`."""
    lora_mods = [n for n, _ in peft_model.named_modules() if n.endswith("lora_A")]
    in_vision = [n for n in lora_mods if "visual" in n]
    assert not in_vision, (
        f"{len(in_vision)} LoRA module(s) landed in the vision tower, which PRD 5.2b's probe found "
        f"should be structurally impossible with this project's target_modules: {in_vision[:5]}"
    )
    return lora_mods
