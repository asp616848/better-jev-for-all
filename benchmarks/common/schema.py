"""
The fixed `choice` schema the Phase 1 `ekvachan-base` checkpoint was trained
on. Canonical source: serve/inference.py's `LABELS` constant.

Duplicated here as a plain literal, rather than imported, because
serve/inference.py imports torch and transformers at module level -- and
deciding which benchmark items are even answerable is cheap, dataset-only
work that has nothing to do with running the model. It needs to work (and
needs to run in CI or a laptop with no GPU/ML stack installed) well before
any backend that can actually answer a question gets instantiated. See
benchmarks/common/backends.py and benchmarks/common/harness.py for how the
two are kept apart on purpose.

Kept honest against drift: every real backend (InProcessBackend, HTTPBackend
in benchmarks/common/backends.py) is checked against this constant before
it's trusted to score anything filtered against it -- see
benchmarks/common/harness.py's consistency check. If serve/inference.py's
LABELS ever changes, that check fails loudly instead of this file silently
going stale.
"""

FIXED_CHECKPOINT_LABELS = ["entailment", "neutral", "contradiction"]

# --- Decoder / multischema path (PRD.md Section 14 Q4, 13a.5) ---------------
#
# The decoder (Qwen3.5-4B LoRA, restricted-logit read) is architecturally not
# fixed-width the way the encoder's classification head is: it answers a
# `choice` question by reading the model's next-token logits restricted to a
# small set of single-uppercase-letter tokens (A, B, C, ...), one per option,
# masking the rest to exact -inf before softmax. That mechanism's real ceiling
# is however many such letters exist as guaranteed single, mutually distinct
# tokens under the base model's tokenizer -- verified (independently, on the
# actual training server, 2026-09-23) to be all 26 of A-Z under Qwen3.5-4B's
# tokenizer. See `training/train_decoder_lora_wideschema.py`'s `MAX_OPTIONS`
# constant and its own runtime assertion (`assert len(ids) == 1` for every
# letter), which re-checks this live on every training run -- that script is
# the canonical source for this number, duplicated here as a plain literal for
# exactly the reason FIXED_CHECKPOINT_LABELS above is: filtering benchmark
# items by option count is cheap, dataset-only work that must not require
# torch/transformers/peft to be installed, and that script isn't even vendored
# on this branch (it landed via a parallel workstream -- PRD.md 13a.5).
#
# Unlike FIXED_CHECKPOINT_LABELS, this is only a *default* cap used when
# filtering happens before any backend is instantiated (this module's whole
# reason to exist). A real DecoderMultischemaBackend reports its own loaded
# checkpoint's `max_options` from its manifest, and benchmarks/common/
# harness.py cross-checks that against whatever cap was used to filter before
# trusting the run -- same discipline as the FIXED_CHECKPOINT_LABELS check.
DECODER_MULTISCHEMA_MAX_OPTIONS = 26
