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

# The variable-width sibling of FIXED_CHECKPOINT_LABELS above: the decoder/restricted-logit
# checkpoints (training/train_decoder_lora_multischema.py, train_decoder_lora_wideschema.py)
# are not tied to one fixed option list -- they read whichever single-uppercase-letter token
# (A-Z) corresponds to the item's own option count, up to this cap. 26 is the real ceiling of
# a single-uppercase-letter restricted-logit scheme (one token per letter), independently
# verified against Qwen3.5-4B's tokenizer by train_decoder_lora_wideschema.py's own runtime
# assertion (every letter A-Z is confirmed a single, mutually distinct token before that script
# trains), not assumed here.
DECODER_MULTISCHEMA_MAX_OPTIONS = 26
