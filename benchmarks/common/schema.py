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
