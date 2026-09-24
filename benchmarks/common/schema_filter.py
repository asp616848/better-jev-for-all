"""
The single most important piece of logic in benchmarks/ -- see PRD.md Section
10a and serve/inference.py's ChoiceUnsupportedError. The Phase 1 encoder
checkpoint answers exactly one fixed 3-way schema: a `choice` question whose
`options` list is literally ["entailment", "neutral", "contradiction"].
`filter_supported()` below decides, for a list of real benchmark items, which
ones that fixed-head checkpoint could legitimately even attempt -- and it is
the only place that decision is allowed to happen, so every harness under
benchmarks/ goes through it rather than re-implementing (and possibly getting
wrong) the check. Left exactly as it was (PRD.md 13a.1/13a.3's fixed-schema
result is still real, still valid evidence, and still worth being able to
reproduce byte-for-byte) -- nothing below this docstring for that function
changed when the multischema filter was added.

`filter_supported_multischema()` is the analogous decision for the decoder/
multischema path (PRD.md Section 14 Q4, 13a.5): a model whose restricted-
logit mechanism can answer ANY `choice`, `noul`, or `score` question with
between 2 and `max_options` options (see benchmarks/common/schema.py's
DECODER_MULTISCHEMA_MAX_OPTIONS docstring for where that ceiling comes from
and why it isn't just assumed to be 26). Rejects items with more options
than the cap, or whose gold `expected` label isn't even present in their own
`options` list (a data-integrity problem no filter should silently paper
over).

**Fixed 2026-09-24**: this used to reject every `noul`/`score` item outright
with the comment "no trained model exists for either" -- true when this was
first written, false since better-jev-for-all PRD.md 13a.8 (noul/score both
trained; 95.34%/81.92% as of 13a.14). Both primitives are internally just
lettered choices to this project's restricted-logit mechanism (noul: a
synthesized 2-way Yes/No choice; score: an N-way choice over the ordinal
scale, in the scale's own given order, never reshuffled) -- exactly the same
shape `filter_supported_multischema` already validates for `choice`, so no
new logic was needed here, only removing the type check that blocked it.
The loaders (`benchmarks/jevbench/loader.py`, `benchmarks/jabr_v2/loader.py`)
were fixed alongside this to actually populate `options`/`expected` for
`noul`/`score` items -- jabr-v2's previously returned `options=None` for
both (see that loader's own docstring for the real fix).

Nothing in either function ever forces an unanswerable item onto a model, and
neither silently drops an item. An item either is answerable and is
attempted, or it is marked unsupported, counted, and reported by reason.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from benchmarks.common.items import Item


@dataclass
class SchemaFilterResult:
    supported: list[Item]
    unsupported: list[Item]
    unsupported_reasons: Counter = field(default_factory=Counter)


def filter_supported(items: list[Item], model_labels: list[str]) -> SchemaFilterResult:
    supported: list[Item] = []
    unsupported: list[Item] = []
    reasons: Counter = Counter()

    for item in items:
        if item.question_type != "choice":
            reasons[f"question_type={item.question_type!r} (checkpoint only has a 'choice' head)"] += 1
            unsupported.append(item)
            continue
        if not item.options:
            reasons["'choice' item has no options/labels recorded"] += 1
            unsupported.append(item)
            continue
        if list(item.options) == list(model_labels):
            supported.append(item)
        else:
            reasons[f"'choice' options != checkpoint's fixed schema {model_labels}"] += 1
            unsupported.append(item)

    return SchemaFilterResult(supported=supported, unsupported=unsupported, unsupported_reasons=reasons)


def filter_supported_multischema(items: list[Item], max_options: int) -> SchemaFilterResult:
    """Decides which items a decoder/multischema `choice` model could
    legitimately attempt: any `choice` item with between 2 and `max_options`
    options inclusive, whose gold `expected` label is actually one of them.
    `score` and `noul` are rejected outright -- no model exists for either,
    same as the fixed-schema filter above. Unlike `filter_supported()`, this
    never compares the option *text* to anything -- only the option *count*
    -- because the whole point of this model family is that it doesn't need
    to have seen a particular option set during training."""
    supported: list[Item] = []
    unsupported: list[Item] = []
    reasons: Counter = Counter()

    for item in items:
        if item.question_type not in ("choice", "noul", "score"):
            reasons[f"question_type={item.question_type!r} (not choice/noul/score)"] += 1
            unsupported.append(item)
            continue
        if not item.options:
            reasons[f"{item.question_type!r} item has no options/labels recorded"] += 1
            unsupported.append(item)
            continue
        n = len(item.options)
        if n < 2:
            reasons[f"{item.question_type!r} item has fewer than 2 options ({n})"] += 1
            unsupported.append(item)
            continue
        if n > max_options:
            reasons[f"{item.question_type!r} item has {n} options > max_options cap ({max_options})"] += 1
            unsupported.append(item)
            continue
        if item.expected not in item.options:
            reasons[f"{item.question_type!r} item's gold 'expected' label is not in its own 'options' list (data integrity)"] += 1
            unsupported.append(item)
            continue
        supported.append(item)

    return SchemaFilterResult(supported=supported, unsupported=unsupported, unsupported_reasons=reasons)
