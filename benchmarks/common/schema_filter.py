"""
The single most important piece of logic in benchmarks/ -- see PRD.md Section
10a and serve/inference.py's ChoiceUnsupportedError. The Phase 1 checkpoint
answers exactly one fixed 3-way schema: a `choice` question whose `options`
list is literally ["entailment", "neutral", "contradiction"]. This module
decides, for a list of real benchmark items, which ones the checkpoint could
legitimately even attempt -- and it is the only place that decision is
allowed to happen, so every harness under benchmarks/ goes through it rather
than re-implementing (and possibly getting wrong) the check.

Nothing here ever forces a wrong-schema item onto the fixed head, and nothing
here silently drops an item either. An item either matches exactly and is
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
