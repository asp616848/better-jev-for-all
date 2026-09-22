"""
Canonical in-memory representation of a single benchmark decision item, shared
by every benchmark loader under benchmarks/ (JevBench, jabr-v2, and whatever
gets added later). One shape in, one shape out, so benchmarks/common/* only
has to be written once and every harness downstream (schema filter, backends,
evidence writer) works the same way regardless of which benchmark it came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Item:
    """One frozen decision from a benchmark's public dataset (or a synthetic
    self-test fixture — see fixtures/ under each benchmark dir)."""

    benchmark: str            # "jevbench" | "jabr_v2"
    source_split: str         # provenance breadcrumb, e.g. "hard" or "v2:grammar_issue"
    item_id: str              # the benchmark's own item/task+case id
    question_type: str        # "choice" | "noul" | "score" -- only "choice" can ever be
                               # attempted by the Phase 1 checkpoint (PRD.md 10a)
    state: str
    instructions: str | None
    options: list[str] | None  # only meaningful for question_type == "choice"
    expected: Any               # gold label, exactly as the source dataset encodes it
                                 # (str for choice, bool for noul, int for score)

    def canonical_json(self) -> str:
        """Deterministic JSON used for hashing: sorted keys, no incidental
        whitespace, so the same item always hashes the same way regardless of
        how its source dict happened to be ordered."""
        payload = {
            "benchmark": self.benchmark,
            "source_split": self.source_split,
            "item_id": self.item_id,
            "question_type": self.question_type,
            "state": self.state,
            "instructions": self.instructions,
            "options": self.options,
            "expected": self.expected,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
