"""
Loader for JevBench's public dataset (github.com/fstandhartinger/jevbench,
MIT license -- verified 2026-09-22, see README.md in this directory).
Vendored copies of the exact files as fetched that day live in
benchmarks/jevbench/vendored/ -- see NOTICE.md for attribution and provenance.

Record format (one JSON object per line in each .jsonl file), confirmed by
actually downloading and parsing the real files, not assumed from the README:

  {"expected": "billing", "family": "policy", "group": "...", "id": "...",
   "labels": ["billing", "technical", "other"],
   "provenance": {"license": "MIT", "source": "...", ...},
   "question": {"type": "choice", "instructions": "...", "criteria": {...}},
   "split": "public", "state": "..."}

`labels` is the item's exact answer-option list; for "choice" items it is what
benchmarks.common.items.Item.options is filled from. "noul" and "score" items
also carry a `labels` field (e.g. ["no","yes"] or ["0","1","2","3"]) but that
list is not "options" in the sense the checkpoint's fixed schema cares about --
the schema filter (benchmarks/common/schema_filter.py) rejects them on
question_type before options are even compared.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.common.items import Item

DATA_DIR = Path(__file__).resolve().parent / "vendored"

PUBLIC_FILES = {
    "original": DATA_DIR / "original.jsonl",
    "easy": DATA_DIR / "easy.jsonl",
    "hard": DATA_DIR / "hard.jsonl",
}


def _record_to_item(rec: dict) -> Item:
    q = rec["question"]
    return Item(
        benchmark="jevbench",
        source_split=f"{rec.get('split', 'public')}:{rec.get('family', '?')}",
        item_id=rec["id"],
        question_type=q["type"],
        state=rec["state"],
        instructions=q.get("instructions"),
        options=list(rec["labels"]) if rec.get("labels") else None,
        expected=rec.get("expected"),
    )


def load_file(path: Path) -> list[Item]:
    items: list[Item] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(_record_to_item(json.loads(line)))
    return items


def load_public(splits: list[str] | None = None) -> list[Item]:
    """Load JevBench's public dataset splits. `splits` defaults to all three
    public files (original, easy, hard) -- the 231 of 534 total v1.2 items
    that are actually public. The other 303 (heldout, router, judge,
    easy-heldout, and 109 more hard items) are deliberately unpublished by
    upstream, by the same anti-gaming design PRD.md 3.1a/8.1a cites -- they
    cannot be vendored here because they are not public anywhere."""
    names = splits or list(PUBLIC_FILES)
    items: list[Item] = []
    for name in names:
        items.extend(load_file(PUBLIC_FILES[name]))
    return items
