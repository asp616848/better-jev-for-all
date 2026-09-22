"""Loads the synthetic self-test fixture (selftest.jsonl, real JevBench record
shape, NOT real JevBench data) through the real loader, so --selftest also
exercises the actual parsing path, not a shortcut around it."""

from __future__ import annotations

from pathlib import Path

from benchmarks.common.items import Item
from benchmarks.jevbench.loader import load_file

SELFTEST_PATH = Path(__file__).resolve().parent / "selftest.jsonl"


def load_selftest_items() -> list[Item]:
    return load_file(SELFTEST_PATH)
