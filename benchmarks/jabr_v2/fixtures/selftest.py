"""Loads the synthetic self-test fixture (selftest.toml, real jabr-v2 task/case
shape, NOT real jabr-v2 data) through the real TOML loader, so --selftest also
exercises the actual parsing path, not a shortcut around it."""

from __future__ import annotations

from pathlib import Path

from benchmarks.common.items import Item
from benchmarks.jabr_v2.loader import load_toml_file

SELFTEST_PATH = Path(__file__).resolve().parent / "selftest.toml"


def load_selftest_items() -> list[Item]:
    return load_toml_file(SELFTEST_PATH, "selftest")
