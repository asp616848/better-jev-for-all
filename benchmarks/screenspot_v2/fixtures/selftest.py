"""Loads the synthetic self-test fixture (selftest.jsonl, real bjb-export
record shape, NOT real ScreenSpot-v2 data) through the real reframing logic
in loader.reframe_records_to_items(), so --selftest also exercises the
actual grouping/pool-reconstruction path, not a shortcut around it.

Two tiny synthetic PNGs live next to this file (selftest_screenshot.png,
16x16; selftest_screenshot_b.png, 12x12) -- not real screenshots, but real
image files with real, checked-in bytes, so a --selftest run also proves the
sha256 hash-verification path in loader.py actually runs, not just the
happy-path grouping logic. Three of the four synthetic rows share the first
image (giving it a real 3-member candidate pool, exercising genuine N=3
reconstruction); the fourth row is alone on the second image with a
degenerate (label == distractor) options pair, deliberately constructed so
its reconstructed pool collapses to a single distinct string -- exercising
loader.py's "too few annotated elements" exclusion path, which the real
858-item dataset never triggers (see loader.py's module docstring)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.common.items import Item
from benchmarks.screenspot_v2.loader import reframe_records_to_items

FIXTURES_DIR = Path(__file__).resolve().parent
SELFTEST_PATH = FIXTURES_DIR / "selftest.jsonl"

_FIXTURE_IMAGES = {
    "13025b586663efd2db4300c648126113b04f91cc1a3f422486f870568142a432": FIXTURES_DIR / "selftest_screenshot.png",
    "68b2c253186ccd2dde5a459570f6433a47526f64533c90656bee2b0123117edc": FIXTURES_DIR / "selftest_screenshot_b.png",
}


def _resolve_fixture_image(sha256: str, media_type: str) -> Path:
    assert media_type == "image/png", f"selftest fixture images are all PNG, got {media_type!r}"
    path = _FIXTURE_IMAGES[sha256]
    real_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if real_hash != sha256:
        raise RuntimeError(f"fixture image {path} hashes to {real_hash}, expected {sha256}")
    return path


def load_selftest_items() -> tuple[list[Item], dict]:
    records = [json.loads(line) for line in SELFTEST_PATH.read_text().splitlines() if line.strip()]
    return reframe_records_to_items(records, _resolve_fixture_image, "selftest:synthetic")
