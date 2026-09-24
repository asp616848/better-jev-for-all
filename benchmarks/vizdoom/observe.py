"""
benchmarks/vizdoom/observe.py -- byte-faithful reimplementation of Von's
`format_doom_state()` (`doom_eval.py`), the text-arm observation builder
(PRD.md 8.1d: "Observation is text, not vision" -- labels buffer + depth
buffer + 3 game variables, serialized to a sentence, no pixels ever read for
content). `observe_frame()` at the bottom is the vision arm's future
observation builder (PRD.md 8.1d step 8, not built yet -- see its own
docstring for why it is a stub here and what it will need to become real).

The five position buckets, four size buckets, and four depth buckets below
are vendored verbatim from Von's `format_position`/`format_range`/
`format_depth` (Apache-2.0 -- see `../NOTICE.md`); unit-tested against
hand-built snapshots in `benchmarks/vizdoom/test_observe.py`-equivalent
coverage inside this module's own `if __name__` self-check (see bottom).
"""

from __future__ import annotations

from typing import Optional

from benchmarks.vizdoom.env import PRETTY_NAMES, DoomSnapshot


def format_position(cx: float) -> str:
    """Five buckets over the horizontal center-of-frame fraction cx in [0, 1]."""
    if cx < 0.20:
        return "on the far left"
    if cx < 0.44:
        return "on the left"
    if cx <= 0.56:
        return "dead center on the crosshair"
    if cx <= 0.80:
        return "on the right"
    return "on the far right"


def format_range(size: float) -> str:
    """Four buckets over apparent size (bounding-box height / frame height)."""
    if size > 0.45:
        return "point blank"
    if size > 0.25:
        return "close"
    if size > 0.12:
        return "at medium range"
    return "far away"


def format_depth(d: float) -> str:
    """Four buckets over a depth-buffer band's median value."""
    if d < 12:
        return "a solid wall right in front"
    if d < 24:
        return "a wall nearby"
    if d < 45:
        return "some room to maneuver"
    return "wide open space"


def observe_text(snap: DoomSnapshot, scenario: str, last_action: Optional[str] = None) -> str:
    """Renders one tick's `DoomSnapshot` into the sentence Von's own model
    actually reads. Top 3 monsters / top 2 items by apparent size (matching
    `DoomSnapshot.things` already being sorted largest-first by `env.py`), a
    one-step memory of the previous action, and a per-scenario preamble +
    status line. This function's output is exactly what gets passed as
    `state` to `backend.predict_choice()` -- see `benchmarks/vizdoom/run.py`.
    """
    monsters = [t for t in snap.things if t.kind == "monster"][:3]
    items = [t for t in snap.things if t.kind == "item"][:2]

    parts: list[str] = []
    if scenario == "defend_the_center":
        parts.append("Arena Combat Situation: Stationed in central turret facing incoming demons.")
        if monsters:
            for m in monsters:
                p_name = PRETTY_NAMES.get(m.name, m.name.lower())
                parts.append(f"A {p_name} is visible {format_position(m.cx)}, {format_range(m.size)}.")
        else:
            parts.append("No active enemies visible in field of view.")
        parts.append(f"Status: Health {snap.health}, Ammo {snap.ammo}, Total Kills: {snap.kills}.")
    elif scenario == "health_gathering":
        parts.append("Survival Situation: Acidic terrain inflicts continuous damage. Medkits required.")
        if items:
            for item in items:
                p_name = PRETTY_NAMES.get(item.name, item.name.lower())
                parts.append(f"A {p_name} is located {format_position(item.cx)}, {format_range(item.size)}.")
        else:
            parts.append("No medkits visible in field of view.")
        parts.append(
            f"Obstacle Navigation: Ahead: {format_depth(snap.depth_center)}. "
            f"Left: {format_depth(snap.depth_left)}. Right: {format_depth(snap.depth_right)}."
        )
        parts.append(f"Status: Health {snap.health}.")
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    if last_action:
        parts.append(f"Previous executed action: {last_action}.")
    return " ".join(parts)


def observe_frame(snap: DoomSnapshot, out_dir, tick: int) -> tuple[str, str]:
    """**Not implemented -- PRD.md 8.1d step 8, the vision arm, built after
    `checkpoints/ekvachan-decoder-qwen-vision` (13a.11) lands and
    ScreenSpot-v2 (13a.12) has produced its first real number.** Left as a
    documented stub rather than omitted so `benchmarks/vizdoom/run.py`'s
    `--arm vision` branch has a single, obvious place to land real code
    later: write `snap.frame` (populated when the environment was
    constructed with `capture_frames=True`) to a PNG under `out_dir`, and
    return a short *status-line-only* text state (no label/depth prose --
    per PRD.md 8.1d step 8, prose describing what's in the image defeats the
    point of a vision arm; the model must read the pixels itself).

    The frame-writing half of this already exists and is exercised today:
    `benchmarks/common/episodic.py::DaggerSink` writes the exact same
    `snap.frame` array to PNG for the on-policy (text, frame, action) teacher
    data (PRD.md 8.1d Finding 4 point 3) -- this function will reuse that
    same PNG-writing path once the vision arm is built, not reinvent it.
    """
    raise NotImplementedError(
        "observe_frame() is PRD.md 8.1d step 8 (the vision arm), scoped out of this build -- "
        "see this function's docstring and benchmarks/vizdoom/README.md's 'What this harness "
        "does not do yet'. benchmarks/common/episodic.py's DaggerSink already writes real frame "
        "PNGs from the text arm's own rollouts, which is what this function will read from."
    )


if __name__ == "__main__":
    # A tiny hand-built-snapshot self-check, run directly (`python -m
    # benchmarks.vizdoom.observe`) with no ViZDoom engine involved -- checks
    # the bucket boundaries themselves, per PRD.md 8.1d's implementation
    # checklist step 4 ("unit-test the bucket boundaries against hand-built
    # snapshots").
    from benchmarks.vizdoom.env import Thing

    assert format_position(0.0) == "on the far left"
    assert format_position(0.19) == "on the far left"
    assert format_position(0.20) == "on the left"
    assert format_position(0.43) == "on the left"
    assert format_position(0.44) == "dead center on the crosshair"
    assert format_position(0.56) == "dead center on the crosshair"
    assert format_position(0.561) == "on the right"
    assert format_position(0.80) == "on the right"
    assert format_position(0.801) == "on the far right"
    assert format_position(1.0) == "on the far right"

    assert format_range(0.46) == "point blank"
    assert format_range(0.26) == "close"
    assert format_range(0.13) == "at medium range"
    assert format_range(0.05) == "far away"

    assert format_depth(0.0) == "a solid wall right in front"
    assert format_depth(11.9) == "a solid wall right in front"
    assert format_depth(12.0) == "a wall nearby"
    assert format_depth(23.9) == "a wall nearby"
    assert format_depth(24.0) == "some room to maneuver"
    assert format_depth(44.9) == "some room to maneuver"
    assert format_depth(45.0) == "wide open space"

    snap_no_monsters = DoomSnapshot(health=100, ammo=26, kills=2, tic=140, things=[])
    text = observe_text(snap_no_monsters, "defend_the_center", last_action="attack")
    assert "No active enemies visible" in text
    assert "Health 100, Ammo 26, Total Kills: 2" in text
    assert text.endswith("Previous executed action: attack.")

    snap_with_monster = DoomSnapshot(
        health=80, ammo=10, kills=1, tic=80,
        things=[Thing(name="DoomImp", kind="monster", cx=0.5, size=0.3)],
    )
    text2 = observe_text(snap_with_monster, "defend_the_center")
    assert "A imp is visible dead center on the crosshair, close." in text2

    snap_health = DoomSnapshot(
        health=60, ammo=0, kills=0, tic=200, things=[Thing(name="Medikit", kind="item", cx=0.1, size=0.2)],
        depth_left=5.0, depth_center=50.0, depth_right=30.0,
    )
    text3 = observe_text(snap_health, "health_gathering")
    assert "medkit is located on the far left, at medium range" in text3
    assert "Ahead: wide open space. Left: a solid wall right in front. Right: some room to maneuver." in text3

    print("observe.py self-check OK")
