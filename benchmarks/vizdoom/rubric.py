"""
benchmarks/vizdoom/rubric.py -- PRD.md 8.1d Finding 3 and Finding 4.

Two things live here:

1. **The two rubric conditions**, `--rubric von` and `--rubric none`
   (`get_choice`). Von's own `get_doom_question()` embeds a complete
   four-clause decision rule in `instructions`, plus a per-option `criteria`
   dict restating it -- a model that simply follows the instructions solves
   the task without any learned game understanding (PRD.md 8.1d Finding 3).
   `--rubric von` reproduces that text verbatim (vendored Apache-2.0 data,
   see `../NOTICE.md`); `--rubric none` is a bare task description with no
   if/then rule, run and published alongside it so the gap between the two
   -- the most interesting single quantity this benchmark produces -- is
   measured, not hidden by only ever publishing one condition.

2. **The rubric-oracle policy** (`rubric_action`), PRD.md 8.1d Finding 4:
   Von's own rule is a deterministic function of the snapshot (it keys only
   on whether a monster/medkit is visible and its quantised horizontal
   bucket, both of which `env.py`/`observe.py` already compute), so the
   harness can evaluate the rule itself, per tick, with no model at all.
   This is (a) a zero-cost, CPU-only, no-GPU baseline policy
   (`benchmarks.common.episodic.OraclePolicy` runs it directly), and (b) the
   "gold label" a real policy's per-decision `rubric_agreement` is scored
   against, since ViZDoom itself has no gold label for what the *right*
   action is (`benchmarks/common/episodic.py`'s whole reason to exist).

**Protocol conversion, the one place this project is not byte-identical to
Von (recorded per PRD.md 8.1d):** Von's own `Choice` type carries `criteria`
as an option->criterion mapping; this project's `DecoderMultischemaBackend`/
`DecoderVisionMultischemaBackend` take a flat `options: list[str]` plus a
free-text `instructions` string. Conversion: `options = list(criteria.keys())`
in declared order (which already matches each scenario's own
`SCENARIO_ACTIONS` order -- verified by inspection, not assumed), and each
criterion is appended to `instructions` as an `"<option>: <criterion>"` line
so no text is dropped. This conversion is deterministic and a no-op under
`--rubric none` (which carries no `criteria` at all).
"""

from __future__ import annotations

from benchmarks.vizdoom.env import SCENARIO_ACTIONS, DoomSnapshot
from benchmarks.vizdoom.observe import format_position

# --- Vendored verbatim from Von's doom_eval.py (Apache-2.0, see NOTICE.md) --

_DEFEND_INSTRUCTIONS = (
    "Arena combat decision rule: "
    "1. If an enemy is visible dead center in the crosshair, select attack. "
    "2. If an enemy is visible on the left side of the screen, select turn left. "
    "3. If an enemy is visible on the right side of the screen, select turn right. "
    "4. If no enemies are visible in field of view, select turn left to sweep the arena."
)

_DEFEND_CRITERIA = {
    "attack": "An enemy is visible dead center on the crosshair. Fire weapon immediately.",
    "turn left": "Target is on the left side, or no active enemies are visible in view; rotate view left to search.",
    "turn right": "Target is visible on the right side of the view; rotate view right to center target.",
}

_HEALTH_INSTRUCTIONS = (
    "Acid survival navigation rules: "
    "1. If a medkit is located dead center or straight ahead, select move forward. "
    "2. If a medkit is located on the left side of the screen, select turn left. "
    "3. If a medkit is located on the right side of the screen, select turn right. "
    "4. If facing a wall right in front or no medkits visible, select turn right to explore open space."
)

_HEALTH_CRITERIA = {
    "move forward": "A medkit is straight ahead, or wide open space ahead with no obstacle in front; advance forward.",
    "turn left": "A medkit is visible on the left side; rotate left toward it.",
    "turn right": "A medkit is visible on the right side, or facing a wall obstacle ahead; rotate right toward open room.",
}

_VON_INSTRUCTIONS = {"defend_the_center": _DEFEND_INSTRUCTIONS, "health_gathering": _HEALTH_INSTRUCTIONS}
_VON_CRITERIA = {"defend_the_center": _DEFEND_CRITERIA, "health_gathering": _HEALTH_CRITERIA}

# Verified at import time, not just asserted in prose: the protocol
# conversion's "options = list(criteria.keys()) already matches
# SCENARIO_ACTIONS order" claim above.
for _scenario, _criteria in _VON_CRITERIA.items():
    assert list(_criteria.keys()) == SCENARIO_ACTIONS[_scenario], (
        f"{_scenario}: criteria key order {list(_criteria.keys())} != SCENARIO_ACTIONS "
        f"{SCENARIO_ACTIONS[_scenario]} -- the 'no reordering needed' claim in this module's "
        "docstring would be false."
    )

# --- `--rubric none`: bare task description, no if/then rule ---------------

_NONE_INSTRUCTIONS = {
    "defend_the_center": (
        "You are controlling a marine stationed in a central turret in a ViZDoom arena. Your goal "
        "is to survive and kill as many enemies as possible. Choose the action to take next, given "
        "the situation described below."
    ),
    "health_gathering": (
        "You are controlling a marine navigating a room with acid-damage terrain in ViZDoom. Your "
        "goal is to survive as long as possible by finding medkits and avoiding hazards. Choose the "
        "action to take next, given the situation described below."
    ),
}

RUBRIC_MODES = ("von", "none")


def get_choice(scenario: str, rubric: str) -> tuple[list[str], str]:
    """Returns `(options, instructions)` ready to pass straight to
    `backend.predict_choice(state, options, instructions=instructions)`.
    `options` is always `SCENARIO_ACTIONS[scenario]`, regardless of rubric
    mode -- the rubric only changes what the model is *told*, never what it
    can choose from."""
    if scenario not in SCENARIO_ACTIONS:
        raise ValueError(f"unknown scenario {scenario!r}")
    options = list(SCENARIO_ACTIONS[scenario])

    if rubric == "von":
        criteria = _VON_CRITERIA[scenario]
        criteria_lines = "\n".join(f"{opt}: {criteria[opt]}" for opt in options)
        instructions = f"{_VON_INSTRUCTIONS[scenario]}\n{criteria_lines}"
    elif rubric == "none":
        instructions = _NONE_INSTRUCTIONS[scenario]
    else:
        raise ValueError(f"unknown rubric {rubric!r} -- expected one of {RUBRIC_MODES}")

    return options, instructions


def rubric_action(snap: DoomSnapshot, scenario: str) -> str:
    """The deterministic oracle: evaluates Von's own 4-clause rule directly
    against a snapshot, independent of any model or rubric-text framing. The
    rule keys on the *nearest* (largest-apparent-size) relevant thing in
    view -- `env.py`'s `DoomSnapshot.things` is already sorted largest-first,
    so `things[0]` (after filtering by kind) is exactly that."""
    if scenario == "defend_the_center":
        monsters = [t for t in snap.things if t.kind == "monster"]
        if not monsters:
            return "turn left"  # clause 4: sweep the arena
        pos = format_position(monsters[0].cx)
        if pos == "dead center on the crosshair":
            return "attack"
        if pos in ("on the far left", "on the left"):
            return "turn left"
        return "turn right"  # "on the right" / "on the far right"

    if scenario == "health_gathering":
        items = [t for t in snap.things if t.kind == "item"]
        if not items:
            return "turn right"  # clause 4: no medkits visible, explore
        pos = format_position(items[0].cx)
        if pos == "dead center on the crosshair":
            return "move forward"
        if pos in ("on the far left", "on the left"):
            return "turn left"
        return "turn right"  # "on the right" / "on the far right"

    raise ValueError(f"unknown scenario {scenario!r}")
