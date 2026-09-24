"""
benchmarks/vizdoom/env.py -- DoomEnvironment + DoomSnapshot.

A from-scratch reimplementation of Von's `benchmarks/doom_eval.py::DoomEnvironment`
(`wfzyx/von@master`, fetched via the GitHub API 2026-09-24, 274 lines,
Apache-2.0 -- see `../NOTICE.md`), reproducing its ViZDoom configuration
exactly so a text-arm run here is a like-for-like comparison to Von's own
published numbers, not a reimagining of the protocol. See PRD.md 8.1d for
the full derivation of every decision below; this docstring only records
what a reader of this file specifically needs.

Reimplemented independently rather than copy-pasted from Von's file, on
purpose: writing this from the verified protocol table (not from reading
Von's source once and trusting it) forces every design choice -- button
order, buffer indexing, the labels-buffer name-suffix quirk -- to be
re-checked against the real, running engine. The `get_available_buttons()`
order assertion below exists for exactly this reason: a silent mismatch
there would map every action name to the wrong button and produce a
plausible, entirely wrong score, which is exactly the kind of bug that
reading code once (instead of running it) does not catch.

Reused verbatim as vendored Apache-2.0 protocol constants (see
`../NOTICE.md`): the two scenarios' action sets, the button mapping, the
published 8-seed lists per scenario, `tics=4`, the 300-step cap, and the
`MONSTERS`/`ITEMS`/`PRETTY_NAMES` name sets used to classify and label
labels-buffer entries. These are data, not code -- reproducing them
verbatim is the entire point of a like-for-like reproduction (PRD.md 7.4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import vizdoom as vzd

# --- Action vocabulary (vendored verbatim from Von's doom_eval.py) ---------

BUTTON_MAP: dict[str, "vzd.Button"] = {
    "attack": vzd.Button.ATTACK,
    "turn left": vzd.Button.TURN_LEFT,
    "turn right": vzd.Button.TURN_RIGHT,
    "move forward": vzd.Button.MOVE_FORWARD,
    "move backward": vzd.Button.MOVE_BACKWARD,
    "strafe left": vzd.Button.MOVE_LEFT,
    "strafe right": vzd.Button.MOVE_RIGHT,
}

SCENARIO_ACTIONS: dict[str, list[str]] = {
    "defend_the_center": ["attack", "turn left", "turn right"],
    "health_gathering": ["move forward", "turn left", "turn right"],
}

# Von's own published 8-seed lists (PRD.md 8.1d: "Seeds | 8 per scenario,
# published literally"). Using these, not arbitrary seeds, is the entire
# point of "same 8 shared seeds Von used" (PRD.md 8.1 item 2).
DEFEND_SEEDS: list[int] = [42106076, 42106077, 42106078, 42106079, 42106080, 42106081, 42106082, 42106083]
HEALTH_SEEDS: list[int] = [41006394, 41006395, 41006396, 41006397, 41006398, 41006399, 41006400, 41006401]

SCENARIO_SEEDS: dict[str, list[int]] = {
    "defend_the_center": DEFEND_SEEDS,
    "health_gathering": HEALTH_SEEDS,
}

# Labels-buffer classification sets, vendored verbatim (Von's doom_eval.py) --
# used to decide which ViZDoom actor names are "a monster" / "an item" worth
# describing in the text observation, and how to spell them in prose.
MONSTERS = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "LostSoul", "Cacodemon",
    "HellKnight", "BaronOfHell", "Arachnotron", "PainElemental", "Revenant", "Fatso", "Archvile",
    "SpiderMastermind", "Cyberdemon", "WolfensteinSS", "CommanderKeen", "MarineChainsaw", "MarineBFG",
}

ITEMS = {
    "GreenArmor", "BlueArmor", "ArmorBonus", "HealthBonus", "Medikit", "Stimpack", "Soulsphere",
    "Megasphere", "Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "RocketBox", "Cell", "CellPack",
    "Backpack", "Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle", "BFG9000", "Chainsaw",
}

PRETTY_NAMES = {
    "DoomImp": "imp", "ShotgunGuy": "shotgun zombie", "Zombieman": "zombie", "ChaingunGuy": "chaingunner",
    "Demon": "pinky demon", "Spectre": "spectre", "MarineChainsaw": "chainsaw marine",
    "Medikit": "medkit", "Stimpack": "stimpack", "HealthBonus": "health bonus",
}

TICS_PER_SECOND = 35.0  # ViZDoom's fixed simulation rate; survival_s = steps * tics / 35.0.


@dataclass
class Thing:
    """One labels-buffer entry, classified and positioned. `cx` is the
    horizontal center of its bounding box in [0, 1] (0 = left edge of frame,
    1 = right edge); `size` is its bounding-box height as a fraction of
    frame height -- both exactly Von's own normalization."""

    name: str
    kind: str  # "monster" | "item"
    cx: float
    size: float


@dataclass
class DoomSnapshot:
    """One tick's worth of game state -- everything `observe.py` and
    `rubric.py` need, and nothing more. `frame` is the raw RGB screen buffer,
    populated only when the environment was constructed with
    `capture_frames=True` (PRD.md 8.1d Finding 4 point 3's DAgger-teacher
    data path -- see `benchmarks/common/episodic.py`'s `DaggerSink`); every
    other consumer of a snapshot never looks at it, matching Von's own
    finding that the RGB frame is otherwise unused ("screen_buffer is read
    only to get H, W")."""

    health: int
    ammo: int
    kills: int
    tic: int
    things: list[Thing] = field(default_factory=list)
    depth_left: float = 0.0
    depth_center: float = 0.0
    depth_right: float = 0.0
    frame: Optional[np.ndarray] = None


class DoomEnvironment:
    """Stock `defend_the_center.cfg` / `health_gathering.cfg` (both ship
    inside the `vizdoom` wheel at `vizdoom.scenarios_path` -- nothing to
    download), configured exactly as Von's own `DoomEnvironment.__init__`:
    `RES_320X240`, RGB24, window invisible, sound off, HUD off, crosshair on,
    labels + depth buffers on, one seed per instance, and a 3-button action
    set restricted to this scenario's own `SCENARIO_ACTIONS`.
    """

    def __init__(self, scenario: str, seed: int, capture_frames: bool = False):
        if scenario not in SCENARIO_ACTIONS:
            raise ValueError(f"unknown scenario {scenario!r} -- expected one of {list(SCENARIO_ACTIONS)}")
        self.scenario = scenario
        self.seed = seed
        self.capture_frames = capture_frames
        self.action_names = SCENARIO_ACTIONS[scenario]

        self.game = vzd.DoomGame()
        cfg_path = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        self.game.load_config(cfg_path)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)

        available_buttons = [BUTTON_MAP[a] for a in self.action_names]
        self.game.set_available_buttons(available_buttons)
        self.game.set_available_game_variables([
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
            vzd.GameVariable.KILLCOUNT,
        ])
        self.game.set_seed(seed)
        self.game.init()

        # The trap this whole assertion exists to catch (see module
        # docstring): ViZDoom's own button order is whatever
        # set_available_buttons() was called with, but nothing stops a
        # future edit here from reordering SCENARIO_ACTIONS/BUTTON_MAP out of
        # sync with the one-hot action vectors built in step() below. Fail
        # loudly at construction time, not with a silently-wrong score.
        actual = list(self.game.get_available_buttons())
        expected = available_buttons
        if actual != expected:
            raise RuntimeError(
                f"get_available_buttons() returned {actual!r}, expected {expected!r} for "
                f"scenario {scenario!r} -- action-to-button mapping would be silently wrong."
            )

    def reset(self) -> DoomSnapshot:
        self.game.new_episode()
        snap = self.get_snapshot()
        assert snap is not None, "get_snapshot() returned None immediately after new_episode()"
        return snap

    def is_finished(self) -> bool:
        return self.game.is_episode_finished()

    def step(self, action: str, tics: int = 4) -> None:
        btn = BUTTON_MAP.get(action, BUTTON_MAP[self.action_names[0]])
        vec = [1 if b == btn else 0 for b in self.game.get_available_buttons()]
        self.game.make_action(vec, tics)

    def get_snapshot(self) -> Optional[DoomSnapshot]:
        s = self.game.get_state()
        if s is None:
            return None
        H, W = s.screen_buffer.shape[0], s.screen_buffer.shape[1]
        hp, ammo, kills = (int(v) for v in s.game_variables)

        things: list[Thing] = []
        for lab in s.labels:
            # Von's own quirk, reproduced exactly: some labels-buffer actor
            # names carry a "Vzd" suffix ViZDoom itself appends; strip it
            # before matching against MONSTERS/ITEMS, but also still treat a
            # "Vzd"-suffixed name as a monster even if the stripped form
            # isn't in the set (Von's own `or lab.object_name.endswith("Vzd")`
            # branch) -- reproduced verbatim rather than "simplified," since
            # the whole point here is byte-for-byte behavioral parity.
            raw = lab.object_name.removesuffix("Vzd")
            if lab.height <= 0 or lab.width <= 0:
                continue
            if raw in MONSTERS or lab.object_name.endswith("Vzd"):
                kind = "monster"
            elif raw in ITEMS:
                kind = "item"
            else:
                continue
            things.append(Thing(raw, kind, (lab.x + lab.width / 2) / W, lab.height / H))
        things.sort(key=lambda t: -t.size)

        d = s.depth_buffer
        if d is not None:
            dh, dw = d.shape
            band = d[dh // 3: 2 * dh // 3]
            dl = float(np.median(band[:, :dw // 5]))
            dc = float(np.median(band[:, dw * 2 // 5:dw * 3 // 5]))
            dr = float(np.median(band[:, -dw // 5:]))
        else:
            dl, dc, dr = 50.0, 50.0, 50.0

        return DoomSnapshot(
            health=hp,
            ammo=max(0, ammo),
            kills=kills,
            tic=self.game.get_episode_time(),
            things=things,
            depth_left=dl,
            depth_center=dc,
            depth_right=dr,
            frame=s.screen_buffer.copy() if self.capture_frames else None,
        )

    def close(self) -> None:
        self.game.close()


def survival_seconds(steps: int, tics: int) -> float:
    """Von's own metric definition: `survival_s = steps * tics / 35.0`."""
    return steps * tics / TICS_PER_SECOND


# --- Episode-outcome metric extraction, one per scenario -------------------
# Signature: (last_snapshot, steps, tics) -> float. Kept here (not in
# episodic.py) because what a scenario's "outcome" even means is a property
# of the scenario, exactly like SCENARIO_ACTIONS/SCENARIO_SEEDS above.

def _extract_kills(last_snapshot: DoomSnapshot, steps: int, tics: int) -> float:
    return float(last_snapshot.kills)


def _extract_survival(last_snapshot: DoomSnapshot, steps: int, tics: int) -> float:
    return survival_seconds(steps, tics)


SCENARIO_METRIC_NAME: dict[str, str] = {
    "defend_the_center": "kills",
    "health_gathering": "survival_s",
}

SCENARIO_METRIC_FN: dict[str, Callable[[DoomSnapshot, int, int], float]] = {
    "defend_the_center": _extract_kills,
    "health_gathering": _extract_survival,
}
