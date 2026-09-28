"""
Snake played by the real ekVachan model -- every move is one genuine
`predict_choice` call (choice of up/down/left/right from a text description
of the board), timed with CUDA sync. The displayed action + latency_ms on
every frame are that run's real numbers, not scripted.

Board dynamics (wall death, food growth, no 180-degree reversals) are plain
code -- the model only ever picks the direction. If the model picks a move
that kills the snake, the snake dies on screen: no safety net, no re-rolls.

Usage (on the machine with the checkpoints + GPU):
    uv run python3 demo/game_snake/snake_demo.py \
        --checkpoints-dir checkpoints-0.5b --episodes 3 --out-dir ~/ekvachan/demo_snake
Keeps the highest-scoring episode's frames + decisions.jsonl, prints the rest.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch
from PIL import Image, ImageDraw

try:
    from serve.inference import RoutingDecoderModel
except ImportError:  # pragma: no cover
    RoutingDecoderModel = None

GRID = 20
CELL = 24
HUD_H = 64
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
ARROW = {"up": "^", "down": "v", "left": "<", "right": ">"}


class Snake:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        cx, cy = GRID // 2, GRID // 2
        self.body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = "right"
        self.score = 0
        self.ticks = 0
        self.alive = True
        self.history: list[str] = []
        self.since_food = 0
        self._place_food()

    def _place_food(self):
        free = [(x, y) for x in range(GRID) for y in range(GRID) if (x, y) not in self.body]
        self.food = self.rng.choice(free) if free else None

    def danger(self, move: str) -> str:
        dx, dy = DIRS[move]
        nx, ny = self.body[0][0] + dx, self.body[0][1] + dy
        if not (0 <= nx < GRID and 0 <= ny < GRID):
            return "wall"
        if (nx, ny) in self.body[:-1]:  # tail cell frees up unless growing
            return "own body"
        return "free"

    def state_text(self) -> tuple[str, list[str]]:
        hx, hy = self.body[0]
        fx, fy = self.food
        options = [m for m in DIRS if m != OPPOSITE[self.direction]]
        danger = "; ".join(f"{m}: {self.danger(m)}" for m in options)
        recent = ", ".join(self.history[-6:]) if self.history else "none yet"
        state = (
            f"Snake on a {GRID}x{GRID} grid. Head at ({hx},{hy}), currently moving "
            f"{self.direction}. Food (red dot) at ({fx},{fy}). Adjacent danger -- {danger}. "
            f"Your last moves: {recent}. Steps since last food: {self.since_food}. "
            f"GOAL: eat as much food as possible in as few steps as possible -- score is food "
            f"eaten, and every step without eating is a wasted step. Take the shortest safe "
            f"route to the food; dying ends the game with your current score."
        )
        return state, options

    def step(self, move: str):
        self.direction = move
        self.history.append(move)
        self.since_food += 1
        dx, dy = DIRS[move]
        head = (self.body[0][0] + dx, self.body[0][1] + dy)
        self.ticks += 1
        if not (0 <= head[0] < GRID and 0 <= head[1] < GRID) or head in self.body[:-1]:
            self.alive = False
            return
        self.body.insert(0, head)
        if head == self.food:
            self.score += 1
            self.since_food = 0
            self._place_food()
            if self.food is None:
                self.alive = False  # board cleared: a win, stop here
        else:
            self.body.pop()


def render(snake: Snake, action: str, latency_ms: float, tick: int, model_label: str) -> Image.Image:
    img = Image.new("RGB", (GRID * CELL, GRID * CELL + HUD_H), (12, 14, 20))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, GRID * CELL, HUD_H], fill=(22, 27, 34))
    d.text((10, 8), f"TICK {tick}   SCORE {snake.score}   MOVE {ARROW.get(action, '?')} {action.upper()}",
           fill=(255, 255, 255))
    d.text((10, 34), f"{model_label} decision  {latency_ms:.1f} ms", fill=(120, 220, 130))
    for i, (x, y) in enumerate(snake.body):
        color = (80, 200, 120) if i == 0 else (40, 140, 90)
        d.rectangle([x * CELL, HUD_H + y * CELL, (x + 1) * CELL - 1, HUD_H + (y + 1) * CELL - 1], fill=color)
    fx, fy = snake.food
    d.ellipse([fx * CELL + 4, HUD_H + fy * CELL + 4, (fx + 1) * CELL - 4, HUD_H + (fy + 1) * CELL - 4],
              fill=(230, 70, 70))
    if not snake.alive:
        d.rectangle([0, HUD_H, GRID * CELL, GRID * CELL + HUD_H], outline=(230, 70, 70), width=6)
    return img


def play_episode(model, seed: int, max_ticks: int, frames_dir: Path, model_label: str):
    snake = Snake(seed)
    decisions = []
    tick = 0
    while snake.alive and tick < max_ticks:
        state, options = snake.state_text()
        t0 = time.perf_counter()
        out = model.predict_choice(state, options)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action = out["choice"]
        snake.step(action)
        tick += 1
        decisions.append({"tick": tick, "action": action, "latency_ms": round(latency_ms, 2),
                          "score": snake.score, "confidence": round(float(out.get("confidence", 0.0)), 4)})
        render(snake, action, latency_ms, tick, model_label).save(frames_dir / f"snake_{seed}_{tick:04d}.png")
    return snake.score, tick, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", default="checkpoints-0.5b")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--max-ticks", type=int, default=800)
    ap.add_argument("--model-label", default="ekVachan")
    ap.add_argument("--grid", type=int, default=20,
                    help="board is grid x grid cells; smaller boards mean shorter routes to food")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    global GRID
    GRID = args.grid

    if RoutingDecoderModel is None:  # pragma: no cover
        print("FATAL: serve.inference not importable", file=sys.stderr)
        sys.exit(1)
    model = RoutingDecoderModel(checkpoints_dir=Path(args.checkpoints_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or list(range(7, 7 + args.episodes))
    best = None
    for seed in seeds:
        frames_dir = out_dir / f"frames_{seed}"
        frames_dir.mkdir(exist_ok=True)
        score, ticks, decisions = play_episode(model, seed, args.max_ticks, frames_dir, args.model_label)
        print(f"seed {seed}: score={score} ticks={ticks}", flush=True)
        if best is None or score > best[0]:
            best = (score, ticks, seed, decisions, frames_dir)
    score, ticks, seed, decisions, frames_dir = best
    (out_dir / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n")
    print(f"BEST seed={seed} score={score} ticks={ticks} frames={frames_dir}")


if __name__ == "__main__":
    main()
