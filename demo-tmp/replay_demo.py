"""
Finance backtest replay -- a simulated intraday market (geometric random
walk with regimes, stated on screen as SIMULATED) where every round is one
genuine `predict_choice` call: given recent returns, volatility, position
and P&L, the model picks LONG / FLAT / SHORT. Plain code executes the
trade, marks equity to market, and renders the chart. Displayed decisions
and latency_ms are that run's real numbers.

Framed as a backtest replay, never as trading advice.

Usage (machine with checkpoints + GPU):
    uv run python3 demo/finance_trading/replay_demo.py \
        --checkpoints-dir checkpoints-0.5b --rounds 120 --out-dir ~/ekvachan/demo_fin
"""

import argparse
import json
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

W, H = 640, 400
CHART_TOP, CHART_H = 90, 180
N_LOOKBACK = 12


def simulate(seed: int, n: int):
    import random
    rng = random.Random(seed)
    prices, rets = [100.0], []
    drift, vol = 0.0008, 0.004
    for i in range(n):
        if i % 30 == 0:  # regime shifts, announced in the state text
            drift = rng.choice([-0.0012, -0.0004, 0.0004, 0.0012])
            vol = rng.choice([0.003, 0.005, 0.008])
        r = drift + rng.gauss(0, vol)
        rets.append(r)
        prices.append(prices[-1] * (1 + r))
    return prices, rets


def state_text(prices, rets, pos, pnl, i):
    window = rets[max(0, i - N_LOOKBACK):i]
    recent = ", ".join(f"{r * 100:+.2f}%" for r in window)
    import statistics
    vol = statistics.pstdev(window) * 100 if len(window) > 1 else 0.0
    trend = sum(window) * 100
    return (
        f"SIMULATED intraday market replay. Last {len(window)} period returns: {recent}. "
        f"Net trend {trend:+.2f}%, volatility {vol:.2f}%. Current position: {pos}. "
        f"Open P&L: {pnl:+.2f} points. Decide the position for the next period."
    ), ["long", "flat", "short"]


def render(prices, equity, i, action, latency_ms, pos, pnl):
    img = Image.new("RGB", (W, H), (10, 12, 18))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 56], fill=(20, 24, 32))
    d.text((10, 6), "SIMULATED MARKET REPLAY  —  not trading advice", fill=(255, 200, 90))
    d.text((10, 30), f"ROUND {i}   {action.upper()}   {latency_ms:.1f} ms   pos={pos}  P&L={pnl:+.2f}",
           fill=(255, 255, 255))
    lo, hi = min(prices[:i + 1]), max(prices[:i + 1])
    span = max(hi - lo, 1e-9)
    px = lambda k: 20 + k * (W - 40) / max(len(prices) - 1, 1)
    py = lambda p: CHART_TOP + CHART_H - (p - lo) / span * CHART_H
    d.text((10, CHART_TOP - 16), "price", fill=(120, 140, 160))
    for k in range(1, i + 1):
        d.line([px(k - 1), py(prices[k - 1]), px(k), py(prices[k])], fill=(90, 160, 255), width=2)
    elo, ehi = min(equity[:i + 1]), max(equity[:i + 1])
    espan = max(ehi - elo, 1e-9)
    ey = lambda e: 300 + 70 - (e - elo) / espan * 70
    d.text((10, 284), "equity", fill=(120, 140, 160))
    ecol = (110, 220, 130) if equity[i] >= 0 else (230, 90, 90)
    for k in range(1, i + 1):
        d.line([px(k - 1), ey(equity[k - 1]), px(k), ey(equity[k])], fill=ecol, width=2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", default="checkpoints-0.5b")
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if RoutingDecoderModel is None:  # pragma: no cover
        print("FATAL: serve.inference not importable", file=sys.stderr)
        sys.exit(1)
    model = RoutingDecoderModel(checkpoints_dir=Path(args.checkpoints_dir))
    out_dir = Path(args.out_dir)
    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    prices, rets = simulate(args.seed, args.rounds)
    pos, pnl, equity = "flat", 0.0, [0.0]
    decisions = []
    for i in range(1, args.rounds + 1):
        state, options = state_text(prices, rets, pos, pnl, i)
        t0 = time.perf_counter()
        out = model.predict_choice(state, options)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action = out["choice"]
        pos = action
        r = rets[i - 1]
        pnl += (100 * r if pos == "long" else (-100 * r if pos == "short" else 0.0))
        equity.append(pnl)
        decisions.append({"round": i, "action": action, "latency_ms": round(latency_ms, 2),
                          "pos": pos, "pnl": round(pnl, 2),
                          "confidence": round(float(out.get("confidence", 0.0)), 4)})
        render(prices, equity, i, action, latency_ms, pos, pnl).save(frames / f"fin_{i:04d}.png")
    (out_dir / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n")
    print(f"rounds={args.rounds} final_pnl={pnl:+.2f} frames={frames}")


if __name__ == "__main__":
    main()
