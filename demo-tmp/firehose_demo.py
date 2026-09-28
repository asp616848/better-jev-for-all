"""
Moderation firehose -- a stream of synthetic chat messages (labeled on
screen as SYNTHETIC) where every message gets two genuine model calls:
`predict_noul` (should this be blocked?) and `predict_choice` over
{ok, spam, toxic, phishing}. Displayed verdicts and latency_ms are that
run's real numbers, timed with CUDA sync.

Usage (machine with checkpoints + GPU):
    uv run python3 demo/moderation_firehose/firehose_demo.py \
        --checkpoints-dir checkpoints-0.5b --out-dir ~/ekvachan/demo_mod
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

W, H = 640, 420
CATEGORIES = ["ok", "spam", "toxic", "phishing"]

MESSAGES = [
    ("anyone up for a match later tonight?", "ok"),
    ("CLICK HERE to claim your FREE prize money now!!!", "spam"),
    ("gg everyone, that last round was close", "ok"),
    ("you are all terrible, uninstall the game", "toxic"),
    ("does anyone know the raid time tomorrow?", "ok"),
    ("double your coins: send 100 to this address first", "phishing"),
    ("looking for one more player for ranked", "ok"),
    ("BUY NOW limited offer, act fast, 90% off!!!", "spam"),
    ("nice clutch play in the final minute", "ok"),
    ("verify your account now or it gets deleted today", "phishing"),
    ("can someone explain the new map rotation?", "ok"),
    ("shut up, nobody wants you on this team", "toxic"),
    ("stream starts in ten minutes, see you there", "ok"),
    ("FREE skins giveaway, just log in with this link", "phishing"),
    ("what sensitivity do you all play on?", "ok"),
    ("EARN $500/day from home, no experience needed!", "spam"),
    ("that strategy worked really well, thanks all", "ok"),
    ("your package delivery failed, re-enter card details", "phishing"),
    ("lol that whiff was painful to watch", "ok"),
    ("kys, you're throwing every single game", "toxic"),
    ("tournament bracket is posted in announcements", "ok"),
    ("crypto doubling event, send ETH to participate", "phishing"),
    ("any tips for aiming on controller?", "ok"),
    ("CONGRATS WINNER claim gift cards here", "spam"),
    ("welcome to the server, new folks!", "ok"),
    ("password reset required: confirm credentials now", "phishing"),
    ("that comeback was actually insane", "ok"),
    ("everyone on red team is garbage, delete the game", "toxic"),
    ("patch notes are out, check the forum", "ok"),
    ("limited-time deal on coins, 2x bonus today only", "spam"),
    ("who wants to squad up for the evening?", "ok"),
    ("security alert: unusual login, verify identity here", "phishing"),
    ("great comms that round, well played", "ok"),
    ("you play like a bot, embarrassing", "toxic"),
    ("new emotes just dropped in the shop", "ok"),
    ("make money fast with this simple trick, banks hate it", "spam"),
    ("remember to hydrate during long sessions", "ok"),
    ("final warning: account suspension, click to appeal", "phishing"),
    ("that sniper shot deserved a replay", "ok"),
    ("trash team, zero coordination, hope you lose", "toxic"),
]


def render(history, blocked, avg_ms):
    img = Image.new("RGB", (W, H), (10, 12, 18))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 60], fill=(20, 24, 32))
    d.text((10, 6), "LIVE CHAT MODERATION  —  synthetic messages", fill=(255, 200, 90))
    d.text((10, 32), f"reviewed={len(history)}  blocked={blocked}  avg={avg_ms:.1f} ms/decision",
           fill=(255, 255, 255))
    y = 70
    for text, verdict, ms in history[-9:]:
        bad = verdict != "ok"
        d.rectangle([8, y, W - 8, y + 32], fill=(60, 20, 20) if bad else (20, 34, 26))
        d.text((14, y + 3), text[:52], fill=(255, 255, 255))
        d.text((14, y + 17), f"{verdict.upper()}  {ms:.1f} ms", fill=(255, 120, 120) if bad else (120, 220, 130))
        y += 38
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", default="checkpoints-0.5b")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if RoutingDecoderModel is None:  # pragma: no cover
        print("FATAL: serve.inference not importable", file=sys.stderr)
        sys.exit(1)
    model = RoutingDecoderModel(checkpoints_dir=Path(args.checkpoints_dir))
    out_dir = Path(args.out_dir)
    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    history, decisions, blocked = [], [], 0
    total_ms = 0.0
    for i, (text, _) in enumerate(MESSAGES):
        state = f"Chat message: {text!r}. Should this message be blocked by moderation?"
        t0 = time.perf_counter()
        p_block = model.predict_noul(state)
        cat = model.predict_choice(f"Chat message: {text!r}. Classify it.", CATEGORIES)["choice"]
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        verdict = cat if p_block >= 0.5 else "ok"
        if verdict != "ok":
            blocked += 1
        total_ms += ms
        history.append((text, verdict, ms))
        decisions.append({"i": i, "text": text, "p_block": round(p_block, 4),
                          "category": cat, "verdict": verdict, "latency_ms": round(ms, 2)})
        render(history, blocked, total_ms / len(history)).save(frames / f"mod_{i:04d}.png")
    (out_dir / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n")
    print(f"messages={len(MESSAGES)} blocked={blocked} avg_ms={total_ms / len(MESSAGES):.1f}")


if __name__ == "__main__":
    main()
