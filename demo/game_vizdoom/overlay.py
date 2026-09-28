"""
Burns per-tick decision overlays onto ViZDoom dagger frames: tick, the
model's real action, and that tick's real measured latency_ms, read from
the run's own .raw evidence file (matched by tick). Numbers are never
edited, only drawn.

Usage:
    python3 demo/game_vizdoom/overlay.py \
        --frames-dir ~/ekvachan/demo_viz/<session>/health_gathering_von/frames \
        --raw results/vizdoom-health_gathering-von-routing_decoder-<ts>.raw \
        --seed 41006394 --episode 0 --out-dir ~/ekvachan/demo_viz/overlay
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

BANNER_H = 44


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--model-label", default="ekVachan-flash")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    recs = {}
    for line in Path(args.raw).read_text().splitlines():
        r = json.loads(line)
        if r.get("seed") == args.seed and r.get("episode_index", r.get("episode")) == args.episode:
            recs[r["tick"]] = r
    print(f"loaded {len(recs)} tick records from raw")

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob(f"*_{args.seed}_{args.episode}_*.png"))
    print(f"overlaying {len(frames)} frames")
    n = 0
    for fp in frames:
        tick = int(fp.stem.split("_")[-1])
        rec = recs.get(tick, {})
        action = rec.get("model_action") or rec.get("effective_action") or "?"
        latency = rec.get("latency_ms")
        ms_txt = f"{latency:.1f} ms" if latency is not None else "n/a"
        img = Image.open(fp).convert("RGB")
        canvas = Image.new("RGB", (img.width, img.height + BANNER_H), (16, 18, 24))
        canvas.paste(img, (0, BANNER_H))
        d = ImageDraw.Draw(canvas)
        d.text((10, 6), f"TICK {tick}   ACTION {action}", fill=(255, 255, 255))
        d.text((10, 24), f"{args.model_label} decision  {ms_txt}", fill=(120, 220, 130))
        canvas.save(out_dir / fp.name)
        n += 1
    print(f"wrote {n} overlay frames to {out_dir}")


if __name__ == "__main__":
    main()
