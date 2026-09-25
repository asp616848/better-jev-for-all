"""
Records a short mp4 + gif of demo/web_demo/index.html driven through 2-3
real scenarios, calling a LIVE ekVachan server for every decision shown --
no staged/fake numbers (see demo/README.md's "before using the output
anywhere public" checklist).

Written without Playwright/a live server/execution access (this session
can't run any of that) -- meant for the local agent to actually run. Fails
loudly (an uncaught exception, no silent fallback) if the server is
unreachable or a request errors, on purpose: a demo video built from a
failed call is worse than no video.

Usage:
    uv run python3 -u demo/web_demo/record.py --scenario health \
        --server http://localhost:8000 --out demo/output/health
    uv run python3 -u demo/web_demo/record.py --scenario finance \
        --server http://localhost:8000 --out demo/output/finance
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

HTML_PATH = Path(__file__).resolve().parent / "index.html"
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def check_server(server: str) -> None:
    try:
        resp = requests.get(f"{server}/health", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(
            f"FATAL: {server}/health did not respond OK ({e!r}) -- start the real server first, "
            f"e.g. EKVACHAN_TEXT_ADAPTER=vision uv run uvicorn serve.server:app --port 8000. "
            f"Refusing to record a demo against a server that isn't actually up.",
            file=sys.stderr,
        )
        sys.exit(1)


def call_model(server: str, state: str, options: list[str]) -> dict:
    resp = requests.post(
        f"{server}/v1/systemone",
        json={"state": state, "questions": {"q": {"type": "choice", "options": options}}},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    result = body["results"]["q"]
    result["latency_ms"] = body["usage"]["latency_ms"]
    return result


def _ffmpeg_binary() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def record(scenario_name: str, server: str, out_prefix: Path, pause_s: float) -> None:
    check_server(server)
    scenario = json.loads((SCENARIOS_DIR / f"{scenario_name}.json").read_text())

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    video_dir = out_prefix.parent / f"_{scenario_name}_video_tmp"
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": 900, "height": 620},
            record_video_dir=str(video_dir),
            record_video_size={"width": 900, "height": 620},
        )
        page = context.new_page()
        page.goto(HTML_PATH.as_uri())
        page.wait_for_timeout(300)
        page.evaluate(
            "([label, disclaimer]) => demoSetDomain(label, disclaimer)",
            [scenario["domain_label"], scenario["disclaimer"]],
        )

        for q in scenario["questions"]:
            page.evaluate("demoReset()")
            page.wait_for_timeout(400)
            # demoTypeState returns a Promise; page.evaluate awaits it, so
            # this line blocks until the typing animation actually finishes.
            page.evaluate("(t) => demoTypeState(t)", q["state"])
            page.wait_for_timeout(300)
            page.evaluate("(opts) => demoShowOptions(opts)", q["options"])
            page.evaluate("demoShowThinking()")

            print(f"[{scenario_name}] calling live model: {q['state'][:70]}...")
            t0 = time.time()
            result = call_model(server, q["state"], q["options"])
            wall_ms = (time.time() - t0) * 1000
            print(
                f"[{scenario_name}] real response in {wall_ms:.0f}ms wall "
                f"(server-reported latency_ms={result['latency_ms']:.1f}): choice={result['choice']!r}"
            )

            # Video-pacing floor only -- keeps "thinking" on screen long
            # enough to read even when the real call was very fast. Does
            # NOT alter the latency_ms value shown afterwards, which is
            # always the server's own real measurement.
            min_thinking_ms = 900
            if wall_ms < min_thinking_ms:
                page.wait_for_timeout(min_thinking_ms - wall_ms)

            page.evaluate("(r) => demoShowResult(r)", result)
            page.wait_for_timeout(int(pause_s * 1000))

        context.close()
        browser.close()

    webm_files = list(video_dir.glob("*.webm"))
    if not webm_files:
        print(f"FATAL: no .webm recorded in {video_dir}", file=sys.stderr)
        sys.exit(1)
    webm_path = webm_files[0]

    mp4_path = out_prefix.with_suffix(".mp4")
    gif_path = out_prefix.with_suffix(".gif")
    ffmpeg = _ffmpeg_binary()

    subprocess.run(
        [ffmpeg, "-y", "-i", str(webm_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)],
        check=True,
    )
    subprocess.run(
        [ffmpeg, "-y", "-i", str(mp4_path), "-vf", "fps=12,scale=640:-1:flags=lanczos", str(gif_path)],
        check=True,
    )
    shutil.rmtree(video_dir)
    print(f"Wrote {mp4_path} and {gif_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, choices=["health", "finance"])
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--out", required=True, help="output path prefix, e.g. demo/output/health -> .mp4/.gif")
    parser.add_argument("--pause-seconds", type=float, default=2.5, help="how long each result stays on screen")
    args = parser.parse_args()
    record(args.scenario, args.server, Path(args.out), args.pause_seconds)


if __name__ == "__main__":
    main()
