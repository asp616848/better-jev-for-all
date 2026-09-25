"""
Stitches a sequence of per-tick PNG frames (as written by
benchmarks/common/episodic.py's DaggerSink -- reused as-is, not
reimplemented, since it already saves exactly one real-gameplay PNG per
tick when a ViZDoom run is given --dagger-out) into an mp4 and a gif.

Frame filenames follow DaggerSink's own convention:
    <scenario>_<rubric>_<seed>_<episode_index>_<tick:04d>.png
so sorting lexicographically on that glob is already the correct tick
order (4-digit zero-padded tick).

Written without ffmpeg/execution access -- meant to run on the machine
that actually recorded the frames.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _ffmpeg_binary() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--frames-glob", required=True,
        help="glob for one episode's frames, e.g. "
             "'training/dagger_data/vizdoom/<run_id>/frames/defend_the_center_von_<seed>_0_*.png'",
    )
    parser.add_argument("--out", required=True, help="output path prefix -> <out>.mp4 / <out>.gif")
    parser.add_argument("--fps", type=int, default=12, help="ViZDoom's own tics=4 frameskip already "
                         "compresses real game time -- 12fps here plays each captured tick as one "
                         "video frame at a watchable pace, not real wall-clock speed.")
    args = parser.parse_args()

    frames = sorted(Path(".").glob(args.frames_glob))
    if not frames:
        print(f"FATAL: no frames matched {args.frames_glob!r} -- run "
              f"benchmarks/vizdoom/run.py with --dagger-out first (see demo/game_vizdoom/record.sh).",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(frames)} frames: {frames[0].name} .. {frames[-1].name}")

    # ffmpeg needs a single glob pattern it resolves itself (glob demuxer),
    # not a Python-expanded file list -- pass the same pattern straight
    # through rather than writing a concat-list file, since these frame
    # filenames are already lexicographically tick-ordered.
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    mp4_path = out_prefix.with_suffix(".mp4")
    gif_path = out_prefix.with_suffix(".gif")
    ffmpeg = _ffmpeg_binary()

    subprocess.run(
        [
            ffmpeg, "-y",
            "-framerate", str(args.fps),
            "-pattern_type", "glob", "-i", args.frames_glob,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(mp4_path),
        ],
        check=True,
    )
    subprocess.run(
        [ffmpeg, "-y", "-i", str(mp4_path), "-vf", f"fps={args.fps},scale=480:-1:flags=lanczos", str(gif_path)],
        check=True,
    )
    print(f"Wrote {mp4_path} and {gif_path}")


if __name__ == "__main__":
    main()
