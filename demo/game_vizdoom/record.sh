#!/usr/bin/env bash
# Records one real ViZDoom episode -- the actual trained checkpoint, driving
# actual gameplay through the actual production `routing-decoder` benchmark
# backend (PRD.md 13a.13/13a.21/13a.32/13a.33's own validated path, not a
# special-cased demo model) -- and stitches the captured frames into an
# mp4 + gif.
#
# Written without ViZDoom/GPU/execution access -- meant to run on the
# machine that actually has the trained checkpoints and a working ViZDoom
# install (both already project dependencies, see pyproject.toml).
#
# Usage: bash demo/game_vizdoom/record.sh [checkpoints_dir]
set -euo pipefail

CHECKPOINTS_DIR="${1:-checkpoints}"
SCENARIO="defend_the_center"
RUBRIC="von"
# First of Von's own published defend_the_center seeds (env.py's
# DEFEND_SEEDS) -- one real episode is enough for a demo clip; not cherry-
# picked for a good outcome, just the first published seed in the list.
SEED=42106076
DAGGER_ROOT="demo/output/_vizdoom_dagger"
OUT_PREFIX="demo/output/vizdoom_defend_the_center"

rm -rf "$DAGGER_ROOT"
mkdir -p "$DAGGER_ROOT"

echo "Running one real ViZDoom episode (scenario=$SCENARIO rubric=$RUBRIC seed=$SEED) with the real trained model..."
uv run python3 -u -m benchmarks.vizdoom.run \
  --scenario "$SCENARIO" \
  --rubric "$RUBRIC" \
  --policy model \
  --backend routing-decoder \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --text-adapter vision \
  --seeds "$SEED" \
  --episodes-per-seed 1 \
  --dagger-out "$DAGGER_ROOT"

# run.py timestamps its own dagger_root/<session_ts>/ subdirectory -- find
# the one this invocation just created (there should be exactly one).
SESSION_DIR="$(ls -td "$DAGGER_ROOT"/*/ | head -n1)"
FRAMES_GLOB="${SESSION_DIR}${SCENARIO}_${RUBRIC}/frames/${SCENARIO}_${RUBRIC}_${SEED}_0_*.png"

echo "Stitching frames matching: $FRAMES_GLOB"
uv run python3 -u demo/game_vizdoom/frames_to_video.py \
  --frames-glob "$FRAMES_GLOB" \
  --out "$OUT_PREFIX" \
  --fps 12

echo "Done: ${OUT_PREFIX}.mp4 and ${OUT_PREFIX}.gif"
