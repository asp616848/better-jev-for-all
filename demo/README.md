# ekVachan demo videos

Three short demo clips (mp4 + gif) for the public launch, each in a different
domain, per the request that spawned this branch. **Written without any
execution/GPU/browser access** (this session can't run Playwright, ffmpeg,
ViZDoom, or hit a live model server) -- everything here is real, runnable
code, meant for the local agent (which has all of that) to actually execute,
review the output of, and re-record if something looks wrong. Same
"written blind, run for real, verify before trusting" pattern as the rest of
this project (see `PRD.md`'s "written without GPU access" sections) --
nothing here should be treated as done until someone has actually watched
the output video.

**This branch is intentionally not meant to merge into `main`.** These are
marketing assets (recording scripts + a demo HTML page + output videos), not
product code -- keeping them off `main` keeps the production repo's history
free of megabyte-sized mp4/gif files and page-specific demo copy. If any of
`demo/web_demo/record.py`'s reusable pieces turn out to be broadly useful,
promote them deliberately later; don't merge this branch wholesale.

## What each demo is, and why it's a legitimate showcase (not vague/hand-wavy)

All three call the **real, live `/v1/systemone` endpoint** (or, for the game
demo, the real trained policy through the actual benchmark harness) -- no
staged/fake numbers, no scripted "AI" theater. The `latency_ms` and
`probabilities` shown in every clip are real numbers from a real forward
pass on that run.

1. **Health-domain triage** (`demo/web_demo`, `--scenario health`) -- routing
   a written patient intake complaint to `Emergency` / `Urgent Care` /
   `Routine Appointment`. This is a **triage-routing** decision (which queue
   a case goes to), not a diagnosis -- the demo page says so on screen. This
   distinction matters: real intake/triage-routing tools already do exactly
   this kind of choice classification in production, whereas a public demo
   that looked like it was "diagnosing" would be the kind of vague/
   overclaiming demo the task asked to avoid.
2. **ViZDoom gameplay** (`demo/game_vizdoom`) -- the real trained checkpoint,
   through the same benchmark harness PRD.md 13a.13/13a.21 already validated
   (`defend_the_center`, `von` rubric, a published seed), making real
   in-game decisions at real inference speed. Reuses the production
   `--backend routing-decoder` path, not a special-cased demo model.
3. **Finance-domain risk triage** (`demo/web_demo`, `--scenario finance`) --
   routing a described transaction to `Approve` / `Hold for Review` /
   `Decline`. Framed as **operational risk triage** (the same shape of
   decision real fraud/risk-ops teams make), deliberately **not** framed as
   trading or investment advice -- that would be the "questionable" framing
   the task explicitly said to avoid.

## Prerequisites (on the machine that actually runs this)

```bash
pip install playwright imageio imageio-ffmpeg pillow requests
playwright install chromium
# ffmpeg must also be on PATH (imageio-ffmpeg vendors its own binary, so a
# system ffmpeg isn't strictly required, but having one is a good sanity check)
```

The web demos need a **live ekVachan server** to call:
```bash
EKVACHAN_TEXT_ADAPTER=vision uv run uvicorn serve.server:app --port 8000
```
The ViZDoom demo needs the real checkpoints available locally (same
`--checkpoints-dir` the rest of the benchmark suite uses) and ViZDoom
installed (already a project dependency, `pyproject.toml`).

## Running everything

```bash
# 1. Health + finance (HTML animation, real API calls, screen-recorded)
python3 -u demo/web_demo/record.py --scenario health  --server http://localhost:8000 --out demo/output/health
python3 -u demo/web_demo/record.py --scenario finance --server http://localhost:8000 --out demo/output/finance

# 2. ViZDoom gameplay (real trained policy, real game engine frames)
bash demo/game_vizdoom/record.sh
```

Each produces `<out>.mp4` and `<out>.gif` (gif is a lighter, autoplay-friendly
cut of the same clip -- see each script's own `--gif-fps`/`--gif-max-seconds`
for how it's trimmed down, not just re-encoded 1:1).

## Before using the output anywhere public

Watch all three end to end. Things to actually check, not assume:
- The displayed `choice`/`probabilities`/`latency_ms` in every clip are
  real numbers from that run, not the scenario JSON's placeholder text
  bleeding through a failed API call (`record.py` fails loudly rather than
  falling back to fake data on a request error -- but confirm the output
  video doesn't show a stuck "thinking" state or an error toast).
- The ViZDoom clip's actions look like genuine gameplay, not the agent
  standing still because the policy silently fell back to a no-op action.
- No hardware/GPU model name is visible on screen anywhere (matches the
  redaction already applied to `main` -- PRD.md 13a.35).
