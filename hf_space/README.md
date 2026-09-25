---
title: ekVachan
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# ekVachan demo

Try the model: give it a `state` and a comma-separated set of `options`, get
back a calibrated probability distribution in one forward pass.

This Space runs `serve.inference.RoutingDecoderModel` directly -- the real
production class, not a reimplementation. Its source (`serve/`,
`benchmarks/common/`, `training/decoder_lora_lib.py`, `eval/`) is vendored
into this Space repo alongside `app.py` by
[`scripts/publish_to_huggingface.py`](https://github.com/asp616848/better-jev-for-all/blob/main/scripts/publish_to_huggingface.py)
in the main repo -- no Docker, no git clone at request time (see that
script's docstring for exactly what it copies and why).

**Deployment notes** (full detail in the main repo's `PRD.md`, "HF Space
deployment" section):
- Cost-minimized on purpose: `@spaces.GPU` wraps only the actual forward
  pass, the model is loaded once at cold start (not per request, not
  per-GPU-attachment), no `torch.compile`, no CUDA graphs, no background
  process of any kind.
- If this Space is running on **dedicated hardware** (e.g. T4-small) rather
  than free ZeroGPU, remember: billing is by wall-clock "Running" time, not
  by request volume. **Pause the Space from its Settings tab whenever it's
  not actively being demoed** -- that's the only way to fully stop billing
  on dedicated hardware. Configure the shortest available auto-sleep-after-
  inactivity setting there too, so an idle Space doesn't run (and bill)
  indefinitely if someone forgets to pause it by hand.

**On latency**: whatever this Space reports includes its own allocation/
queueing overhead on top of the model's forward pass, and (if running on a
smaller/older card than our own dedicated server) may simply be slower per
forward pass too. For the real, verified accuracy and latency figures
(JevBench, jabr-v2, ViZDoom, our own corpus, all against our own dedicated
server), see the [README](https://github.com/asp616848/better-jev-for-all#readme)
and `PRD.md` (sections 13a.29-13a.34) in the main repo.

**Updating this Space**: re-run `scripts/publish_to_huggingface.py` from the
main repo to push a new commit here -- this Space intentionally does not
auto-track `main`, so a bad push to the model repo can never silently take
the public demo down or serve stale/wrong numbers.
