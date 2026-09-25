---
title: ekVachan
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# ekVachan demo

Try the model: give it a `state` and a comma-separated set of `options`, get
back a calibrated probability distribution in one forward pass.

This Space runs `serve.inference.RoutingDecoderModel` directly, cloned from
[asp616848/better-jev-for-all](https://github.com/asp616848/better-jev-for-all)
at a pinned commit (see `Dockerfile`'s `EKVACHAN_COMMIT` build arg) -- it is
the real production class, not a reimplementation.

**On latency**: this Space runs on a shared free ZeroGPU allocation. The
latency you see here includes ZeroGPU's own queueing/cold-start overhead on
top of the model's forward pass, and will be higher and noisier than our own
benchmarked numbers. For the real, verified accuracy and latency figures
(JevBench, jabr-v2, ViZDoom, our own corpus, all against a dedicated warm
server), see the [README](https://github.com/asp616848/better-jev-for-all#readme)
and `PRD.md` (sections 13a.29-13a.34) in the main repo.

**Updating the pinned commit**: this Space intentionally does not track
`main` automatically -- bump `EKVACHAN_COMMIT` in `Dockerfile` and rebuild
when you want the Space to pick up a new commit, so a bad push to the model
repo can never silently take the public demo down or serve stale/wrong
numbers.
