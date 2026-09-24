# Attribution

`benchmarks/screenspot_v2/vendored/heldout.jsonl` is derived from
[`OS-Copilot/ScreenSpot-v2`](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2)
(HF revision `5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c`), **not fetched from
HuggingFace directly** but exported from the sibling repo
[`better-jev-bench`](https://github.com/asp616848/better-jev-bench), which
already builds, license-verifies, and hash-commits this dataset
(`better_jev_bench/datasets/screenspot_v2.py`,
`datasets/screenspot_v2/manifest.toml`).

- **License: Apache-2.0.** Verified two ways, independently: (1) the
  HuggingFace Hub API reports `cardData['license'] == "apache-2.0"` for
  `OS-Copilot/ScreenSpot-v2`, queried directly on 2026-09-24, not copied from
  better-jev-bench's own claim; (2) that independently-queried result matches
  better-jev-bench's own verification in its manifest.
- **No separate `LICENSE` file exists in the upstream HF repo** -- unlike
  JevBench and jabr-v2, whose upstream repos ship one. The upstream repo's
  file listing (checked via the HF Hub API, 2026-09-24) is exactly
  `.gitattributes`, `README.md`, three `screenspot_*_v2.json` files, and
  `screenspotv2_image.zip` -- no `LICENSE`. `vendored/LICENSE` here is
  therefore the **standard Apache License 2.0 text**, included because the
  `license:apache-2.0` tag legally refers to it, not a byte-for-byte copy of
  an upstream file (there is none to copy). Flagged here explicitly rather
  than silently presented as "the upstream LICENSE file," matching this
  project's own discipline (see benchmarks/jevbench/NOTICE.md's stale-
  THIRD-PARTY.md flag for the precedent).
- **Copyright**: per the dataset's own provenance, ScreenSpot-v2 is a
  re-annotation effort by the OS-Copilot project correcting ~11.32% of the
  original ScreenSpot's mislabelled grounding targets.
- **Obligation: attribution** (per better-jev-bench's own
  `datasets/screenspot_v2/manifest.toml`, `license.obligations = ["attribution"]`,
  and independently confirmed here). This NOTICE file is that attribution.

**Vendored file**: `heldout.jsonl`, sha256
`62631dc9a5ff1fc13a32adf696ccf911373919b6730fc6d8eb3391b6c0ae8826`, 858
records, produced 2026-09-24 by running (for real, on the server both repos
live on, not assumed from reading the code):

```sh
cd better-jev-bench
uv run bjb export --slice heldout --datasets screenspot_v2 --out <dir>
# -> <dir>/heldout.jsonl, 858 lines
```

See `vendored/manifest.json` for the full provenance record (export command,
license verification, the real pool-size histogram from the N-way
reframing) and `../loader.py`'s module docstring for why this is 858 items,
not the 898 the upstream manifest's headline `item_count` implies (40 of
those 898 are `bjb export --slice public` rows, structurally unreachable
because this dataset is `eval_only` in better-jev-bench's own schema).

**Images are not vendored here** -- see `../README.md`'s "Vendoring: images
vs. metadata" section for the full reasoning. They are resolved fresh, every
load, from better-jev-bench's own content-addressed image cache and
re-hashed before use.

Nothing else from either upstream repo (OS-Copilot's own tooling, or
better-jev-bench's harness/CI code) is vendored here -- only the one exported
JSONL file this benchmark needs.
