# Attribution

`benchmarks/jevbench/vendored/original.jsonl`, `easy.jsonl`, `hard.jsonl`, and
`manifest.json` are vendored, byte-for-byte, from
[`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench)
(the "JevBench" benchmark; "Benchmark Heaven"'s own benchmark, not affiliated
with or endorsed by TypeSafe AI), fetched 2026-09-22 via the GitHub Contents
API at that repo's `main` branch tip as of that date.

- License: MIT. `benchmarks/jevbench/vendored/LICENSE` is the repo's own LICENSE
  file, copied verbatim, as MIT requires. Verified two ways: (1) GitHub's API
  reports `license.spdx_id: "MIT"` for the repo; (2) every vendored record's
  own `provenance.license` field independently says `"MIT"` (checked
  programmatically across all 231 records — see README.md).
- Copyright: per the vendored LICENSE, "Copyright (c) 2026 Florian
  Standhartinger and contributors."
- Repo README's `THIRD-PARTY.md` states "MIT covers this harness and the 72
  original public decisions in `datasets/public/original.jsonl`" without
  separately naming `easy.jsonl`/`hard.jsonl` — that sentence predates the
  easy/hard tiers (added in v1.1/v1.2) and looks stale, but every individual
  record in all three files, including easy/hard, carries its own
  `"license": "MIT"` and `"source": "JevBench ... original authored
  scenario"` provenance field, which is the more specific and more recently
  written claim. Treated here as authoritative; flagged in case a future
  upstream revision clarifies otherwise.

Vendored file hashes (sha256, computed 2026-09-22 — these match the hashes
JevBench's own `datasets/manifest.json` records for `easy`, `hard`, and
`original`, confirming byte-identical vendoring):

```
231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b  easy.jsonl
89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb  hard.jsonl
5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180  original.jsonl
```

Nothing else from the upstream repo (its harness code, adapters, other
models' results) is vendored here — only the three public dataset files and
their manifest, which is all this harness needs.
