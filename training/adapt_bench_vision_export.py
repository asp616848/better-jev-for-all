"""
Thin adapter bridging better-jev-bench's real `bjb export` image-record format
to `build_vision_slice.py`'s `--vision-source` contract -- exactly the
"thin adapter, not a rewrite" that script's own docstring predicted would be
needed once the bench repo's image support landed (it landed 2026-09-24,
after that docstring was written; this file is the predicted follow-up).

**The real format mismatch, found and bridged here**: `bjb export` (as of
2026-09-24, `better-jev-bench` commit history same date) emits `images` as a
list of dicts -- `{"sha256": "...", "path": "<absolute local path>",
"media_type": "...", "width": int, "height": int}` -- verified directly by
running a real export (`bjb:atari_head/action`, `bjb export --slice public
--datasets atari_head --max-per-task 3`), not assumed from the plan.
`build_vision_slice.py`'s `load_vision_source_records()` expects a flatter
shape: `"images": ["<path string>"]` plus a separate `"images_sha256":
["<hash>"]` list. Both repos are on the same server (`abhijeet-labgpu`), so
the bench export's `path` field is already a real, directly-readable local
file -- no copy or re-download needed, just a field reshape.

v1 scope, matching `build_vision_slice.py`'s own stated scope note: exactly
one image per record. Every bench export record so far (atari_head,
os_atlas) already satisfies this.

Run:
    uv run python3 -m training.adapt_bench_vision_export \\
        --in-jsonl /tmp/atari_export/public.jsonl --split train \\
        --in-jsonl /tmp/os_atlas_export/public.jsonl --split train \\
        --out /tmp/vision_source.jsonl
"""

import argparse
import json


def adapt(in_paths_and_splits: list[tuple[str, str]], out_path: str) -> dict:
    n_in, n_out, n_skipped_no_image = 0, 0, 0
    by_source = {}
    with open(out_path, "w") as out_f:
        for in_path, split in in_paths_and_splits:
            with open(in_path) as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    rec = json.loads(line)
                    images = rec.get("images") or []
                    if not images:
                        n_skipped_no_image += 1
                        continue
                    if len(images) != 1:
                        raise ValueError(f"{in_path}: expected exactly 1 image, got {len(images)} for {rec.get('source')}")
                    img = images[0]
                    adapted = dict(rec)
                    adapted["images"] = [img["path"]]
                    adapted["images_sha256"] = [img["sha256"]]
                    adapted["split"] = split
                    out_f.write(json.dumps(adapted) + "\n")
                    n_out += 1
                    by_source[rec["source"]] = by_source.get(rec["source"], 0) + 1
    return {"n_in": n_in, "n_out": n_out, "n_skipped_no_image": n_skipped_no_image, "by_source": by_source}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", action="append", required=True, dest="in_jsonls")
    ap.add_argument("--split", action="append", required=True, dest="splits",
                     help="train|eval_id, one per --in-jsonl, same order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.in_jsonls) == len(args.splits), "--in-jsonl and --split counts must match, paired by order"
    stats = adapt(list(zip(args.in_jsonls, args.splits)), args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
