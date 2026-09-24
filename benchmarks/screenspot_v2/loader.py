"""
Loader for ScreenSpot-v2's grounding-as-choice reframing (PRD.md 5.2b:
"Reframed as `choice`, an item is 'here is the screenshot and an instruction;
which of these N elements is the target,' with distractors drawn from the
same screenshot's other annotated elements" -- and PRD.md 13a.11, where the
vision training pipeline this benchmark evaluates was actually built).

**Where the data comes from, and why 858, not the manifest's 898.**
ScreenSpot-v2 is not vendored from HuggingFace directly -- it is built,
license-verified, and hash-committed in the sibling repo `better-jev-bench`
(`better_jev_bench/datasets/screenspot_v2.py`, `datasets/screenspot_v2/
manifest.toml`, both Apache-2.0). That manifest's `item_count = 898` is
`40 public + 858 heldout`, but this dataset is marked `eval_only = true`, and
`better_jev_bench/export.py`'s own slice-gating logic skips any `eval_only`
dataset outright when `--slice public` is requested (verified by reading
`export.py` and then actually running the export -- see below, not assumed
from the manifest's headline number). So the only real, reproducible way to
get ScreenSpot-v2 items out of `bjb export` is `--slice heldout`, and that
slice only ever reads `bench/heldout/<name>.jsonl.gz` -- it does not also
pull in the 40 public-slice rows. The real, run command and its real output:

    cd better-jev-bench && uv run bjb export --slice heldout \\
        --datasets screenspot_v2 --out <dir>
    # -> <dir>/heldout.jsonl, 858 lines (verified 2026-09-24)

**858 real items, vendored here as `vendored/heldout.jsonl`** -- the exact
output of that command, byte for byte (see NOTICE.md for its sha256 and the
license chain). The 40 public-slice rows are real too but structurally
unreachable through the sanctioned export path for an eval_only dataset, so
this benchmark reports 858, not 898, and says why rather than rounding up.

**The grounding-as-choice reframing itself.** Each of those 858 rows, as
`better_jev_bench.datasets.screenspot_v2.ScreenSpotV2` already represents it,
is a *binary* choice: one real target-element descriptor (`label`) against
one real same-screenshot distractor (its cyclic neighbour in reading order --
see that module's own docstring for why it stopped at binary: its export
schema requires one fixed `option_count` per task, and the real per-image
candidate count varies from 2 to 6). `benchmarks.common.backends.
DecoderMultischemaBackend`/`DecoderVisionMultischemaBackend` have no such
fixed-width constraint -- they answer any 2..26-option `choice` question --
so `reframe_records_to_items()` below recovers the fuller, genuine N-way
question PRD.md 5.2b actually specified: group every exported row by its
shared image, and take the union of every row's own (label, distractor) pair
as that image's real candidate pool. Every option string in that pool is
real, verbatim text that already appears in the real, hash-verified export --
this loader invents nothing, it only regroups what's already there.

**Real per-image pool-size distribution** (computed 2026-09-24 over the real
858-row export, not assumed):

    N=2: 412 items   N=3: 310 items   N=4: 92 items   N=5: 39 items   N=6: 5 items

**The "too few annotated elements" case, handled explicitly, not silently.**
A pool of size < 2 cannot support even a binary choice; `reframe_records_to_
items()` excludes any such image's rows and reports the count in its
returned stats dict rather than dropping them unremarked or padding the pool
with an invented option. On the real 858-item export this affects **0
items** -- every image's real pool has >= 2 members, checked, not assumed
(`better_jev_bench`'s own `MIN_CANDIDATES_PER_IMAGE = 2` guarantees this at
build time; this loader re-verifies it independently rather than trusting
that upstream invariant blindly). The synthetic self-test fixture
(`fixtures/selftest.jsonl`) deliberately includes one image whose pool
collapses to a single distinct descriptor, specifically so this exclusion
path is exercised by something, since the real data never triggers it.

One honest caveat on the reconstructed pool, stated rather than assumed: it
is a *lower bound* on an image's true annotated-element count.
`ScreenSpotV2.load_items()` drops a candidate's own row when its instruction
is empty or its descriptor text collides with its cyclic neighbour's; a
dropped candidate can still surface here as someone else's surviving
distractor, but in the (unobserved, on this real 858-row export) case where
both a candidate's own row AND the row it would have been a distractor for
are dropped, it is genuinely lost to this reconstruction. This does not
change any of the 858 items' correctness -- `expected` is always drawn from
a row's own real `label`, which is always in its own image's pool by
construction -- it only means a small, currently-zero number of images might
offer fewer real options than they truly have.

**Images: referenced from better-jev-bench's cache, not vendored here.**
See `resolve_bench_repo_image()` below and README.md's "Vendoring: images vs.
metadata" section for the full tradeoff writeup. Short version: the 356
distinct images behind these 858 items are ~370MB on disk; vendoring a second
copy into this repo's git history would duplicate that, diverge from the
precedent `training/adapt_bench_vision_export.py` already set (PRD.md 13a.11:
"both repos share a server, so `path` is already a real, directly-readable
local file, no copy needed"), and go stale the moment either repo's image
cache changes. This loader instead vendors only the small, portable metadata
(`vendored/heldout.jsonl` -- text, ~150KB) and re-resolves + re-hashes each
image's actual bytes fresh, every load, against better-jev-bench's own
content-addressed cache layout. The real, stated cost of that choice: this
benchmark's real (non-selftest) path cannot run at all without
`better-jev-bench` cloned and built as a sibling of this repo -- exactly the
same requirement `training/adapt_bench_vision_export.py` already has, not a
new one.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from benchmarks.common.items import Item

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent / "vendored"
HELDOUT_FILE = DATA_DIR / "heldout.jsonl"

# Both repos are cloned under one common parent directory on the server this
# was built on (dev-guidelines rule 2: `~/ekvachan/repo` and
# `~/ekvachan/bench-repo`) -- so `REPO_ROOT.parent / "bench-repo"` is a
# portable default rather than a hardcoded machine-specific path. Override
# with `--bench-repo-dir` (run.py) for any other layout.
DEFAULT_BENCH_REPO_DIR = REPO_ROOT.parent / "bench-repo"
IMAGE_CACHE_SUBDIR = "data/images/screenspot_v2"

_MEDIA_TYPE_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}


class ImageResolutionError(RuntimeError):
    """Raised when a real image referenced by a vendored record can't be
    found or doesn't hash-verify -- always fatal, never silently skipped,
    since a screenshot that can't be trusted shouldn't be scored against."""


def resolve_bench_repo_image(bench_repo_dir: Path, sha256: str, media_type: str) -> Path:
    """Real image path resolution for the vendored dataset: content-
    addressed, bucketed by the first 2 hex characters of the sha256 -- the
    real layout of `better-jev-bench/data/images/screenspot_v2/` (confirmed
    by listing it on the server: `01/`, `02/`, ..., `ff/`, each holding files
    named `<sha256>.<ext>`, 370MB / 356 distinct images total). Never trusts
    a machine-specific absolute `path` string that might have been baked into
    some export at some other time on some other checkout -- only the
    (sha256, media_type) pair, which is exactly what travels in the small
    vendored JSONL committed to this repo. Re-hashes the file's actual bytes
    before returning it, rather than trusting the cache's own naming
    convention to mean what it claims.
    """
    if not bench_repo_dir.exists():
        raise ImageResolutionError(
            f"better-jev-bench not found at {bench_repo_dir} -- ScreenSpot-v2's real images live "
            "in its content-addressed cache, not vendored into this repo (see README.md's "
            "'Vendoring: images vs. metadata'). Clone it as a sibling of this repo, run `bjb build "
            "--datasets screenspot_v2`, or pass --bench-repo-dir to point at wherever it lives."
        )
    ext = _MEDIA_TYPE_EXT.get(media_type)
    if ext is None:
        raise ImageResolutionError(f"unrecognized media_type {media_type!r} for image {sha256}")
    path = bench_repo_dir / IMAGE_CACHE_SUBDIR / sha256[:2] / f"{sha256}.{ext}"
    if not path.exists():
        raise ImageResolutionError(
            f"image {sha256} not found at {path} -- is better-jev-bench's screenspot_v2 dataset "
            "actually built there (`bjb build --datasets screenspot_v2`)?"
        )
    real_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if real_hash != sha256:
        raise ImageResolutionError(
            f"image at {path} hashes to {real_hash}, expected {sha256} -- refusing to score "
            "against a screenshot whose bytes don't match this record's own provenance."
        )
    return path


def reframe_records_to_items(
    records: list[dict], resolve_image, benchmark_source_split: str,
) -> tuple[list[Item], dict]:
    """The actual grounding-as-choice reframing -- see the module docstring
    above for the full reasoning. `resolve_image(sha256, media_type) -> Path`
    is injected so this same function serves both the real loader
    (`resolve_bench_repo_image`, above) and `fixtures/selftest.py`'s
    synthetic-image resolver, exercising identical grouping/pool-
    reconstruction logic either way -- selftest is not a shortcut around this
    function, it's the same function fed synthetic input.
    """
    by_image: dict[str, list[dict]] = collections.defaultdict(list)
    image_meta: dict[str, dict] = {}
    for rec in records:
        img = rec["images"][0]
        by_image[img["sha256"]].append(rec)
        image_meta[img["sha256"]] = img

    items: list[Item] = []
    excluded_pool_too_small = 0
    pool_size_histogram: collections.Counter = collections.Counter()

    for sha256, recs in by_image.items():
        pool: list[str] = []
        seen: set[str] = set()
        for r in recs:
            for opt in r["options"]:
                if opt not in seen:
                    seen.add(opt)
                    pool.append(opt)

        if len(pool) < 2:
            # Explicit, counted exclusion -- never a silent drop, never
            # padded with an invented option. See module docstring: 0 real
            # items hit this path today; this branch exists so that if a
            # future re-export ever does trigger it, the run reports it
            # rather than crashing or quietly under-counting.
            excluded_pool_too_small += len(recs)
            continue

        pool_size_histogram[len(pool)] += len(recs)
        image_path = resolve_image(sha256, image_meta[sha256]["media_type"])
        # Deterministic, not fixed-position: sorting the pool avoids "the
        # correct answer's list position correlates with anything" the way a
        # constant order might -- same reasoning ScreenSpotV2.load_items()
        # itself gives for its own stable-hash pair ordering, applied here to
        # an N-way list rather than a binary pair.
        sorted_pool = sorted(pool)

        for i, r in enumerate(recs):
            item_id = "ss2v2-" + hashlib.sha256(f"{sha256}|{r['state']}|{i}".encode()).hexdigest()[:16]
            items.append(Item(
                benchmark="screenspot_v2",
                source_split=benchmark_source_split,
                item_id=item_id,
                question_type="choice",
                state=r["state"],
                instructions=r["instructions"],
                options=sorted_pool,
                expected=r["label"],
                image_path=str(image_path),
            ))

    stats = {
        "n_images": len(by_image),
        "n_records_in": len(records),
        "n_items_out": len(items),
        "n_excluded_pool_too_small": excluded_pool_too_small,
        "pool_size_histogram": dict(sorted(pool_size_histogram.items())),
    }
    return items, stats


def load_heldout(bench_repo_dir: Path | None = None) -> tuple[list[Item], dict]:
    """Load the real, vendored ScreenSpot-v2 heldout export (858 rows -- see
    module docstring for why this is 858, not the manifest's headline 898)
    and reframe it into genuine N-way grounding-as-choice items.
    `bench_repo_dir` defaults to `DEFAULT_BENCH_REPO_DIR` (a `bench-repo`
    sibling of this repo); pass an explicit path for any other layout.
    """
    resolved_bench_repo_dir = Path(bench_repo_dir) if bench_repo_dir else DEFAULT_BENCH_REPO_DIR
    records = [json.loads(line) for line in HELDOUT_FILE.read_text().splitlines() if line.strip()]

    def resolver(sha256: str, media_type: str) -> Path:
        return resolve_bench_repo_image(resolved_bench_repo_dir, sha256, media_type)

    return reframe_records_to_items(records, resolver, "heldout:bjb-export")
