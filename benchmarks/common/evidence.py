"""
Evidence-bundle writer -- implements PRD.md Section 8.2's discipline: raw
outputs, seeds, item/prompt hashes, weight hash, timestamp, all saved under
results/. Every run produces exactly two files:

  results/<run_id>.manifest.json   -- small, meant to be checked into git:
                                      provenance, schema-filter summary,
                                      aggregate scores. No raw model output
                                      lives here.
  results/<run_id>.raw             -- one JSON line per attempted item: its
                                      hash, the full model response, latency,
                                      correctness. Matches the pattern already
                                      in .gitignore (`results/*.raw`), so raw
                                      per-item evidence stays out of the repo
                                      the same way trained checkpoints do,
                                      while the small manifest that documents
                                      *what happened* does not.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def write_bundle(
    *,
    out_dir: Path,
    run_id: str,
    benchmark: str,
    seed: int,
    dataset_provenance: dict[str, Any],
    schema_filter_summary: dict[str, Any],
    model_info: dict[str, Any],
    results_summary: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{run_id}.manifest.json"
    raw_path = out_dir / f"{run_id}.raw"

    manifest = {
        "benchmark": benchmark,
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "harness_git_commit": _git_commit(),
        "dataset": dataset_provenance,
        "schema_filter": schema_filter_summary,
        "model": model_info,
        "results": results_summary,
        "raw_output_file": raw_path.name,
    }
    if extra:
        manifest.update(extra)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False, default=str))
    with raw_path.open("w") as f:
        for row in raw_rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    return manifest_path, raw_path
