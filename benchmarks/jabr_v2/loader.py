"""
Loader for jabr-v2 (github.com/jabr/classifier-benchmark, CC0 1.0 Universal --
public domain; verified 2026-09-22, see README.md in this directory). PRD.md
Section 8.1 was written on the assumption that jabr-v2 had "no confirmed
public repo"; it does, and this is it -- Von's own README links directly to
this repo's results/v1v2-summary.md and calls it "the jabr v2 benchmark".
Vendored copies of cases/v1.toml, cases/v2.toml, and cases/hashes.json as
fetched on 2026-09-22 live in benchmarks/jabr_v2/vendored/.

TOML schema (confirmed by downloading and reading the real files, matching
upstream bench/cases.py):

  [[task]]
  id = "..."
  type = "choice" | "noul" | "score"

  [task.question]
  instructions = "..."
  criteria = { label = "description", ... }   # choice: a dict -> its keys are
                                                # the options, in declaration
                                                # order; score: a list of level
                                                # descriptions, index = level

  [[task.cases]]
  state = "..."
  expected = ...   # choice: a label string; noul: true/false; score: an int
                    # level index

v1.toml is upstream-locked (its hash lives in cases/hashes.json, 78 cases,
8 tasks). v2.toml -- the 49-task/869-case suite PRD.md 3.1 and Von's own
README cite -- is still marked "under review" / unlocked upstream as of the
vendor date. See README.md for what that means for reproducibility, and note
this loader records its own sha256 of exactly what was vendored, since
upstream has not frozen one for v2 yet.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from benchmarks.common.items import Item

DATA_DIR = Path(__file__).resolve().parent / "vendored"

SUITE_FILES = {
    "v1": DATA_DIR / "v1.toml",
    "v2": DATA_DIR / "v2.toml",
}


def _options_for(task: dict) -> list[str] | None:
    if task.get("type") != "choice":
        return None
    criteria = task.get("question", {}).get("criteria")
    if not isinstance(criteria, dict):
        return None
    return list(criteria.keys())


def load_toml_file(path: Path, suite_name: str) -> list[Item]:
    data = tomllib.loads(path.read_text())
    items: list[Item] = []
    for task in data.get("task", []):
        options = _options_for(task)
        question = task.get("question", {})
        for i, case in enumerate(task.get("cases", [])):
            items.append(Item(
                benchmark="jabr_v2",
                source_split=f"{suite_name}:{task['id']}",
                item_id=f"{task['id']}-{i}",
                question_type=task["type"],
                state=case["state"],
                instructions=question.get("instructions"),
                options=options,
                expected=case.get("expected"),
            ))
    return items


def load_suite(name: str) -> list[Item]:
    return load_toml_file(SUITE_FILES[name], name)


def load_public(suites: list[str] | None = None) -> list[Item]:
    """Load jabr-v2's public suites. Defaults to both v1 (78 cases, locked)
    and v2 (869 cases, unlocked/under-review upstream) -- both are fully
    public in this repo, unlike JevBench's split public/held-out design."""
    names = suites or list(SUITE_FILES)
    items: list[Item] = []
    for name in names:
        items.extend(load_suite(name))
    return items
