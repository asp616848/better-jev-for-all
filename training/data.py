"""
Phase 1 data pipeline: the adversarial-NLI slice of the Section 5.3 data mix
(ANLI + WANLI + MultiNLI + SNLI), normalized into ekVachan's `choice` schema.

This is a real, verified-against-primary-source pipeline, not a stub:
- Dataset IDs and column/label schemas below were confirmed by actually loading
  each dataset on 2026-09-22, not assumed from memory (see PRD.md Section 5.3
  and the commit history for the verification transcript).
- SNLI ships ~0.1% rows with label == -1 (no annotator consensus) — dropped.
- WANLI encodes its label as strings under `gold`; ANLI/MultiNLI/SNLI use
  ClassLabel ints — both normalized to the same LABELS order below.
- Global dedup on (premise, hypothesis) is applied *after* pooling train and
  eval separately, and again to strip any eval pair that also appears in
  train — cheap leakage a quality-first pipeline shouldn't ship with.
"""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, Dataset

LABELS = ["entailment", "neutral", "contradiction"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}

QUESTION_KEY = "relation"
INSTRUCTIONS = (
    "Given a premise and a hypothesis, determine whether the hypothesis is "
    "entailed by the premise (must be true if the premise is true), neutral "
    "(could be true or false), or contradicted by the premise (cannot be true "
    "if the premise is true)."
)


def _pair_hash(premise: str, hypothesis: str) -> str:
    return hashlib.sha1(f"{premise.strip()}\x1f{hypothesis.strip()}".encode()).hexdigest()


def _to_records(premises, hypotheses, label_idxs, source: str) -> list[dict]:
    out = []
    for p, h, y in zip(premises, hypotheses, label_idxs):
        if y is None or y < 0 or y >= len(LABELS):
            continue
        p, h = (p or "").strip(), (h or "").strip()
        if not p or not h:
            continue
        out.append(
            {
                "state": f"Premise: {p}\nHypothesis: {h}",
                "question_key": QUESTION_KEY,
                "question_type": "choice",
                "instructions": INSTRUCTIONS,
                "options": LABELS,
                "label": LABELS[y],
                "label_idx": y,
                "source": source,
                "_pair_hash": _pair_hash(p, h),
            }
        )
    return out


def load_anli() -> tuple[list[dict], list[dict]]:
    train, eval_ = [], []
    for round_ in ("r1", "r2", "r3"):
        tr = load_dataset("facebook/anli", split=f"train_{round_}")
        train += _to_records(tr["premise"], tr["hypothesis"], tr["label"], f"anli_{round_}")
        for split in (f"dev_{round_}", f"test_{round_}"):
            ev = load_dataset("facebook/anli", split=split)
            eval_ += _to_records(ev["premise"], ev["hypothesis"], ev["label"], f"anli_{round_}_{split.split('_')[0]}")
    return train, eval_


def load_wanli() -> tuple[list[dict], list[dict]]:
    label_str_to_idx = {"entailment": 0, "neutral": 1, "contradiction": 2}
    tr = load_dataset("alisawuffles/WANLI", split="train")
    train = _to_records(tr["premise"], tr["hypothesis"], [label_str_to_idx.get(g, -1) for g in tr["gold"]], "wanli")
    te = load_dataset("alisawuffles/WANLI", split="test")
    eval_ = _to_records(te["premise"], te["hypothesis"], [label_str_to_idx.get(g, -1) for g in te["gold"]], "wanli_test")
    return train, eval_


def load_mnli() -> tuple[list[dict], list[dict]]:
    tr = load_dataset("nyu-mll/multi_nli", split="train")
    train = _to_records(tr["premise"], tr["hypothesis"], tr["label"], "mnli")
    eval_ = []
    for split in ("validation_matched", "validation_mismatched"):
        ev = load_dataset("nyu-mll/multi_nli", split=split)
        eval_ += _to_records(ev["premise"], ev["hypothesis"], ev["label"], f"mnli_{split}")
    return train, eval_


def load_snli() -> tuple[list[dict], list[dict]]:
    tr = load_dataset("stanfordnlp/snli", split="train")
    train = _to_records(tr["premise"], tr["hypothesis"], tr["label"], "snli")
    eval_ = []
    for split in ("validation", "test"):
        ev = load_dataset("stanfordnlp/snli", split=split)
        eval_ += _to_records(ev["premise"], ev["hypothesis"], ev["label"], f"snli_{split}")
    return train, eval_


def dedup(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        if r["_pair_hash"] in seen:
            continue
        seen.add(r["_pair_hash"])
        out.append(r)
    return out


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"sources": {}}

    all_train, all_eval = [], []
    for name, loader in [("anli", load_anli), ("wanli", load_wanli), ("mnli", load_mnli), ("snli", load_snli)]:
        tr, ev = loader()
        stats["sources"][name] = {"train_raw": len(tr), "eval_raw": len(ev)}
        all_train += tr
        all_eval += ev

    before_train, before_eval = len(all_train), len(all_eval)
    all_train = dedup(all_train)
    all_eval = dedup(all_eval)

    train_hashes = {r["_pair_hash"] for r in all_train}
    all_eval = [r for r in all_eval if r["_pair_hash"] not in train_hashes]

    for r in all_train + all_eval:
        del r["_pair_hash"]

    stats["train_before_dedup"] = before_train
    stats["train_after_dedup"] = len(all_train)
    stats["eval_before_dedup_and_leak_filter"] = before_eval
    stats["eval_after_dedup_and_leak_filter"] = len(all_eval)

    import collections
    stats["train_label_balance"] = dict(collections.Counter(r["label"] for r in all_train))
    stats["eval_label_balance"] = dict(collections.Counter(r["label"] for r in all_eval))

    Dataset.from_list(all_train).save_to_disk(str(out_dir / "train"))
    Dataset.from_list(all_eval).save_to_disk(str(out_dir / "eval"))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    return stats


if __name__ == "__main__":
    stats = build(Path(__file__).resolve().parent.parent / "data" / "processed" / "nli_slice")
    print(json.dumps(stats, indent=2))
