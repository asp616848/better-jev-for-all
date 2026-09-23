"""
Reference inference wrappers around the trained checkpoints (PRD.md Section 7
targets a Rust/ONNX server long-term; this is the Python reference
implementation used to validate the wire contract first).

Two model classes live here, on purpose, not because one replaced the other:

- `EncoderChoiceModel` (`ekvachan-base` encoder) — the ORIGINAL Phase 1
  checkpoint. Fixed to exactly one `choice` schema
  (["entailment","neutral","contradiction"]) — the classifier head never sees
  `options` text at train time, so it cannot answer an arbitrary options list.
  Kept as-is and NOT wired into the live server by default anymore (see
  below), because `benchmarks/common/backends.py`'s `InProcessBackend` still
  targets it directly via `load_default()` for the original fixed-schema
  JevBench/jabr-v2 comparison runs (PRD 13a.1/13a.4) — changing what
  `load_default()` returns would silently break that.

- `DecoderChoiceModel` (Qwen3.5-4B LoRA, restricted-logit read) — what the
  live server actually serves now (`serve/server.py`'s `get_model()` calls
  `load_default_decoder()`), because the decoder is the chosen primary
  architecture (PRD.md 14 Q4, decided 2026-09-23). Thin wrapper around
  `benchmarks.common.backends.DecoderMultischemaBackend` -- reuses the exact
  restricted-logit mechanism already proven in PRD 13a.6/13a.7 rather than
  re-implementing it a third time in this file. Answers any `choice` question
  with 2-26 options (PRD 13a.5's real ceiling, not this project's own
  invention), not just one fixed schema.

`score` and `noul` are still not implemented in serve/ (return 501) even
though training data for both now exists (`training/build_primitives_slice.py`,
run in progress as of 2026-09-23) — a trained checkpoint answering them isn't
the same as `serve/`/the wire contract knowing how to present their
primitive-specific output shapes (a float for `noul`, a probability-weighted
scale position for `score`). See STATUS.md for what's actually wired vs still
open.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = ["entailment", "neutral", "contradiction"]


class ChoiceUnsupportedError(ValueError):
    """Raised when a request asks for a choice schema this checkpoint wasn't trained on."""


class EncoderChoiceModel:
    def __init__(self, checkpoint_dir: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = checkpoint_dir / "model"
        manifest_path = checkpoint_dir / "manifest.json"
        if not model_dir.exists():
            raise FileNotFoundError(f"no model dir at {model_dir}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")

        self.manifest = json.loads(manifest_path.read_text())
        self.temperature = self.manifest["temperature"]
        self.model_name = checkpoint_dir.name

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_choice(self, state: str, options: list[str], max_length: int = 256) -> dict:
        if list(options) != LABELS:
            raise ChoiceUnsupportedError(
                f"this checkpoint only supports options=={LABELS} (trained schema); "
                f"got {options}. See PRD.md Section 10a — arbitrary option sets aren't "
                "supported by the Phase 1 encoder checkpoint."
            )

        enc = self.tokenizer(state, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits.float()
        calibrated = F.softmax(logits / self.temperature, dim=-1).squeeze(0).cpu().numpy()

        probabilities = {label: float(p) for label, p in zip(LABELS, calibrated)}
        best_idx = int(calibrated.argmax())
        return {
            "choice": LABELS[best_idx],
            "probabilities": probabilities,
            "confidence": float(calibrated[best_idx]),
        }


def load_default() -> EncoderChoiceModel:
    return EncoderChoiceModel(REPO_ROOT / "checkpoints" / "ekvachan-base-run2")


class DecoderChoiceModel:
    """Serve/-facing wrapper around `benchmarks.common.backends.DecoderMultischemaBackend`.
    Reuses that backend's exact prompt construction and restricted-logit read
    (the same mechanism validated end-to-end in PRD 13a.6/13a.7 against real
    third-party benchmarks) instead of re-implementing it a third time here.
    All heavy imports (torch/transformers/peft, via the backend) are deferred
    to __init__, matching EncoderChoiceModel's own lazy-import discipline --
    this keeps `from serve.inference import ...` importable without those
    packages installed, which sdk/python/tests/test_client.py relies on."""

    def __init__(self, checkpoint_dir: Path, device: str | None = None):
        from benchmarks.common.backends import DecoderMultischemaBackend

        self._backend = DecoderMultischemaBackend(checkpoint_dir, device=device)
        self.model_name = checkpoint_dir.name
        self.max_options = self._backend.max_options

    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict:
        n = len(options)
        if not (2 <= n <= self.max_options):
            raise ChoiceUnsupportedError(
                f"this checkpoint supports 2-{self.max_options} options (PRD.md 13a.5's real "
                f"ceiling for the single-uppercase-letter restricted-logit mechanism), got {n}."
            )
        out = self._backend.predict_choice(state, options, instructions=instructions)
        out.pop("latency_ms", None)  # server.py's own top-level `usage.latency_ms` already covers this
        return out


def load_default_decoder() -> DecoderChoiceModel:
    # Points at the most recent validated checkpoint, not necessarily the first decoder one --
    # ekvachan-decoder-qwen-primitives (PRD 13a.8) is a superset of ekvachan-decoder-qwen-wideschema's
    # (13a.6) choice capability (95.73% vs 96.10% on the same CLINC150 zero-shot regression check --
    # noise-level difference) plus a first pass at noul/score training, so it's strictly the better
    # default even though serve/ doesn't answer noul/score yet.
    return DecoderChoiceModel(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-primitives")


if __name__ == "__main__":
    # Manual smoke test against the real trained checkpoint — not a unit test fixture.
    model = load_default()
    examples = [
        ("Premise: The cat sat on the mat.\nHypothesis: An animal was on the mat.", "entailment"),
        ("Premise: The cat sat on the mat.\nHypothesis: The dog was outside.", "neutral"),
        ("Premise: The cat sat on the mat.\nHypothesis: No animal was on the mat.", "contradiction"),
    ]
    for state, expected in examples:
        t0 = time.time()
        result = model.predict_choice(state, LABELS)
        elapsed_ms = (time.time() - t0) * 1000
        mark = "OK " if result["choice"] == expected else "MISS"
        print(f"[{mark}] expected={expected:13s} got={result['choice']:13s} "
              f"conf={result['confidence']:.3f} ({elapsed_ms:.1f}ms) probs={result['probabilities']}")
