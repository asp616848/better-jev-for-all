"""
Phase 1 reference inference wrapper around the trained `ekvachan-base` encoder
checkpoint (PRD.md Section 7 targets a Rust/ONNX server long-term; this is the
Python reference implementation used to validate the wire contract first,
which is the proportionate thing to build before committing to a runtime port).

Important, honest limitation (see PRD.md Section 10a): the current checkpoint
is trained on ONE fixed `choice` schema — options are always exactly
["entailment", "neutral", "contradiction"] over a premise/hypothesis `state`
(training/data.py). The classifier head never sees `options` text at train
time, so it cannot answer an arbitrary options list the way Jev's real API
promises (e.g. ["billing", "technical", "other"]). This wrapper enforces that
by rejecting any `choice` request whose options don't match what the model
was actually trained on, rather than silently guessing. `score` and `noul`
are not implemented at all yet — no model has been trained for them.
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
