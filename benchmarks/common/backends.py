"""
Backends that can answer a `choice` request, all exposing the same interface
(`labels`, `describe()`, `predict_choice()`) so benchmarks/jevbench/run.py and
benchmarks/jabr_v2/run.py don't care which one they're pointed at.

- InProcessBackend: imports serve.inference.EncoderChoiceModel directly and
  runs the trained checkpoint in this process. Deterministic, no server
  lifecycle to manage -- the default, and what produces a real evidence-
  bundle score. Requires torch/transformers and a real checkpoint on disk
  (checkpoints/ is gitignored -- see PRD.md Section 12).
- HTTPBackend: calls a running `POST /v1/systemone` server (serve/server.py)
  over the network with the stdlib http client -- no new dependency needed
  (mirrors the pattern JevBench's own adapters/base.py uses). Adds real
  wire/serialization overhead on top of InProcessBackend; it's the only
  backend that measures what a client of the reference server actually
  experiences, the way PRD.md 13a.3's 27-36ms number was measured server-side.
- MockBackend: a tiny, explicitly non-trained heuristic over the exact
  "Premise: ...\\nHypothesis: ..." state format training/data.py produces.
  It exists purely so the harness's wiring (schema filter -> backend call ->
  scoring -> evidence bundle) can be run and checked end to end in an
  environment with no GPU, no torch, and no checkpoint on disk -- exactly the
  situation this was built in (see benchmarks/README.md). Its accuracy means
  nothing about ekvachan-base; every manifest it produces is stamped
  "harness_selftest": true by benchmarks/common/harness.py so it can never be
  mistaken for a benchmark score.

InProcessBackend imports serve.inference lazily (inside __init__, not at
module import time) specifically so that importing this file -- and
therefore running the schema filter, which is the part that has to work
everywhere -- never requires torch/transformers to be installed. HTTPBackend
never imports serve.inference at all, for the same reason: it only ever
speaks HTTP, so it uses benchmarks.common.schema.FIXED_CHECKPOINT_LABELS.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from benchmarks.common.schema import FIXED_CHECKPOINT_LABELS


class ChoiceBackend(Protocol):
    name: str
    labels: list[str]

    def describe(self) -> dict: ...
    def predict_choice(self, state: str, options: list[str]) -> dict: ...


class InProcessBackend:
    """Runs serve/inference.py's EncoderChoiceModel directly in this process."""

    name = "in_process"

    def __init__(self, checkpoint_dir: Path | None = None):
        from serve.inference import EncoderChoiceModel, LABELS, load_default

        self.labels = list(LABELS)
        self._model = EncoderChoiceModel(checkpoint_dir) if checkpoint_dir else load_default()

    def describe(self) -> dict:
        m = self._model
        return {
            "backend": self.name,
            "checkpoint_name": m.model_name,
            "weight_sha256": m.manifest.get("weight_sha256"),
            "temperature": m.manifest.get("temperature"),
            "base_model": m.manifest.get("base_model"),
            "device": m.device,
        }

    def predict_choice(self, state: str, options: list[str]) -> dict:
        t0 = time.perf_counter()
        result = dict(self._model.predict_choice(state, options))
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return result


class HTTPBackend:
    """Calls a running serve/server.py instance's POST /v1/systemone."""

    name = "http"

    def __init__(self, endpoint: str = "http://127.0.0.1:8000"):
        # Deliberately FIXED_CHECKPOINT_LABELS, not an import of serve.inference.LABELS:
        # this backend's whole point is to need no local ML stack (it only ever talks
        # HTTP), and serve/inference.py imports torch/transformers at module level. The
        # wire contract doesn't self-describe its supported schema (the server just 501s
        # on a mismatch), so this is the same duplicated constant InProcessBackend checks
        # itself against -- see benchmarks/common/schema.py's docstring.
        self.labels = list(FIXED_CHECKPOINT_LABELS)
        self.endpoint = endpoint.rstrip("/")
        # Fail fast rather than reporting a run full of connection errors as
        # "0 correct" -- confirm the server is actually reachable up front.
        req = urllib.request.Request(f"{self.endpoint}/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                raise ConnectionError(f"server at {self.endpoint} unhealthy: HTTP {resp.status}")

    def describe(self) -> dict:
        return {"backend": self.name, "endpoint": self.endpoint}

    def predict_choice(self, state: str, options: list[str]) -> dict:
        body = {"state": state, "questions": {"decision": {"type": "choice", "options": options}}}
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/v1/systemone",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"server returned HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from e
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = dict(payload["results"]["decision"])
        result["latency_ms"] = latency_ms
        return result


class MockBackend:
    """Deterministic, non-trained heuristic. Harness wiring self-test only --
    see the module docstring above. Never used against the real vendored
    datasets by benchmarks/common/harness.py's CLI (--selftest is required to
    reach the fixtures this can meaningfully "answer" at all)."""

    name = "mock"
    labels = list(FIXED_CHECKPOINT_LABELS)

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "note": "NOT a trained model -- a deterministic keyword heuristic used only to "
                     "prove the harness pipeline (filter -> call -> score -> evidence bundle) "
                     "works end to end without torch, transformers, or a checkpoint on disk.",
        }

    def predict_choice(self, state: str, options: list[str]) -> dict:
        if list(options) != self.labels:
            raise ValueError(f"MockBackend only answers {self.labels}, got {options}")
        t0 = time.perf_counter()

        _, _, hyp_part = state.partition("Hypothesis:")
        premise_part = state.partition("Hypothesis:")[0]
        premise = premise_part.replace("Premise:", "").strip().lower()
        hypothesis = hyp_part.strip().lower()

        negations = (" no ", " not ", "n't ", " never ", " none ")
        hyp_negated = any(n in f" {hypothesis} " for n in negations)
        premise_negated = any(n in f" {premise} " for n in negations)
        premise_words = set(premise.replace(".", "").replace(",", "").split())
        hyp_content_words = [w.strip(".,") for w in hypothesis.split() if len(w.strip(".,")) > 3]
        overlap = sum(1 for w in hyp_content_words if w in premise_words)
        overlap_ratio = overlap / max(len(hyp_content_words), 1)

        if hyp_negated != premise_negated:
            label, conf = "contradiction", 0.75
        elif overlap_ratio >= 0.6:
            label, conf = "entailment", 0.7
        else:
            label, conf = "neutral", 0.6

        other = (1.0 - conf) / 2
        probs = {l: (conf if l == label else other) for l in self.labels}
        return {
            "choice": label,
            "probabilities": probs,
            "confidence": conf,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

class DecoderMultischemaBackend:
    """Runs a LoRA-tuned decoder checkpoint from training/train_decoder_lora_wideschema.py (or
    its narrower sibling train_decoder_lora_multischema.py) via the restricted-logit-read
    mechanism, at whatever option count (2..manifest["max_options"]) each item actually has --
    unlike InProcessBackend, which only ever answers the one fixed 3-way schema baked into
    benchmarks.common.schema.FIXED_CHECKPOINT_LABELS. Pair this with
    benchmarks.common.schema_filter.filter_supported_multischema and
    benchmarks.common.schema.DECODER_MULTISCHEMA_MAX_OPTIONS.

    Prompt construction and the letter-logit read exactly mirror
    train_decoder_lora_wideschema.py's build_prompt()/evaluate(): same
    "instructions\n\nstate\n\nOptions:\nA) ...\n...\n\nAnswer with a single letter (...)."
    layout, same left-padding + last-token restricted-logit softmax. The one deliberate
    difference from training: options are presented in their given (dataset) order rather than
    a per-example seeded shuffle -- shuffling exists at training time to stop the model from
    learning a positional shortcut under gradient descent; at inference there is no gradient to
    protect from, so a fixed, honest presentation order is simpler and equally valid.
    """

    name = "decoder_multischema"

    def __init__(self, checkpoint_dir: Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        manifest_path = checkpoint_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        self.max_options = self.manifest["max_options"]
        self.letters = [chr(ord("A") + i) for i in range(self.max_options)]
        base_model = self.manifest["base_model"]
        adapter_dir = checkpoint_dir / "adapter"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map=self.device)
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()

        letter_ids_full = [self.tokenizer.encode(l, add_special_tokens=False) for l in self.letters]
        for l, ids in zip(self.letters, letter_ids_full):
            assert len(ids) == 1, f"letter {l!r} is not a single token under this tokenizer: {ids}"
        self.letter_ids = torch.tensor([ids[0] for ids in letter_ids_full], device=self.device)
        self.temperature = float(self.manifest.get("temperature") or 1.0)

        self.labels = None  # variable per item -- no fixed label list, unlike FIXED_CHECKPOINT_LABELS backends

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "base_model": self.manifest["base_model"],
            "max_options": self.max_options,
            "temperature": self.temperature,
            "device": self.device,
        }

    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict:
        torch = self._torch
        import torch.nn.functional as F

        t0 = time.perf_counter()
        n = len(options)
        if not (2 <= n <= self.max_options):
            raise ValueError(f"DecoderMultischemaBackend got {n} options, outside [2, {self.max_options}]")
        instructions = instructions or "Choose the option that best answers the question."
        option_lines = "\n".join(f"{self.letters[i]}) {options[i]}" for i in range(n))
        prompt_text = (
            f"{instructions}\n\n{state}\n\nOptions:\n{option_lines}\n\n"
            f"Answer with a single letter ({'/'.join(self.letters[:n])})."
        )
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
            last_logits = out.logits[0, -1, :]
            restricted = last_logits[self.letter_ids[:n]]
            probs = F.softmax(restricted.float() / self.temperature, dim=-1).cpu().numpy()

        probabilities = {options[i]: float(probs[i]) for i in range(n)}
        choice_idx = int(probs.argmax())
        return {
            "choice": options[choice_idx],
            "probabilities": probabilities,
            "confidence": float(probs[choice_idx]),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


class DecoderMultischemaMockBackend:
    """Deterministic, non-trained heuristic for the multischema harness wiring self-test --
    the variable-width sibling of MockBackend above. Always answers with the first option;
    exists purely to exercise filter_supported_multischema -> backend call -> scoring ->
    evidence bundle without needing torch/transformers or a real checkpoint. Never used against
    real vendored datasets outside --selftest (see benchmarks/common/harness.py)."""

    name = "decoder_multischema_mock"
    labels = None

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "note": "NOT a trained model -- always predicts the first option, used only to prove "
                     "the multischema harness pipeline works end to end without torch, "
                     "transformers, or a checkpoint on disk.",
        }

    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict:
        t0 = time.perf_counter()
        n = len(options)
        conf = 1.0 / n
        probabilities = {opt: conf for opt in options}
        return {
            "choice": options[0],
            "probabilities": probabilities,
            "confidence": conf,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
