"""
Backends that can answer a `choice` request, all exposing the same interface
(`describe()`, `predict_choice()`, plus either a fixed `labels` list or a
`max_options` cap depending on which schema family they belong to) so
benchmarks/jevbench/run.py and benchmarks/jabr_v2/run.py don't care which one
they're pointed at.

Fixed-schema family (PRD.md 13a.1/13a.3 -- the `ekvachan-base` encoder, one
hardcoded 3-way head, `labels` is always exactly
`benchmarks.common.schema.FIXED_CHECKPOINT_LABELS`):

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

Decoder / multischema family (PRD.md Section 14 Q4, 13a.5 -- the Qwen3.5-4B
LoRA decoder, restricted-logit read, answers ANY `choice` question with 2 to
`max_options` options; see benchmarks/common/schema.py for where that cap
comes from):

- DecoderMultischemaBackend: loads a LoRA adapter checkpoint produced by
  `training/train_decoder_lora_wideschema.py` (or its `_multischema`
  predecessor -- same manifest/adapter shape, smaller `max_options`) via
  `peft.PeftModel` on top of the base causal LM named in the checkpoint's
  manifest, and answers a `choice` question the same way that script's own
  `evaluate()` does at inference time: read the next-token logits restricted
  to the single-uppercase-letter tokens for this item's own option count,
  softmax over just those, argmax for the answer. See its docstring below for
  exactly which parts of that script's logic this mirrors and why.
- DecoderMultischemaMockBackend: the multischema-path equivalent of
  MockBackend -- a tiny, explicitly non-trained keyword-overlap heuristic
  that accepts any 2-to-`max_options`-option `choice` question, so the
  multischema harness wiring (filter -> call -> score -> evidence bundle) can
  be proven end to end with no torch/transformers/peft/checkpoint at all.
  Same "never mistaken for a real score" discipline as MockBackend.

InProcessBackend and DecoderMultischemaBackend both import their ML stack
lazily (inside __init__, not at module import time) specifically so that
importing this file -- and therefore running either schema filter, which is
the part that has to work everywhere -- never requires torch/transformers/
peft to be installed. HTTPBackend never imports serve.inference at all, for
the same reason: it only ever speaks HTTP, so it uses
benchmarks.common.schema.FIXED_CHECKPOINT_LABELS.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from benchmarks.common.schema import DECODER_MULTISCHEMA_MAX_OPTIONS, FIXED_CHECKPOINT_LABELS


class ChoiceBackend(Protocol):
    name: str
    labels: list[str]

    def describe(self) -> dict: ...
    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict: ...
    # Decoder/multischema backends below expose `max_options: int` instead of
    # `labels: list[str]` -- they don't have one fixed option set to name.


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

    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict:
        # instructions is accepted for interface parity with the multischema
        # backends but unused here: the fixed 3-way classifier head never saw
        # `options`/instructions text at train time (PRD.md 10a) -- it only
        # ever reads `state`.
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

    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict:
        # instructions unused -- see InProcessBackend.predict_choice's comment;
        # the wire contract this backend speaks doesn't carry it either.
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

    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict:
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


def _build_multischema_prompt(state: str, options: list[str], letters: list[str], instructions: str | None) -> str:
    """Same shape as `training/train_decoder_lora_wideschema.py`'s (and its
    `_multischema` predecessor's) `build_prompt()`, specialized to the
    identity ordering (letter i always names options[i]) since there is no
    training-time position-bias concern at inference: the model was trained
    on randomly-shuffled letter assignments specifically so it can't learn
    "the answer is always A" as a shortcut, which is exactly what makes any
    fixed presentation order -- including natural order, used here -- fine to
    serve with."""
    n = len(options)
    option_lines = "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))
    instructions = instructions or "Choose the option that best answers the question below."
    return (
        f"{instructions}\n\n{state}\n\nOptions:\n{option_lines}\n\n"
        f"Answer with a single letter ({'/'.join(letters[:n])})."
    )


class DecoderMultischemaBackend:
    """Runs a LoRA-adapter checkpoint produced by
    `training/train_decoder_lora_wideschema.py` (or its `_multischema`
    predecessor -- both save the same `<checkpoint_dir>/manifest.json` +
    `<checkpoint_dir>/adapter/` shape, just with a different `max_options` in
    the manifest) in this process, and answers a `choice` question the way
    that script's own `evaluate()` does at inference time:

      1. Build a prompt listing this item's own options as A), B), C), ...
         (see `_build_multischema_prompt` above) and apply the base model's
         chat template with `add_generation_prompt=True`.
      2. Run one forward pass, take the last token's logits (this is a
         single, unpadded sequence, so "last token" is unambiguous -- no
         left-padding bookkeeping needed the way the batched training-time
         `evaluate()` requires).
      3. Restrict those logits to the single-token ids for letters
         `A..<n-th letter>` (only this item's own `n = len(options)` of
         them -- there is nothing to mask to -inf here, since we simply never
         look at logit columns beyond `n` in the first place; softmax over an
         already-restricted-to-n vector is mathematically identical to the
         training script's mask-to-`-inf`-then-softmax-over-`MAX_OPTIONS`
         approach, since a softmax is invariant to dropping `-inf` terms).
      4. Softmax (optionally after dividing by a fitted temperature -- see
         `apply_temperature` below), argmax for the answer.

    **Calibration is raw by default, not the manifest's fitted temperature**:
    PRD.md 13a.2/13a.5 both found temperature scaling makes the decoder
    family's ECE and Brier *worse*, not better, likely because the
    calibration fit works from a `log(restricted_softmax_probs)` surrogate
    rather than true pre-softmax logits -- "use raw for now" is that
    section's own stated conclusion. Pass `apply_temperature=True` (or
    `--decoder-apply-temperature` on the CLI) to opt back in once that
    procedure is fixed; until then this backend's default matches the
    documented recommendation rather than silently trusting the "calibrated"
    number.

    Requires torch/transformers/peft and a real checkpoint on disk -- all
    imported lazily in __init__, same reasoning as InProcessBackend's."""

    name = "decoder_multischema"

    def __init__(
        self,
        checkpoint_dir: Path,
        base_model: str | None = None,
        device: str | None = None,
        apply_temperature: bool = False,
    ):
        # Check the checkpoint exists before importing the ML stack, so a
        # wrong --checkpoint-dir fails with a clear FileNotFoundError even in
        # an environment (like the one this was built in) with no torch/
        # transformers/peft installed at all -- same reasoning as
        # InProcessBackend, applied slightly more defensively since this
        # backend has three heavy imports instead of one.
        manifest_path = checkpoint_dir / "manifest.json"
        adapter_dir = checkpoint_dir / "adapter"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        if not adapter_dir.exists():
            raise FileNotFoundError(f"no adapter dir at {adapter_dir}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.manifest = json.loads(manifest_path.read_text())
        self.checkpoint_name = checkpoint_dir.name
        self.max_options = int(self.manifest["max_options"])
        self.letters = [chr(ord("A") + i) for i in range(self.max_options)]
        self.apply_temperature = apply_temperature
        self.temperature = float(self.manifest.get("temperature", 1.0)) if apply_temperature else 1.0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        base_model_name = base_model or self.manifest["base_model"]
        self.base_model_name = base_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.to(self.device)
        self.model.eval()

        # Same runtime check train_decoder_lora_wideschema.py's main() makes,
        # re-verified here rather than trusted from the manifest: refuse to
        # silently misread a letter position if this checkpoint's tokenizer
        # doesn't tokenize A..<max_options-th letter> as single, mutually
        # distinct tokens.
        letter_ids: list[int] = []
        for letter in self.letters:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"letter {letter!r} is not a single token under {base_model_name}'s tokenizer: {ids} "
                    "-- this checkpoint's restricted-logit mechanism assumes exactly one token per letter."
                )
            letter_ids.append(ids[0])
        if len(set(letter_ids)) != len(letter_ids):
            raise RuntimeError(f"letter token ids are not mutually distinct: {letter_ids}")
        self._letter_ids = letter_ids

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "checkpoint_name": self.checkpoint_name,
            "base_model": self.base_model_name,
            "architecture": self.manifest.get("architecture"),
            "max_options": self.max_options,
            "temperature_applied": self.apply_temperature,
            "temperature": self.temperature if self.apply_temperature else None,
            "calibration_note": (
                "raw (uncalibrated) probabilities by default -- PRD.md 13a.2/13a.5 found temperature "
                "scaling makes this architecture's ECE/Brier worse, not better; pass "
                "apply_temperature=True / --decoder-apply-temperature to override"
            ),
            "device": self.device,
        }

    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict:
        n = len(options)
        if not (2 <= n <= self.max_options):
            raise ValueError(
                f"DecoderMultischemaBackend supports 2-{self.max_options} options (this checkpoint's "
                f"manifest max_options={self.max_options}), got {n}"
            )
        torch = self._torch
        t0 = time.perf_counter()

        prompt_text = _build_multischema_prompt(state, options, self.letters, instructions)
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
        last_logits = out.logits[0, -1, :].float()  # unpadded single sequence -- last column is the last real token
        letter_ids_n = torch.tensor(self._letter_ids[:n], device=last_logits.device)
        restricted = last_logits[letter_ids_n] / self.temperature
        probs_t = torch.softmax(restricted, dim=-1).cpu().numpy()

        probabilities = {opt: float(p) for opt, p in zip(options, probs_t)}
        best_idx = int(probs_t.argmax())
        return {
            "choice": options[best_idx],
            "probabilities": probabilities,
            "confidence": float(probs_t[best_idx]),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


class DecoderMultischemaMockBackend:
    """The multischema-path equivalent of MockBackend above: a tiny,
    explicitly non-trained keyword-overlap heuristic that accepts any
    2-to-`max_options`-option `choice` question. Exists purely so the
    multischema harness wiring (filter -> call -> score -> evidence bundle)
    can be run and checked end to end with no torch, transformers, peft, or
    checkpoint on disk -- the decoder/multischema analogue of the situation
    MockBackend's own docstring describes. Its accuracy means nothing about
    any real decoder checkpoint; results produced with it are only ever
    trustworthy as a wiring proof, never as a benchmark score."""

    name = "decoder_multischema_mock"

    def __init__(self, max_options: int = DECODER_MULTISCHEMA_MAX_OPTIONS):
        self.max_options = max_options

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "max_options": self.max_options,
            "note": "NOT a trained model -- a deterministic keyword-overlap heuristic used only to "
                    "prove the multischema harness pipeline (filter -> call -> score -> evidence "
                    "bundle) works end to end without torch, transformers, peft, or a checkpoint on "
                    "disk.",
        }

    def predict_choice(self, state: str, options: list[str], instructions: str | None = None) -> dict:
        n = len(options)
        if not (2 <= n <= self.max_options):
            raise ValueError(f"DecoderMultischemaMockBackend supports 2-{self.max_options} options, got {n}")
        t0 = time.perf_counter()

        context = f"{instructions or ''} {state}".lower()
        context_words = set(w.strip(".,:;!?()\"'") for w in context.split())

        scores = []
        for opt in options:
            opt_words = [w.strip(".,:;!?()\"'") for w in opt.lower().split() if len(w.strip(".,:;!?()\"'")) > 2]
            overlap = sum(1 for w in opt_words if w in context_words)
            scores.append(overlap / max(len(opt_words), 1))

        best_idx = max(range(n), key=lambda i: (scores[i], -i))
        # Turn raw overlap scores into a non-degenerate distribution (a flat
        # +0.05 floor keeps every option's probability strictly positive) --
        # meaningful enough evidence-bundle output to exercise Brier/ECE
        # without pretending this heuristic is a real confidence estimate.
        floored = [s + 0.05 for s in scores]
        z = sum(floored)
        probabilities = {opt: v / z for opt, v in zip(options, floored)}
        return {
            "choice": options[best_idx],
            "probabilities": probabilities,
            "confidence": probabilities[options[best_idx]],
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
