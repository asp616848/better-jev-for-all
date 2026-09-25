"""
Backends that can answer a `choice` request, all exposing the same interface
(`describe()`, `predict_choice()`, plus either a fixed `labels` list or a
`max_options` cap depending on which schema family they belong to) so
benchmarks/jevbench/run.py, benchmarks/jabr_v2/run.py, and
benchmarks/screenspot_v2/run.py don't care which one they're pointed at.

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
  `training/train_decoder_lora_wideschema.py` (or its `_multischema`/
  `_benchcorpus`/`_general` (text-only data) predecessors -- same
  manifest/adapter shape) via `peft.PeftModel` on top of the base causal LM
  named in the checkpoint's manifest, and answers a `choice` question the
  same way that script's own `evaluate()` does at inference time: read the
  next-token logits restricted to the single-uppercase-letter tokens for this
  item's own option count, softmax over just those, argmax for the answer.
  See its docstring below for exactly which parts of that script's logic this
  mirrors and why.
- DecoderMultischemaMockBackend: the multischema-path equivalent of
  MockBackend -- a tiny, explicitly non-trained keyword-overlap heuristic
  that accepts any 2-to-`max_options`-option `choice` question, so the
  multischema harness wiring (filter -> call -> score -> evidence bundle) can
  be proven end to end with no torch/transformers/peft/checkpoint at all.
  Same "never mistaken for a real score" discipline as MockBackend.

Decoder / vision-multischema family (PRD.md 5.2b, 13a.11 -- the same
Qwen3.5-4B LoRA decoder, trained via `training/train_decoder_lora_general.py`
on a mix of text and image rows, loaded through the vision-language model
class instead of the causal-LM class so a `choice` item can carry a real
screenshot):

- DecoderVisionMultischemaBackend: the vision-capable sibling of
  DecoderMultischemaBackend, not a rewrite of it. It reuses the exact same
  restricted-logit mechanism (factored out below into
  `_build_code_table_for_backend()` and `_restricted_logit_result()`
  specifically so this file never reimplements that mechanism a third time --
  PRD.md 5.2b's own probe found "the restricted-logit read still works,
  unchanged" once a vision-language forward pass is substituted for a
  causal-LM one, and this backend is that finding turned into a benchmark
  backend). What actually differs from DecoderMultischemaBackend, and why
  each difference exists, is documented on the class itself.
- DecoderVisionMultischemaMockBackend: the vision-path wiring self-test,
  same non-trained-heuristic discipline as the two Mock backends above. It
  accepts (and ignores) `image_path` -- a text keyword-overlap heuristic has
  no way to look at an image, which is exactly why its accuracy proves
  nothing about grounding ability and only ever proves the harness wiring
  runs end to end without torch/transformers/peft/a checkpoint/a real
  screenshot on disk.

InProcessBackend, DecoderMultischemaBackend, and DecoderVisionMultischemaBackend
all import their ML stack lazily (inside __init__, not at module import time)
specifically so that importing this file -- and therefore running any schema
filter, which is the part that has to work everywhere -- never requires
torch/transformers/peft to be installed. HTTPBackend never imports
serve.inference at all, for the same reason: it only ever speaks HTTP, so it
uses benchmarks.common.schema.FIXED_CHECKPOINT_LABELS.

Every `predict_choice()` below accepts an `image_path: str | None = None`
keyword, for the same reason `instructions` is already accepted-but-unused by
the fixed-schema backends: interface parity. Only DecoderVisionMultischemaBackend
(and its mock) actually reads it; every other backend ignores it, since none
of JevBench, jabr-v2, or the fixed/text-multischema checkpoints have ever seen
an image.

Random baseline (PRD.md 8.1d Finding 2 -- added for the ViZDoom episodic
harness, but deliberately general rather than ViZDoom-specific, since "what
does a policy with no intelligence at all score" is a control every choice
benchmark here benefits from being able to run, not just ViZDoom):

- RandomChoiceBackend: answers any `choice` question (any option count,
  including a fixed 3-way schema) by drawing uniformly from the options it
  was actually offered, seeded with `random.Random(seed)`. Unlike the Mock
  backends above, this is not a wiring self-test -- it is a real, meaningful,
  publishable baseline in its own right (PRD.md 8.1d Finding 2's "a uniform
  random policy's survival time beats Von's own published number" finding
  depends on exactly this backend existing and being run for real, not
  mocked). `describe()` still flags it plainly as "NOT a trained model" so it
  is never mistaken for one downstream.
"""

from __future__ import annotations

import json
import random
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
    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict: ...
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

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        # instructions/image_path accepted for interface parity but unused
        # here: the fixed 3-way classifier head never saw `options`/
        # instructions/image text at train time (PRD.md 10a) -- it only ever
        # reads `state`.
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

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        # instructions/image_path unused -- see InProcessBackend.predict_choice's
        # comment; the wire contract this backend speaks doesn't carry either.
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

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
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


class RandomChoiceBackend:
    """Uniform-random baseline -- see the module docstring's "Random
    baseline" section for why this exists and is not merely a wiring
    self-test. Works with any option count (it does not belong to either the
    fixed-schema or the decoder/multischema family; it never checks `options`
    against a fixed list at all), which is exactly what makes it usable
    across every harness in this repo, not just the episodic ViZDoom one it
    was added for."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self.seed = seed

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "seed": self.seed,
            "note": "NOT a trained model -- a uniform-random baseline over whatever options an "
                    "item/decision actually offers. A real, publishable control (PRD.md 8.1d "
                    "Finding 2), not a wiring self-test like the Mock backends.",
        }

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        n = len(options)
        choice = self._rng.choice(options)
        p = 1.0 / n
        probabilities = {opt: p for opt in options}
        return {
            "choice": choice,
            "probabilities": probabilities,
            "confidence": p,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


def _build_multischema_prompt(state: str, options: list[str], letters: list[str], instructions: str | None) -> str:
    """Same shape as `training/train_decoder_lora_wideschema.py`'s (and
    `training/decoder_lora_lib.py`'s `build_prompt_text()`'s) prompt text,
    specialized to the identity ordering (letter i always names options[i])
    since there is no training-time position-bias concern at inference: the
    model was trained on randomly-shuffled letter assignments specifically so
    it can't learn "the answer is always A" as a shortcut, which is exactly
    what makes any fixed presentation order -- including natural order, used
    here -- fine to serve with. Shared by both the text-only and vision
    multischema backends below; the image block (if any) is layered on top of
    this text by `_prompt_content()`, not baked into it, matching
    `training/decoder_lora_lib.py`'s own split between `build_prompt_text()`
    and `build_prompt_messages()`."""
    n = len(options)
    option_lines = "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))
    instructions = instructions or "Choose the option that best answers the question below."
    # PRD 5.1b implementation checklist item 2: n <= 26 stays byte-identical
    # to the pre-5.1b mechanism (protects the 8 corpus tasks that already
    # worked); n > 26 names a range instead of enumerating every code
    # (enumerating costs 32% more tokens at width 151, measured, for no
    # benefit since the options are already listed above).
    if n <= 26:
        tail = f"Answer with a single letter ({'/'.join(letters[:n])})."
    else:
        tail = f"Answer with a single option code from the list above ({letters[0]} .. {letters[n-1]})."
    return f"{instructions}\n\n{state}\n\nOptions:\n{option_lines}\n\n{tail}"


def _prompt_content(text: str, image_path: str | None):
    """The one genuine generalization a vision-capable backend needs over the
    text-only prompt shape: when `image_path` is set, chat `content` becomes
    a list with an image block prepended -- exactly the shape PRD.md 5.2b's
    probe verified end to end (`{"type": "image"}` renders to
    `<|vision_start|><|image_pad|><|vision_end|>`, and the rest of the
    rendered prompt is byte-identical to the text-only case). Mirrors
    `training/decoder_lora_lib.py`'s `build_prompt_messages()`. When
    `image_path` is None, content is the plain string every text-only backend
    already used."""
    if image_path is None:
        return text
    return [{"type": "image"}, {"type": "text", "text": text}]


def _build_code_table_for_backend(tokenizer, max_options: int, base_model_name: str) -> tuple[list[str], list[int]]:
    """Shared by DecoderMultischemaBackend and DecoderVisionMultischemaBackend
    -- PRD 5.1b's `training.decoder_lora_lib.build_code_table()`, re-run here
    rather than trusted from the manifest: refuse to silently misread a code
    position if this checkpoint's tokenizer doesn't tokenize the table the
    way training assumed. Sliced to this checkpoint's own `max_options` (an
    older checkpoint may still report 26; a checkpoint's manifest is always
    the authority on its own width, never this table's full size). Factored
    out once so neither backend below reimplements it -- this is the second
    of the two genuinely shared pieces of the restricted-logit mechanism,
    alongside `_restricted_logit_result()`."""
    from training.decoder_lora_lib import build_code_table

    try:
        codes, ids = build_code_table(tokenizer)
    except AssertionError as e:
        raise RuntimeError(
            f"code table construction failed under {base_model_name}'s tokenizer: {e} -- this "
            "checkpoint's restricted-logit mechanism assumes the table `build_code_table()` derives."
        ) from e
    if max_options > len(codes):
        raise RuntimeError(
            f"checkpoint reports max_options={max_options}, but only {len(codes)} single-token codes "
            f"exist under {base_model_name}'s tokenizer -- the checkpoint and the running tokenizer disagree."
        )
    return codes[:max_options], ids[:max_options]


def _restricted_logit_result(last_logits, letter_ids_n, options: list[str], temperature: float, torch) -> dict:
    """The one mechanism DecoderMultischemaBackend and
    DecoderVisionMultischemaBackend share verbatim, factored out so this file
    never reimplements it a third time: restrict the last token's logits to
    this item's own `n = len(options)` letter-token ids, softmax, argmax.
    PRD.md 5.2b's probe found this step needs *no* change at all when the
    forward pass comes from a vision-language model instead of a causal-LM
    one -- "the restricted-logit read still works, unchanged" is that
    section's own conclusion, and this function is what makes that true in
    code rather than just in prose (both backends call the exact same
    function on their own last-token logits). See
    DecoderMultischemaBackend's class docstring for the full derivation of
    why this is mathematically identical to the training-time
    mask-to-`-inf`-then-softmax-over-`MAX_OPTIONS` approach."""
    restricted = last_logits[letter_ids_n] / temperature
    probs_t = torch.softmax(restricted, dim=-1).cpu().numpy()
    probabilities = {opt: float(p) for opt, p in zip(options, probs_t)}
    best_idx = int(probs_t.argmax())
    return {
        "choice": options[best_idx],
        "probabilities": probabilities,
        "confidence": float(probs_t[best_idx]),
    }


class DecoderMultischemaBackend:
    """Runs a LoRA-adapter checkpoint produced by
    `training/train_decoder_lora_wideschema.py` (or its `_multischema`/
    `_benchcorpus`/`_general` (text-only) predecessors -- all save the same
    `<checkpoint_dir>/manifest.json` + `<checkpoint_dir>/adapter/` shape,
    just with a different `max_options` in the manifest) in this process, and
    answers a `choice` question the way that script's own `evaluate()` does
    at inference time:

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
         This step is `_restricted_logit_result()` above, shared with
         DecoderVisionMultischemaBackend.
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
        load_in_8bit: bool = False,
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
        self.apply_temperature = apply_temperature
        self.temperature = float(self.manifest.get("temperature", 1.0)) if apply_temperature else 1.0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        base_model_name = base_model or self.manifest["base_model"]
        self.base_model_name = base_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # PRD.md 13a.26 experiment flag (default off): QLoRA-style inference --
        # the frozen base weights load in int8 (bitsandbytes) while the LoRA
        # adapters stay at full precision; never the reverse. Off by default
        # so every existing caller gets the byte-identical fp16 path.
        self.load_in_8bit = load_in_8bit
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, quantization_config=quant_config,
                dtype=torch.bfloat16, device_map="auto",
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.to(self.device)
        self.model.eval()

        self.letters, self._letter_ids = _build_code_table_for_backend(
            self.tokenizer, self.max_options, base_model_name
        )

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "checkpoint_name": self.checkpoint_name,
            "base_model": self.base_model_name,
            "architecture": self.manifest.get("architecture"),
            "max_options": self.max_options,
            "quantization": "int8-bitsandbytes-base" if self.load_in_8bit else None,
            "temperature_applied": self.apply_temperature,
            "temperature": self.temperature if self.apply_temperature else None,
            "calibration_note": (
                "raw (uncalibrated) probabilities by default -- PRD.md 13a.2/13a.5 found temperature "
                "scaling makes this architecture's ECE/Brier worse, not better; pass "
                "apply_temperature=True / --decoder-apply-temperature to override"
            ),
            "device": self.device,
        }

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        # image_path accepted for interface parity, unused: this checkpoint
        # was trained (train_decoder_lora_wideschema.py and its text-only
        # predecessors) with AutoTokenizer/AutoModelForCausalLM, which never
        # saw an image -- see DecoderVisionMultischemaBackend for the arm
        # that does.
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
        result = _restricted_logit_result(last_logits, letter_ids_n, options, self.temperature, torch)
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return result


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

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        # image_path accepted for interface parity, unused -- see
        # DecoderVisionMultischemaMockBackend for the mock that at least
        # acknowledges an image was passed (it still can't look at it).
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


class DecoderVisionMultischemaBackend:
    """The vision-capable sibling of DecoderMultischemaBackend (PRD.md 5.2b,
    13a.11), not a rewrite of it -- deliberately structured as a sibling class
    rather than a subclass, for one concrete reason: the two classes'
    __init__ methods load genuinely different HF classes for genuinely
    different reasons (AutoTokenizer/AutoModelForCausalLM vs.
    AutoProcessor/AutoModelForImageTextToText -- PRD.md 5.2b's probe found
    the causal-LM class silently has zero vision-tower parameters under this
    base model, so there is no "just add an optional image path" shortcut
    available on top of the existing class), and forcing that into one
    __init__ with branching would be harder to read than two short, separate
    ones. What both classes share -- the restricted-logit mechanism itself --
    is factored out above into `_build_code_table_for_backend()` and
    `_restricted_logit_result()` precisely so it is not reimplemented here a
    third time. That is the concrete meaning of "sibling, not reinvented":
    the ~15 lines every decoder-family backend needs are shared functions;
    the ~30 lines that must differ (tokenizer-vs-processor, model class,
    pixel_values passthrough) are not artificially forced to look identical.

    What genuinely differs from DecoderMultischemaBackend, matching PRD.md
    5.2b's own probe findings one for one:

      1. **`AutoProcessor`, not `AutoTokenizer`.** A bare tokenizer leaves a
         single `<|image_pad|>` placeholder in the prompt and emits no
         pixels; the processor is what expands it to the real per-image
         token count and returns `pixel_values`/`image_grid_thw`. A text-only
         item (`image_path=None`) through the same processor behaves
         identically to a bare tokenizer (verified in
         `training/probe_vision_path.py`), so this class still answers a
         text-only `choice` item correctly -- it just never needs to, since
         DecoderMultischemaBackend already exists for that path and this
         class's own checkpoint was trained with vision rows in the mix
         specifically to be run against real screenshots.
      2. **`AutoModelForImageTextToText`, not `AutoModelForCausalLM`.** For
         this base model, `AutoModelForCausalLM` maps to a class with *zero*
         `visual` parameters -- it silently discards the vision tower even if
         you hand it an image. `AutoModelForImageTextToText` maps to the real
         `Qwen3_5ForConditionalGeneration` class, the one
         `training/train_decoder_lora_general.py` actually trains against.
      3. **A real PIL image is opened and passed to the processor** whenever
         `image_path` is set, using the exact chat-template content shape
         (`[{"type": "image"}, {"type": "text", "text": ...}]`) PRD.md 5.2b
         verified renders to `<|vision_start|><|image_pad|><|vision_end|>`
         with the rest of the prompt byte-identical to the text-only case --
         see `_prompt_content()` above.
      4. **The vision-tower-freeze assertion.** PRD.md 5.2b found the
         existing LoRA `target_modules` freeze the vision tower by *name
         mismatch* (the ViT blocks use `attn.qkv`/`mlp.linear_fc{1,2}`, which
         match nothing in `target_modules`) rather than by an explicit
         freeze -- worth making deliberate rather than trusting it silently.
         Re-checked here at load time, the same discipline
         `_build_code_table_for_backend()` already applies to the letter-token
         mechanism: don't trust the manifest's claim that this holds, verify
         it against the actually-loaded model.
      5. **`max_pixels`**, read from the checkpoint's manifest by default (or
         overridden by `--max-pixels`) and passed to `AutoProcessor` -- PRD.md
         5.2b: "the image processor ships effectively uncapped... this is the
         failure mode most likely to waste a night of GPU time," except here
         at *inference* time an uncapped image just costs more tokens/latency
         rather than truncating a training batch, since this backend's own
         `make_collate`-equivalent (the plain `self.processor(...)` call
         below) never truncates and there is no fixed batch `max_length` to
         violate at inference time the way there is at training time.

    Everything else -- the restricted-logit read itself, the raw-by-default
    calibration stance (13a.2/13a.5's finding was about the mechanism, not
    about whether an image was involved), the letter-token single-token
    assertion -- is identical to DecoderMultischemaBackend's, by construction
    (shared functions, not shared prose).

    Requires torch/transformers/peft/pillow (and, transitively,
    torchvision -- PRD.md 5.2b: "transformers 5.17.0 makes torchvision a hard
    dependency of any image processor") and a real vision-trained checkpoint
    on disk -- all imported lazily in __init__, same reasoning as
    DecoderMultischemaBackend's."""

    name = "decoder_vision_multischema"

    def __init__(
        self,
        checkpoint_dir: Path,
        base_model: str | None = None,
        device: str | None = None,
        apply_temperature: bool = False,
        max_pixels: int | None = None,
    ):
        manifest_path = checkpoint_dir / "manifest.json"
        adapter_dir = checkpoint_dir / "adapter"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        if not adapter_dir.exists():
            raise FileNotFoundError(f"no adapter dir at {adapter_dir}")

        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._Image = Image
        self.manifest = json.loads(manifest_path.read_text())
        self.checkpoint_name = checkpoint_dir.name
        self.max_options = int(self.manifest.get("max_options", DECODER_MULTISCHEMA_MAX_OPTIONS))
        self.apply_temperature = apply_temperature
        self.temperature = float(self.manifest.get("temperature", 1.0)) if apply_temperature else 1.0
        # PRD.md 5.2b: 256*28*28 caps any screenshot at 180 image tokens --
        # the training-time default when vision rows are present. Fall back
        # to the manifest's recorded value (train_decoder_lora_general.py
        # writes `max_pixels` into it) rather than re-guessing it, unless the
        # caller overrides explicitly.
        self.max_pixels = max_pixels or self.manifest.get("max_pixels")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        base_model_name = base_model or self.manifest["base_model"]
        self.base_model_name = base_model_name

        processor_kwargs = {"max_pixels": self.max_pixels} if self.max_pixels else {}
        self.processor = AutoProcessor.from_pretrained(adapter_dir, **processor_kwargs)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        base = AutoModelForImageTextToText.from_pretrained(base_model_name, dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.to(self.device)
        self.model.eval()

        self.letters, self._letter_ids = _build_code_table_for_backend(
            self.processor.tokenizer, self.max_options, base_model_name
        )

        # PRD.md 5.2b / training/decoder_lora_lib.assert_vision_tower_frozen():
        # the vision tower is frozen by target_modules name-mismatch, not by
        # an explicit freeze. Re-verify against the actually-loaded model
        # rather than trust the manifest's claim that the checkpoint this
        # training run produced has that property.
        lora_mods = [n for n, _ in self.model.named_modules() if n.endswith("lora_A")]
        in_vision = [n for n in lora_mods if "visual" in n]
        if in_vision:
            raise RuntimeError(
                f"{len(in_vision)} LoRA module(s) landed in the vision tower, which PRD.md 5.2b's "
                f"probe found should be structurally impossible with this project's target_modules: "
                f"{in_vision[:5]}"
            )
        self._n_lora_modules = len(lora_mods)

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "checkpoint_name": self.checkpoint_name,
            "base_model": self.base_model_name,
            "architecture": self.manifest.get("architecture"),
            "model_class": "AutoModelForImageTextToText (Qwen3_5ForConditionalGeneration)",
            "max_options": self.max_options,
            "max_pixels": self.max_pixels,
            "n_lora_modules": self._n_lora_modules,
            "vision_tower_frozen": True,  # would have raised in __init__ otherwise
            "temperature_applied": self.apply_temperature,
            "temperature": self.temperature if self.apply_temperature else None,
            "calibration_note": (
                "raw (uncalibrated) probabilities by default -- PRD.md 13a.2/13a.5 found temperature "
                "scaling makes this architecture's ECE/Brier worse, not better (finding is about the "
                "mechanism, predates vision); pass apply_temperature=True / "
                "--decoder-apply-temperature to override"
            ),
            "device": self.device,
        }

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        n = len(options)
        if not (2 <= n <= self.max_options):
            raise ValueError(
                f"DecoderVisionMultischemaBackend supports 2-{self.max_options} options (this "
                f"checkpoint's manifest max_options={self.max_options}), got {n}"
            )
        torch = self._torch
        t0 = time.perf_counter()

        prompt_text = _build_multischema_prompt(state, options, self.letters, instructions)
        content = _prompt_content(prompt_text, image_path)
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        images = None
        if image_path is not None:
            images = [self._Image.open(image_path).convert("RGB")]

        enc = self.processor(text=[prompt], images=images, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
        last_logits = out.logits[0, -1, :].float()  # unpadded single sequence -- last column is the last real token
        letter_ids_n = torch.tensor(self._letter_ids[:n], device=last_logits.device)
        result = _restricted_logit_result(last_logits, letter_ids_n, options, self.temperature, torch)
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return result


class DecoderVisionMultischemaMockBackend:
    """The vision-path wiring self-test, same non-trained-heuristic
    discipline as DecoderMultischemaMockBackend above -- and, like it, exists
    purely so benchmarks/screenspot_v2's harness wiring (filter -> call ->
    score -> evidence bundle, including real image-path resolution and
    hash-verification in the loader) can be proven end to end with no torch,
    transformers, peft, or checkpoint on disk.

    **It never looks at the image.** A keyword-overlap heuristic over
    `state`/`options` text has no mechanism to read pixels; `image_path` is
    accepted (matching the real backend's signature so the harness never
    branches on which backend it's holding) and used only to confirm the
    referenced file exists, which is enough to prove the loader handed the
    backend a real, resolvable path without pretending grounding accuracy
    can be evaluated by a heuristic that is, by construction, blind. Its
    accuracy is exactly as meaningless as DecoderMultischemaMockBackend's --
    arguably more so, since real ScreenSpot-v2 options are position/size
    descriptors with essentially no exploitable text overlap with the
    instruction, which is the point (see benchmarks/screenspot_v2/README.md)."""

    name = "decoder_vision_multischema_mock"

    def __init__(self, max_options: int = DECODER_MULTISCHEMA_MAX_OPTIONS):
        self.max_options = max_options

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "max_options": self.max_options,
            "note": "NOT a trained model -- a deterministic keyword-overlap heuristic that never "
                    "looks at the image (it has no mechanism to). Used only to prove the vision "
                    "multischema harness pipeline (filter -> call -> score -> evidence bundle, "
                    "including real image-path resolution) works end to end without torch, "
                    "transformers, peft, or a checkpoint on disk.",
        }

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        n = len(options)
        if not (2 <= n <= self.max_options):
            raise ValueError(f"DecoderVisionMultischemaMockBackend supports 2-{self.max_options} options, got {n}")
        if image_path is not None and not Path(image_path).exists():
            raise FileNotFoundError(
                f"image_path {image_path!r} does not exist -- the loader should never hand this "
                "backend an unresolved path (see benchmarks/screenspot_v2/loader.py)."
            )
        t0 = time.perf_counter()

        context = f"{instructions or ''} {state}".lower()
        context_words = set(w.strip(".,:;!?()\"'") for w in context.split())

        scores = []
        for opt in options:
            opt_words = [w.strip(".,:;!?()\"'") for w in opt.lower().split() if len(w.strip(".,:;!?()\"'")) > 2]
            overlap = sum(1 for w in opt_words if w in context_words)
            scores.append(overlap / max(len(opt_words), 1))

        best_idx = max(range(n), key=lambda i: (scores[i], -i))
        floored = [s + 0.05 for s in scores]
        z = sum(floored)
        probabilities = {opt: v / z for opt, v in zip(options, floored)}
        return {
            "choice": options[best_idx],
            "probabilities": probabilities,
            "confidence": probabilities[options[best_idx]],
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


class RoutingModelBenchmarkBackend:
    """Thin `ChoiceBackend` adapter over `serve.inference.RoutingDecoderModel`
    -- the actual class the real `/v1/systemone` server serves requests
    through (PRD.md 5.2b/13a.20/13a.22/13a.29/13a.30), including its
    text-adapter routing (`EKVACHAN_TEXT_ADAPTER`, defaulting to whichever
    checkpoint sits in the "vision" adapter slot -- `wide`/`stage3` as of
    PRD.md 13a.18-13a.19, not literally vision-only) and, if requested, its
    PRD.md 13a.29/13a.30 CUDA-graph fast path.

    Exists so JevBench/jabr-v2/ViZDoom can be run against **the actual
    production-shaped serving object**, CUDA graphs on or off, instead of
    against `DecoderMultischemaBackend`/`DecoderVisionMultischemaBackend`'s
    own separate, independent model-loading code -- those two classes
    build their own bare `AutoModelForCausalLM`/`AutoModelForImageTextToText`
    + single `PeftModel.from_pretrained()` directly and were never wired to
    `RoutingDecoderModel`'s routing or CUDA-graph work at all, so they
    cannot answer "what does the router / the CUDA-graph path actually
    score" -- only this class can.

    Written without GPU access, like PRD.md 13a.30's own `_CudaGraphRunner`
    integration -- untested against a real checkpoint. `describe()`/
    `predict_choice()`'s shapes were written to match every existing
    backend's own contract as literally as this file's other classes show
    it (see `benchmarks/common/harness.py::run_harness()`'s exact call
    site), but this has not been run.
    """

    name = "routing_decoder"

    def __init__(
        self,
        checkpoints_dir: Path | None = None,
        text_adapter: str | None = None,
        use_cuda_graphs: bool = False,
        device: str | None = None,
        max_options: int | None = None,
    ):
        from serve.inference import RoutingDecoderModel

        self._model = RoutingDecoderModel(
            checkpoints_dir=checkpoints_dir,
            device=device,
            text_adapter=text_adapter,
            use_cuda_graphs=use_cuda_graphs,
        )
        # The cap that actually governs THIS run's text-only items is
        # whichever adapter `self._model.text_adapter_choice` resolves to
        # (PRD.md 13a.22's per-adapter validation, not the global 588-wide
        # code table) -- explicit override available since a caller may
        # want to force a narrower run (e.g. against benchcorpus
        # specifically) without touching EKVACHAN_TEXT_ADAPTER.
        self.max_options = max_options or self._model._adapter_max_options.get(
            self._model.text_adapter_choice, self._model.max_options
        )

    def describe(self) -> dict:
        info = self._model.describe()
        info["backend"] = self.name
        info["max_options"] = self.max_options  # this run's cap, see __init__
        return info

    def predict_choice(
        self, state: str, options: list[str], instructions: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        import time as _time

        image_b64 = None
        if image_path is not None:
            import base64

            image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")

        t0 = _time.perf_counter()
        result = self._model.predict_choice(
            state, options, instructions=instructions, image_b64=image_b64
        )
        result["latency_ms"] = (_time.perf_counter() - t0) * 1000.0
        return result
