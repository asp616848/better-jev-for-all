"""
Reference inference wrappers around the trained checkpoints (PRD.md Section 7
targets a Rust/ONNX server long-term; this is the Python reference
implementation used to validate the wire contract first).

Three model classes live here, on purpose, not because one replaced the last:

- `EncoderChoiceModel` (`ekvachan-base` encoder) — the ORIGINAL Phase 1
  checkpoint. Fixed to exactly one `choice` schema
  (["entailment","neutral","contradiction"]) — the classifier head never sees
  `options` text at train time, so it cannot answer an arbitrary options list.
  Kept as-is and NOT wired into the live server by default anymore (see
  below), because `benchmarks/common/backends.py`'s `InProcessBackend` still
  targets it directly via `load_default()` for the original fixed-schema
  JevBench/jabr-v2 comparison runs (PRD 13a.1/13a.4) — changing what
  `load_default()` returns would silently break that.

- `DecoderChoiceModel` (Qwen3.5-4B LoRA, restricted-logit read, text-only) —
  the server's `choice`-only, text-only model class from 2026-09-23 (PRD.md
  14 Q4). Thin wrapper around `benchmarks.common.backends.
  DecoderMultischemaBackend`. Kept as-is, still importable and still correct,
  but no longer what `serve/server.py` defaults to (see `RoutingDecoderModel`
  below) — retained because nothing forces its removal and it's a smaller,
  single-adapter surface some callers may still want directly.

- `RoutingDecoderModel` (added 2026-09-24, PRD.md 5.2b / 14 Q4) — what the
  live server actually serves now. Answers all three wire-contract primitives
  (`choice`, `noul`, `score`) with 2-26 options, text or image, from **one**
  base model (`AutoModelForImageTextToText`) with **hot-swappable named LoRA
  adapters** (PEFT's `set_adapter()`), not two separate loaded model copies —
  see its own docstring for the full design and the one open routing
  question (which adapter answers a plain-text request) that this file
  deliberately does not resolve on its own.
"""

import base64
import io
import json
import os
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Both of these are cheap, dataset/prompt-shape-only imports -- no torch/
# transformers/peft is triggered by importing benchmarks.common.backends or
# benchmarks.common.schema at module level (see their own docstrings: every
# heavy import in backends.py is deferred to inside a backend's __init__).
# Reused here rather than reimplemented, per this project's standing "don't
# build the restricted-logit mechanism a third time" discipline.
from benchmarks.common.backends import (
    _assert_single_token_letters,
    _build_multischema_prompt,
    _restricted_logit_result,
)
from benchmarks.common.schema import DECODER_MULTISCHEMA_MAX_OPTIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = ["entailment", "neutral", "contradiction"]


class ChoiceUnsupportedError(ValueError):
    """Raised when a request asks for a choice schema this checkpoint wasn't trained on."""


class ImageDecodeError(ValueError):
    """Raised when a request's `image` field can't be decoded as a real image
    within this server's stated constraints (see `RoutingDecoderModel.
    _decode_image`'s docstring for the exact transport/size/format rules)."""


class AdapterNotConfiguredError(RuntimeError):
    """Raised when a text-only request needs a named LoRA adapter that isn't
    loaded -- specifically the "pending regression check" placeholder state
    described in `RoutingDecoderModel`'s docstring. This is a server
    configuration gap, not a bad client request (a 5xx, not a 4xx, at the
    serve/server.py layer)."""


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
    # Points at the most recent validated checkpoint. ekvachan-decoder-qwen-benchcorpus (PRD 13a.10)
    # trained on the real better-jev-bench corpus: noul 95.48% (up from ekvachan-decoder-qwen-primitives's
    # 88.80%), score 80.95% (up from 75.40%), and the CLINC150 zero-shot regression check held/improved
    # (96.77% vs 95.73%) -- strictly the better default for choice/noul, even though serve/ doesn't
    # answer noul/score yet.
    return DecoderChoiceModel(REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-benchcorpus")


# --- Text-only adapter routing (PRD.md 5.2b / 14 Q4; STATUS.md's "vision
# tier" section) -----------------------------------------------------------
#
# Which named LoRA adapter answers a request that carries no image is an
# explicit, OPEN configuration question, not something this file gets to
# guess at. PRD.md 5.2b's own pre-committed gate -- CLINC150 zero-shot >=
# 95.5% and text `choice` on 13a.10's common sources within 1pp, measured
# against checkpoints/ekvachan-decoder-qwen-vision once it finishes training
# -- decides whether the vision-mixed adapter is safe to use for text too
# (one adapter for everything) or whether ekvachan permanently ships two
# adapters (text keeps using the benchcorpus adapter).
#
# RESOLVED 2026-09-24, PRD.md 13a.11/13a.14, real numbers, not a guess:
# checkpoints/ekvachan-decoder-qwen-vision's own eval_id run against
# 13a.10's exact held-out text sources came back BETTER, not just "not
# worse" -- aggregate text `choice` accuracy 83.03% -> 90.71% (+7.68pp),
# CLINC150 zero-shot 96.47% (comfortably clears the pre-committed >=95.5%
# bar). A few small-n sources (WaNLI n=30, SNLI-test n=44) dipped a few
# points, consistent with sampling noise on a slice that size, not a
# regression -- the aggregate move is an order of magnitude larger and in
# the opposite direction. Decision: ship ONE adapter for everything, text
# and vision alike. The sentinel/two-adapter machinery below is kept as
# real, tested infrastructure (not deleted) in case a future retrain ever
# does regress and this decision needs to be revisited -- EKVACHAN_TEXT_ADAPTER
# still overrides this default if that day comes.
TEXT_ADAPTER_PENDING_SENTINEL = "vision"


def _ensure_vlclass_adapter_dir(checkpoint_dir: Path) -> Path:
    """Return a directory holding a VL-class-compatible LoRA adapter for
    `checkpoint_dir`, remapping its saved parameter keys if needed.

    Every decoder checkpoint before PRD.md 13a.11 (`ekvachan-decoder-qwen-
    benchcorpus` included) was trained via `AutoModelForCausalLM`
    (`Qwen3_5ForCausalLM`), whose module tree names the decoder layers
    `...model.model.layers.N...`. `RoutingDecoderModel`'s one shared base
    model is `AutoModelForImageTextToText` (`Qwen3_5ForConditionalGeneration`),
    which nests the same layers one level deeper:
    `...model.model.language_model.layers.N...`. A LoRA adapter's saved keys
    must match its target model's real module names, so a causal-LM-native
    adapter needs that one path segment inserted before it can be attached
    as a named adapter on the VL-class base at all.

    This is a standalone re-implementation of the remap
    `training/vision_class_swap_control.py` already proved neutral end to
    end (PRD.md 5.2b step 0: eval accuracy/Brier/ECE reproduced to the last
    decimal after this exact transformation) -- re-implemented here rather
    than imported, since that script is one of this project's frozen
    training-lineage files and this is a serve-time, not train-time, need.

    Idempotent and cheap on repeat calls: if a checkpoint's adapter already
    has VL-native keys (true for any future checkpoint trained directly via
    `train_decoder_lora_general.py`, e.g. the vision adapter), this returns
    the original adapter directory untouched -- no remap, no extra file. If
    a remap is needed, the result is cached at
    `<checkpoint_dir>-vlclass-remap/adapter/` so the safetensors rewrite
    happens at most once per checkpoint. For `ekvachan-decoder-qwen-
    benchcorpus` specifically, PRD.md 13a.11's own control run already
    produced and validated exactly this cache directory -- this function
    finds and reuses it rather than redoing the work.
    """
    from safetensors import safe_open

    adapter_dir = checkpoint_dir / "adapter"
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    old_prefix = "base_model.model.model."
    new_prefix = "base_model.model.model.language_model."

    with safe_open(str(safetensors_path), framework="pt") as f:
        keys = list(f.keys())
    needs_remap = any(
        k.startswith(old_prefix) and not k.startswith(new_prefix) and ".layers." in k for k in keys
    )
    if not needs_remap:
        return adapter_dir

    remapped_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}-vlclass-remap" / "adapter"
    remapped_safetensors = remapped_dir / "adapter_model.safetensors"
    if remapped_safetensors.exists():
        return remapped_dir

    from safetensors.torch import load_file, save_file

    state_dict = load_file(str(safetensors_path))
    remapped = {}
    n_remapped = 0
    for k, v in state_dict.items():
        if k.startswith(old_prefix) and not k.startswith(new_prefix) and ".layers." in k:
            k = new_prefix + k[len(old_prefix):]
            n_remapped += 1
        remapped[k] = v
    if n_remapped == 0:
        raise RuntimeError(
            f"no keys matched the VL-class remap pattern in {safetensors_path} -- adapter shape unexpected"
        )

    remapped_dir.mkdir(parents=True, exist_ok=True)
    save_file(remapped, str(remapped_safetensors))
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    (remapped_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return remapped_dir


DEFAULT_NOUL_INSTRUCTIONS = (
    "Answer the yes/no question below about the given state. "
    "Respond Yes if the proposition is true, No if it is false."
)


class RoutingDecoderModel:
    """One base model (`AutoModelForImageTextToText`), hot-swappable named
    LoRA adapters (PEFT's `set_adapter()`), all three wire-contract
    primitives. PRD.md 5.2b / 14 Q4.

    **Why one base model, not two.** PRD.md 5.2b measured that
    `AutoModelForImageTextToText` costs a text-only request nothing beyond
    333.5M frozen vision-tower params resident in memory (0.67 GB bf16) --
    a text-only row never emits `pixel_values` and never enters the vision
    tower. A LoRA adapter is ~21.2M trainable params (~85 MB). So loading
    the 4.8B shared backbone twice to save that 0.67 GB would be strictly
    worse than loading it once and switching a small adapter -- which is
    exactly what `peft.PeftModel.load_adapter()` / `set_adapter()` are for.

    **Adapters loaded, by name:**
      - `"benchcorpus"` (`checkpoints/ekvachan-decoder-qwen-benchcorpus`,
        PRD.md 13a.10) -- the real, already-validated, text-only checkpoint.
        Its keys are causal-LM-native and go through
        `_ensure_vlclass_adapter_dir`'s remap before loading (see that
        function's docstring); this is the one genuinely new piece of
        engineering this class needed beyond "call load_adapter twice".
      - `"vision"` (`checkpoints/ekvachan-decoder-qwen-vision`, PRD.md
        13a.11) -- the mixed text+image checkpoint, loaded automatically if
        `<checkpoints_dir>/ekvachan-decoder-qwen-vision/manifest.json`
        exists (i.e. training has actually finished producing it). If it
        doesn't exist yet, `self.vision_available` is False and any
        image-bearing request raises a clear `ChoiceUnsupportedError`
        instead of silently answering text-only.

    **Routing rule, enforced, not assumed:**
      - Any request carrying an image MUST use `"vision"` -- the only
        adapter with a loaded, unfrozen-by-training-data vision tower
        underneath it (the base model's vision tower itself is shared and
        frozen for both adapters; only the "vision" adapter's LoRA weights
        were ever trained on rows that used it). This is not configurable.
      - A text-only request's adapter is `self.text_adapter_choice`, an
        explicit, documented, externally-set config value -- see
        `TEXT_ADAPTER_PENDING_SENTINEL`'s module-level docstring just above
        this class for the full reasoning. This file does not guess it.

    **Concurrency note.** `peft.PeftModel.set_adapter()` mutates shared
    model state (which adapter's weights are active for the next forward
    pass). FastAPI runs a sync endpoint like `serve/server.py`'s
    `systemone()` in a thread pool, so two concurrent requests choosing
    different adapters could otherwise interleave a `set_adapter()` call
    from one request with the forward pass of another. `self._lock` (held
    for the full set-adapter-then-forward critical section of
    `predict_choice`) serializes inference to prevent that -- an accepted,
    documented limitation of this Python reference server (PRD.md 7.1 already
    frames this file as "the reference implementation used to validate the
    wire contract first", not the target low-latency Rust server), not a
    silent gap.
    """

    TEXT_ADAPTER_NAME = "benchcorpus"
    VISION_ADAPTER_NAME = "vision"
    TEXT_CHECKPOINT_NAME = "ekvachan-decoder-qwen-benchcorpus"
    VISION_CHECKPOINT_NAME = "ekvachan-decoder-qwen-vision"
    MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MiB decoded; see _decode_image's docstring

    def __init__(
        self,
        checkpoints_dir: Path | None = None,
        device: str | None = None,
        text_adapter: str | None = None,
        load_vision: bool = True,
    ):
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        checkpoints_dir = checkpoints_dir or (REPO_ROOT / "checkpoints")
        text_checkpoint_dir = checkpoints_dir / self.TEXT_CHECKPOINT_NAME
        vision_checkpoint_dir = checkpoints_dir / self.VISION_CHECKPOINT_NAME

        text_manifest_path = text_checkpoint_dir / "manifest.json"
        if not text_manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest at {text_manifest_path} -- RoutingDecoderModel requires the "
                "known-stable text checkpoint to exist even if only serving vision requests, "
                "since it also supplies the base_model name and the shared letter-token check."
            )
        text_manifest = json.loads(text_manifest_path.read_text())
        self.base_model_name = text_manifest["base_model"]
        self.max_options = DECODER_MULTISCHEMA_MAX_OPTIONS

        # Raw (uncalibrated) by default -- PRD.md 13a.2/13a.5/13a.9 all found
        # temperature scaling makes this architecture's ECE/Brier worse, not
        # better, and 13a.9 confirmed T=1.0 is already the real optimum. Same
        # stance DecoderMultischemaBackend/DecoderVisionMultischemaBackend take.
        self.temperature = 1.0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._lock = threading.Lock()

        self.processor = AutoProcessor.from_pretrained(self.base_model_name)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.letters = [chr(ord("A") + i) for i in range(self.max_options)]
        self._letter_ids = _assert_single_token_letters(
            self.processor.tokenizer, self.letters, self.base_model_name
        )

        base = AutoModelForImageTextToText.from_pretrained(self.base_model_name, dtype=torch.bfloat16)

        text_adapter_dir = _ensure_vlclass_adapter_dir(text_checkpoint_dir)
        self.model = PeftModel.from_pretrained(base, str(text_adapter_dir), adapter_name=self.TEXT_ADAPTER_NAME)
        self._loaded_adapters = {self.TEXT_ADAPTER_NAME}

        self.vision_available = False
        if load_vision and (vision_checkpoint_dir / "manifest.json").exists():
            vision_adapter_dir = _ensure_vlclass_adapter_dir(vision_checkpoint_dir)
            self.model.load_adapter(str(vision_adapter_dir), adapter_name=self.VISION_ADAPTER_NAME)
            self._loaded_adapters.add(self.VISION_ADAPTER_NAME)
            self.vision_available = True

        self.model.set_adapter(self.TEXT_ADAPTER_NAME)
        self.model.to(self.device)
        self.model.eval()

        # PRD.md 5.2b's vision-tower-freeze property, re-verified against the
        # actually-loaded model rather than trusted from either manifest --
        # same discipline DecoderVisionMultischemaBackend already applies.
        lora_mods = [n for n, _ in self.model.named_modules() if n.endswith("lora_A")]
        in_vision = [n for n in lora_mods if "visual" in n]
        if in_vision:
            raise RuntimeError(
                f"{len(in_vision)} LoRA module(s) landed in the vision tower, which PRD.md 5.2b's "
                f"probe found should be structurally impossible with this project's target_modules: "
                f"{in_vision[:5]}"
            )
        self._n_lora_modules = len(lora_mods)

        self.model_name = f"{self.base_model_name}+routed-lora"
        self.text_adapter_choice = text_adapter or os.environ.get(
            "EKVACHAN_TEXT_ADAPTER", TEXT_ADAPTER_PENDING_SENTINEL
        )

    def describe(self) -> dict:
        return {
            "base_model": self.base_model_name,
            "model_class": "AutoModelForImageTextToText (Qwen3_5ForConditionalGeneration)",
            "loaded_adapters": sorted(self._loaded_adapters),
            "vision_available": self.vision_available,
            "text_adapter_choice": self.text_adapter_choice,
            "text_adapter_choice_is_configured": self.text_adapter_choice in self._loaded_adapters,
            "max_options": self.max_options,
            "n_lora_modules": self._n_lora_modules,
            "device": self.device,
        }

    def _select_adapter(self, has_image: bool) -> str:
        if has_image:
            if not self.vision_available:
                raise ChoiceUnsupportedError(
                    "this request carries an image, but no vision-capable adapter is loaded "
                    f"(checkpoints/{self.VISION_CHECKPOINT_NAME}/manifest.json does not exist yet -- "
                    "PRD.md 13a.11's vision training run may still be in progress). Retry once that "
                    "checkpoint lands."
                )
            return self.VISION_ADAPTER_NAME

        choice = self.text_adapter_choice
        if choice not in self._loaded_adapters:
            raise AdapterNotConfiguredError(
                f"text_adapter={choice!r} is not a loaded adapter (loaded: {sorted(self._loaded_adapters)}). "
                "This is the expected state until the coordinating session sets EKVACHAN_TEXT_ADAPTER "
                "once PRD.md 5.2b's text-vs-vision regression check has actually run -- see STATUS.md's "
                f"'vision tier' section. Set EKVACHAN_TEXT_ADAPTER={self.TEXT_ADAPTER_NAME!r} (or pass "
                "text_adapter= explicitly) to use the known-stable text checkpoint today without "
                "changing the shipped default."
            )
        return choice

    def _decode_image(self, image_b64: str):
        """Decode a request's `image` field.

        Transport: base64-encoded raw image bytes (PNG/JPEG/WEBP/BMP/etc --
        whatever Pillow can open), optionally prefixed with a
        `data:<mime>;base64,` URI scheme (stripped if present, so both a
        bare base64 string and a browser-style data URI work). Chosen over
        e.g. multipart/form-data because /v1/systemone is one JSON POST with
        arbitrarily many questions per request (PRD.md 1.2) -- base64-in-JSON
        keeps every question's payload, image or not, in the same body and
        the same content type.

        Constraints enforced here, at the wire boundary, before any image
        token cost is paid: decoded size <= MAX_IMAGE_BYTES (8 MiB -- see the
        class constant; generous versus a 1080p screenshot, PRD.md 5.2b's own
        worked example, while still bounding request-body memory), and the
        bytes must actually decode as an image Pillow recognizes. Both
        failures raise `ImageDecodeError` (a `ValueError`), which
        `serve/server.py` maps to HTTP 422 -- a malformed request, not a
        capability gap.
        """
        from PIL import Image, UnidentifiedImageError

        raw = image_b64
        if raw.startswith("data:") and ";base64," in raw:
            raw = raw.split(";base64,", 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (base64.binascii.Error, ValueError) as e:
            raise ImageDecodeError(f"image field is not valid base64: {e}") from e

        if len(data) > self.MAX_IMAGE_BYTES:
            raise ImageDecodeError(
                f"decoded image is {len(data)} bytes, over the {self.MAX_IMAGE_BYTES}-byte cap "
                "(PRD.md 5.2b: an uncapped screenshot is the failure mode most likely to waste "
                "compute -- enforced here, at the wire boundary, before any image-token cost is paid)."
            )
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except UnidentifiedImageError as e:
            raise ImageDecodeError(f"image field could not be decoded as an image: {e}") from e
        return img.convert("RGB")

    def predict_choice(
        self,
        state: str,
        options: list[str],
        instructions: str | None = None,
        image_b64: str | None = None,
    ) -> dict:
        n = len(options)
        if not (2 <= n <= self.max_options):
            raise ChoiceUnsupportedError(
                f"this checkpoint supports 2-{self.max_options} options (PRD.md 13a.5's real "
                f"ceiling for the single-uppercase-letter restricted-logit mechanism), got {n}."
            )
        torch = self._torch
        image = self._decode_image(image_b64) if image_b64 is not None else None

        prompt_text = _build_multischema_prompt(state, options, self.letters, instructions)
        # `_prompt_content` only checks "is this argument None", matching its
        # own use in DecoderVisionMultischemaBackend where the argument is a
        # real file path -- here there is no path (the image arrived as
        # in-memory bytes, already decoded above), so this sentinel just
        # trips the "an image is present" branch without implying a path.
        content = [{"type": "image"}, {"type": "text", "text": prompt_text}] if image is not None else prompt_text
        messages = [{"role": "user", "content": content}]

        with self._lock:
            adapter_name = self._select_adapter(image is not None)
            self.model.set_adapter(adapter_name)

            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            images = [image] if image is not None else None
            enc = self.processor(text=[prompt], images=images, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.model(**enc)
            last_logits = out.logits[0, -1, :].float()

        letter_ids_n = torch.tensor(self._letter_ids[:n], device=last_logits.device)
        result = _restricted_logit_result(last_logits, letter_ids_n, options, self.temperature, torch)
        result["adapter"] = adapter_name
        return result

    def predict_noul(
        self, state: str, instructions: str | None = None, image_b64: str | None = None
    ) -> float:
        """`noul`: a yes/no proposition, answered as the internal 2-way
        `choice("Yes", "No")` this mechanism already handles (PRD.md 1.2/6.2
        deliberately does not get a separate serving path). Returns a bare
        `float` -- `P(Yes)` -- matching Jev's real wire contract exactly
        (Section 1.2's table: `noul` -> `float 0-1`, not a dict)."""
        result = self.predict_choice(
            state,
            ["Yes", "No"],
            instructions=instructions or DEFAULT_NOUL_INSTRUCTIONS,
            image_b64=image_b64,
        )
        return result["probabilities"]["Yes"]

    def predict_score(
        self,
        state: str,
        levels: list[str],
        instructions: str | None = None,
        image_b64: str | None = None,
    ) -> dict:
        """`score`: an N-way `choice` over `levels`, presented and answered
        in **exactly the order given** -- never shuffled or sorted anywhere
        in this call. This matters because, unlike a generic `choice` item
        (whose training data randomizes letter assignment specifically so
        position carries no signal -- see `_build_multischema_prompt`'s
        docstring), the ordinal-`score` training data
        (`training/build_primitives_slice.py` / `build_benchcorpus_slice.py`)
        keeps its levels in fixed ascending order on purpose, and reorders
        would break the ordinal semantics the model actually learned.

        Returns `{"score", "probabilities", "confidence"}` per PRD.md 1.2's
        table -- `score` is the probability-weighted position
        `sum(i * P(levels[i]) for i in range(n))`, not the argmax index, so
        it can genuinely "land between levels" the way the PRD's own wording
        promises.
        """
        n = len(levels)
        if not (2 <= n <= self.max_options):
            raise ChoiceUnsupportedError(
                f"score supports 2-{self.max_options} levels (same restricted-logit ceiling as "
                f"choice, PRD.md 13a.5), got {n}."
            )
        result = self.predict_choice(state, levels, instructions=instructions, image_b64=image_b64)
        weighted = sum(i * result["probabilities"][level] for i, level in enumerate(levels))
        return {
            "score": weighted,
            "probabilities": result["probabilities"],
            "confidence": result["confidence"],
        }


def load_default_router(text_adapter: str | None = None) -> RoutingDecoderModel:
    """Constructs the routing layer described in `RoutingDecoderModel`'s own
    docstring: one base model, hot-swappable named LoRA adapters. Loads the
    vision adapter automatically if `checkpoints/ekvachan-decoder-qwen-vision`
    exists; otherwise serves text-only, with any image-bearing request
    raising a clear `ChoiceUnsupportedError` until that checkpoint lands.
    `text_adapter`, if given, overrides `EKVACHAN_TEXT_ADAPTER` -- used by
    this repo's own validation scripts to exercise the known-stable
    `"benchcorpus"` adapter without touching the shipped (unset) default."""
    return RoutingDecoderModel(text_adapter=text_adapter)


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
