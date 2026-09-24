"""
Execution-verified tests for ekvachan_client against the REAL serve/server.py
FastAPI app (not a hand-rolled stand-in) — imported and run in-process with a
mocked model (FakeModel, standing in for RoutingDecoderModel), so no GPU or
trained checkpoint is required.

`serve/inference.py` imports `torch` and `transformers` at module load time.
Those are heavy training-time dependencies this SDK's own test environment
doesn't need, so before importing `serve.server` we install minimal fake
`torch`/`transformers` modules into `sys.modules`. This only affects import
time — the actual predict_choice()/predict_noul()/predict_score() calls in
these tests come from FakeModel below, never from real torch code — so it
doesn't touch or need the real inference numerics, only exercises the real
FastAPI request/response wiring that this client talks to. `RoutingDecoderModel`
itself stays safely importable here too: its heavy imports (peft, PIL,
safetensors, AutoModelForImageTextToText/AutoProcessor) are all deferred to
inside `__init__`/its methods, never executed just by defining the class —
same lazy-import discipline `serve/inference.py`'s module docstring
describes for every class in that file.

Run with:
    uv run --with fastapi --with uvicorn --with pytest \
        pytest sdk/python/tests/test_client.py
(or any environment with fastapi + uvicorn + pytest installed; torch/
transformers are NOT required, see above).
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SDK_PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SDK_PYTHON_ROOT))


def _install_fake_torch_stack() -> None:
    """Stub just enough of torch/transformers for `serve.inference` to
    import cleanly, without requiring those (large, GPU-oriented) packages
    to be installed in the SDK's own test environment."""
    if "torch" in sys.modules:
        return  # real torch is installed; nothing to fake

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: (lambda f: f)  # decorator no-op
    fake_torch.Tensor = object

    def _cuda_is_available() -> bool:
        return False

    fake_torch.cuda = types.SimpleNamespace(is_available=_cuda_is_available)
    sys.modules["torch"] = fake_torch

    fake_nn = types.ModuleType("torch.nn")
    fake_functional = types.ModuleType("torch.nn.functional")
    fake_nn.functional = fake_functional
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.nn.functional"] = fake_functional

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForSequenceClassification = object
    fake_transformers.AutoTokenizer = object
    sys.modules["transformers"] = fake_transformers


_install_fake_torch_stack()

import uvicorn  # noqa: E402

import serve.server as server_module  # noqa: E402
from serve.inference import LABELS, ChoiceUnsupportedError  # noqa: E402

from ekvachan_client import EkVachanAPIError, EkVachanClient, Question  # noqa: E402


class FakeModel:
    """Stands in for the server's real model (RoutingDecoderModel by default
    since 2026-09-24, DecoderChoiceModel/EncoderChoiceModel before that) —
    same public contract (`model_name`, `max_options`, `predict_choice`,
    `predict_noul`, `predict_score`, all taking `image_b64=None`), no real
    weights or torch involved. `predict_choice` still exercises the fixed
    3-way schema for simplicity; the real server accepts 2-26 options now,
    but that's a property of the real model, not something this
    HTTP-wiring test needs to cover.

    `predict_score`'s stand-in is deliberately order-dependent (all
    probability mass on whichever level is LAST in the caller's `levels`
    list) specifically so a test can prove the server never reorders
    `levels` before it reaches the model — the real "never shuffled"
    contract this primitive exists to enforce (see serve/inference.py's
    RoutingDecoderModel.predict_score docstring)."""

    model_name = "ekvachan-base-fake"
    max_options = 26

    def predict_choice(
        self, state: str, options: list, instructions: str | None = None, image_b64: str | None = None
    ) -> dict:
        if image_b64 is not None:
            raise ChoiceUnsupportedError("FakeModel has no vision-capable adapter loaded")
        if list(options) != LABELS:
            raise ChoiceUnsupportedError(
                f"this checkpoint only supports options=={LABELS}; got {options}"
            )
        return {
            "choice": "entailment",
            "probabilities": {"entailment": 0.97, "neutral": 0.02, "contradiction": 0.01},
            "confidence": 0.97,
        }

    def predict_noul(
        self, state: str, instructions: str | None = None, image_b64: str | None = None
    ) -> float:
        if image_b64 is not None:
            raise ChoiceUnsupportedError("FakeModel has no vision-capable adapter loaded")
        return 0.83  # deterministic stand-in P(Yes), exercises the bare-float wire shape

    def predict_score(
        self,
        state: str,
        levels: list,
        instructions: str | None = None,
        image_b64: str | None = None,
    ) -> dict:
        if image_b64 is not None:
            raise ChoiceUnsupportedError("FakeModel has no vision-capable adapter loaded")
        n = len(levels)
        probabilities = {level: (1.0 if i == n - 1 else 0.0) for i, level in enumerate(levels)}
        return {"score": float(n - 1), "probabilities": probabilities, "confidence": 1.0}


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def live_server():
    """Runs the real serve.server:app in a background thread with a fake
    model injected, so EkVachanClient can be tested over real HTTP against
    the real FastAPI request/response contract."""
    server_module._model = FakeModel()
    port = _free_port()
    config = uvicorn.Config(server_module.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("test server did not start in time")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


def test_health(live_server):
    client = EkVachanClient(live_server)
    assert client.health() == {"status": "ok"}


def test_choice_with_trained_schema_succeeds(live_server):
    client = EkVachanClient(live_server)
    result = client.choice(
        state=(
            "Premise: The cat sat on the mat.\n"
            "Hypothesis: An animal was on the mat."
        ),
        options=["entailment", "neutral", "contradiction"],
    )
    assert result["choice"] == "entailment"
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == {"entailment", "neutral", "contradiction"}


def test_choice_with_unsupported_options_returns_501(live_server):
    """Matches the real, current server behavior: an arbitrary options list
    (e.g. the ["billing", "technical", "other"] example from PRD.md Section
    1.2) is rejected, not silently mapped onto the trained head."""
    client = EkVachanClient(live_server)
    with pytest.raises(EkVachanAPIError) as exc_info:
        client.choice(state="a support ticket", options=["billing", "technical", "other"])
    assert exc_info.value.status_code == 501


def test_noul_returns_bare_float(live_server):
    """`noul` now answers for real (PRD 5.2b/14 Q4): the result is a bare
    float (P(Yes)), not a dict — matching Jev's real wire contract exactly
    (PRD.md Section 1.2's table)."""
    client = EkVachanClient(live_server)
    response = client.system_one(
        state="a support ticket",
        questions={"is_spam": Question(type="noul", instructions="Is this spam?")},
    )
    result = response.results["is_spam"]
    assert isinstance(result, float)
    assert result == pytest.approx(0.83)


def test_noul_convenience_method(live_server):
    client = EkVachanClient(live_server)
    assert client.noul(state="a support ticket") == pytest.approx(0.83)


def test_score_returns_weighted_position_dict(live_server):
    """`score` now answers for real: a dict with a probability-weighted
    `score` (not the argmax index), `probabilities` keyed by the exact level
    strings given, and `confidence`."""
    client = EkVachanClient(live_server)
    response = client.system_one(
        state="a support ticket",
        questions={"urgency": Question(type="score", levels=["low", "medium", "high"])},
    )
    result = response.results["urgency"]
    assert result["score"] == pytest.approx(2.0)  # FakeModel puts all mass on the last level
    assert set(result["probabilities"]) == {"low", "medium", "high"}
    assert result["confidence"] == pytest.approx(1.0)


def test_score_levels_are_never_shuffled(live_server):
    """The load-bearing invariant this primitive exists to enforce: `levels`
    reach the model in the exact order the caller gave, never sorted or
    reshuffled anywhere on the request path. FakeModel's stand-in puts all
    probability mass on whichever level is LAST in the list it actually
    received, so two requests differing only in level order must produce
    different results if -- and only if -- the order genuinely passed
    through unchanged."""
    client = EkVachanClient(live_server)
    ascending = client.score(state="x", levels=["low", "medium", "high"])
    descending = client.score(state="x", levels=["high", "medium", "low"])
    assert ascending["probabilities"] == {"low": 0.0, "medium": 0.0, "high": 1.0}
    assert descending["probabilities"] == {"high": 0.0, "medium": 0.0, "low": 1.0}
    assert ascending != descending


def test_score_missing_levels_returns_422(live_server):
    client = EkVachanClient(live_server)
    with pytest.raises(EkVachanAPIError) as exc_info:
        client.system_one(
            state="a support ticket",
            questions={"urgency": Question(type="score")},
        )
    assert exc_info.value.status_code == 422


def test_choice_with_image_and_no_vision_adapter_returns_501(live_server):
    """A request that carries an image must be routed to the vision
    adapter — FakeModel simulates "no vision adapter loaded" (the real
    server's actual current state, since checkpoints/ekvachan-decoder-qwen-vision
    hasn't finished training yet), so this should 501, not silently answer
    text-only."""
    client = EkVachanClient(live_server)
    with pytest.raises(EkVachanAPIError) as exc_info:
        client.choice(
            state="a screenshot",
            options=["entailment", "neutral", "contradiction"],
            image="aGVsbG8=",  # arbitrary valid base64; FakeModel rejects on presence alone
        )
    assert exc_info.value.status_code == 501


def test_choice_missing_options_returns_422(live_server):
    client = EkVachanClient(live_server)
    with pytest.raises(EkVachanAPIError) as exc_info:
        client.system_one(
            state="a support ticket",
            questions={"category": Question(type="choice")},
        )
    assert exc_info.value.status_code == 422


def test_multiple_questions_in_one_request(live_server):
    """Response should carry one result per question key, plus model +
    usage, per the SystemOneResponse shape in serve/server.py."""
    client = EkVachanClient(live_server)
    response = client.system_one(
        state="Premise: The cat sat on the mat.\nHypothesis: An animal was on the mat.",
        questions={
            "nli_a": Question(type="choice", options=LABELS),
            "nli_b": Question(type="choice", options=LABELS),
        },
    )
    assert response.model == "ekvachan-base-fake"
    assert set(response.results) == {"nli_a", "nli_b"}
    assert "latency_ms" in response.usage
