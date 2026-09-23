"""
Execution-verified tests for ekvachan_client against the REAL serve/server.py
FastAPI app (not a hand-rolled stand-in) — imported and run in-process with a
mocked EncoderChoiceModel, so no GPU or trained checkpoint is required.

`serve/inference.py` imports `torch` and `transformers` at module load time.
Those are heavy training-time dependencies this SDK's own test environment
doesn't need, so before importing `serve.server` we install minimal fake
`torch`/`transformers` modules into `sys.modules`. This only affects import
time — the actual predict_choice() call in these tests comes from FakeModel
below, never from real torch code — so it doesn't touch or need the real
inference numerics, only exercises the real FastAPI request/response wiring
that this client talks to.

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
    """Stands in for the server's real model (DecoderChoiceModel by default
    since 2026-09-23, EncoderChoiceModel before that) — same public contract
    (`model_name`, `predict_choice(state, options, instructions=None)`), no
    real weights or torch involved. Still exercises the fixed 3-way schema
    for simplicity; the real server accepts 2-26 options now, but that's a
    property of the real model, not something this HTTP-wiring test needs to
    cover."""

    model_name = "ekvachan-base-fake"

    def predict_choice(self, state: str, options: list, instructions: str | None = None) -> dict:
        if list(options) != LABELS:
            raise ChoiceUnsupportedError(
                f"this checkpoint only supports options=={LABELS}; got {options}"
            )
        return {
            "choice": "entailment",
            "probabilities": {"entailment": 0.97, "neutral": 0.02, "contradiction": 0.01},
            "confidence": 0.97,
        }


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


def test_score_primitive_returns_501(live_server):
    """`score` has no trained model at all yet — server returns 501."""
    client = EkVachanClient(live_server)
    with pytest.raises(EkVachanAPIError) as exc_info:
        client.system_one(
            state="a support ticket",
            questions={"urgency": Question(type="score", levels=["low", "medium", "high"])},
        )
    assert exc_info.value.status_code == 501


def test_noul_primitive_returns_501(live_server):
    client = EkVachanClient(live_server)
    with pytest.raises(EkVachanAPIError) as exc_info:
        client.system_one(
            state="a support ticket",
            questions={"is_spam": Question(type="noul")},
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
