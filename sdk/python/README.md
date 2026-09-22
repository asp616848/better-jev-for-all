# ekvachan-client (Python)

A minimal Python client for ekVachan's `POST /v1/systemone` endpoint
(`serve/server.py`). Zero third-party runtime dependencies — built on
`urllib` from the standard library only.

## Honest limitation, right up front

The Phase 1 reference server (`serve/server.py` + `serve/inference.py`)
only returns a real answer for **one** question shape:

```python
Question(type="choice", options=["entailment", "neutral", "contradiction"])
```

Any other `choice` options list, or any `score`/`noul` question, currently
gets **HTTP 501** from the server — there's no trained model for those yet
(PRD.md Section 10a / 13a.3). This client supports the full
`choice`/`score`/`noul` wire shape (that's the target contract the
server's Pydantic models already accept), but the example below that
would 501 today is labeled as such rather than presented as if it works.

## Install

This is not yet published to PyPI. Use it locally:

```bash
pip install -e sdk/python
# or, dependency-free (it's stdlib-only, so this also works):
# just add sdk/python to your PYTHONPATH / copy ekvachan_client/ into your project
```

## Usage

### 1. A request that works today (`choice` over the trained schema)

```python
from ekvachan_client import EkVachanClient

client = EkVachanClient("http://localhost:8000")

result = client.choice(
    state=(
        "Premise: The cat sat on the mat.\n"
        "Hypothesis: An animal was on the mat."
    ),
    options=["entailment", "neutral", "contradiction"],
)
print(result)
# {'choice': 'entailment', 'probabilities': {'entailment': 0.98, ...}, 'confidence': 0.98}
```

### 2. The full request shape, including a question that will 501 today

```python
from ekvachan_client import EkVachanClient, EkVachanAPIError, Question

client = EkVachanClient("http://localhost:8000")

try:
    response = client.system_one(
        state="A customer emails asking why their invoice doubled this month.",
        questions={
            # This shape works against the Phase 1 checkpoint:
            "nli": Question(
                type="choice",
                options=["entailment", "neutral", "contradiction"],
            ),
            # This shape mirrors PRD.md Section 1.2's own example request,
            # but WILL raise EkVachanAPIError(status_code=501) today --
            # no model is trained for an arbitrary option set like this one.
            "category": Question(
                type="choice",
                instructions="Classify the customer's issue.",
                options=["billing", "technical", "other"],
            ),
        },
    )
except EkVachanAPIError as e:
    print(f"call failed: {e.status_code} {e.message}")
    # -> call failed: 501 question 'category': this checkpoint only supports
    #    options==['entailment', 'neutral', 'contradiction'] (trained schema); ...
```

### 3. Health check

```python
client.health()  # {"status": "ok"}
```

## Running the tests

The test suite (`tests/test_client.py`) runs the **real** `serve/server.py`
FastAPI app in-process (via `uvicorn` in a background thread) with a fake
`EncoderChoiceModel` injected, then drives it with the real
`EkVachanClient` over actual HTTP on localhost. It stubs `torch` /
`transformers` in `sys.modules` before import so it doesn't require those
(large, GPU-oriented) packages just to exercise the HTTP contract — no GPU
or trained checkpoint needed.

```bash
pip install -e "sdk/python[test]"
pytest sdk/python/tests/test_client.py -v
```

All 7 tests pass as of this writing, covering: health check, a successful
`choice` call, `choice` with an unsupported option set (501), `score`
(501), `noul` (501), a malformed `choice` missing `options` (422), and a
multi-question request.
