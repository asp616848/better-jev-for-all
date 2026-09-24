"""
POST /v1/systemone — reference server, matching Jev's request shape
(PRD.md Section 1.2 / 6.1) as closely as we could verify from secondary
sources (no live Jev API access to diff against byte-for-byte — see the note
in Section 1.2's sourcing). Response field names (`results`, `usage`) are our
best-effort reconstruction, not independently confirmed against a real Jev
response body; treat this as "same request shape, best-effort response
shape" rather than a verified byte-for-byte contract.

Updated 2026-09-24 (PRD.md 5.2b / 14 Q4): serves `serve.inference.
RoutingDecoderModel` by default — one base model, hot-swappable named LoRA
adapters, all three primitives now answered for real:

  - `choice` — any 2-588 option request, text or image (PRD 5.1b's measured ceiling).
  - `noul` — internally a 2-way `choice("Yes","No")`; returns a bare `float`
    (`P(Yes)`), matching Jev's real wire contract (Section 1.2's table), not
    a dict.
  - `score` — an N-way `choice` built from `levels`, presented in the exact
    order given (never shuffled — see `RoutingDecoderModel.predict_score`'s
    docstring for why that specifically matters for this primitive); returns
    `{score, probabilities, confidence}` where `score` is the
    probability-weighted level position, not the argmax index.

`Question.image` (added this pass) carries an optional base64-encoded image,
consumed by all three primitives. Any request that carries one is routed to
the vision-capable adapter — the only one with a loaded vision tower — and
501s with a clear message if that checkpoint hasn't landed yet.

One thing this file deliberately does NOT decide: which named adapter
answers a plain TEXT request. That is `RoutingDecoderModel`'s own open
config value (`EKVACHAN_TEXT_ADAPTER`), pending PRD.md 5.2b's text-vs-vision
regression check — see `serve/inference.py`'s module docstring. Until that
env var is set, every text-only request 500s with an actionable message
naming exactly what to set, rather than silently guessing an adapter.
"""

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serve.inference import (
    AdapterNotConfiguredError,
    ChoiceUnsupportedError,
    ImageDecodeError,
    RoutingDecoderModel,
    load_default_router,
)

app = FastAPI(title="ekvachan reference server")
_model: RoutingDecoderModel | None = None


def get_model() -> RoutingDecoderModel:
    global _model
    if _model is None:
        _model = load_default_router()
    return _model


class Question(BaseModel):
    type: str
    instructions: str | None = None
    options: list[str] | None = None
    levels: list[str] | None = None
    # Optional image, base64-encoded (a bare base64 string, or a
    # `data:<mime>;base64,...` URI — both accepted, see
    # `serve.inference.RoutingDecoderModel._decode_image`'s docstring for the
    # exact transport, the 8 MiB decoded-size cap, and why base64-in-JSON was
    # chosen over multipart). Applies to `choice`, `noul`, and `score` alike
    # — routing to the vision adapter is the same mechanism regardless of
    # which primitive is asking, so there was no reason to restrict it to
    # `choice` only. Any request that sets this MUST be answered by the
    # vision-capable adapter (enforced in RoutingDecoderModel, not just
    # assumed here) and 501s if that checkpoint isn't loaded yet.
    image: str | None = Field(default=None, repr=False)


class SystemOneRequest(BaseModel):
    state: str
    model: str | None = None
    questions: dict[str, Question]


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    model = get_model()
    t0 = time.time()
    results = {}
    for key, q in req.questions.items():
        try:
            if q.type == "choice":
                if q.options is None:
                    raise HTTPException(422, f"question '{key}': choice requires options")
                if not (2 <= len(q.options) <= model.max_options):
                    raise HTTPException(
                        422,
                        f"question '{key}': choice supports 2-{model.max_options} options, "
                        f"got {len(q.options)}",
                    )
                results[key] = model.predict_choice(
                    req.state, q.options, instructions=q.instructions, image_b64=q.image
                )
            elif q.type == "noul":
                results[key] = model.predict_noul(
                    req.state, instructions=q.instructions, image_b64=q.image
                )
            elif q.type == "score":
                if q.levels is None:
                    raise HTTPException(422, f"question '{key}': score requires levels")
                if not (2 <= len(q.levels) <= model.max_options):
                    raise HTTPException(
                        422,
                        f"question '{key}': score supports 2-{model.max_options} levels, "
                        f"got {len(q.levels)}",
                    )
                results[key] = model.predict_score(
                    req.state, q.levels, instructions=q.instructions, image_b64=q.image
                )
            else:
                raise HTTPException(501, f"question '{key}': primitive '{q.type}' not implemented")
        except ChoiceUnsupportedError as e:
            raise HTTPException(501, str(e))
        except ImageDecodeError as e:
            raise HTTPException(422, str(e))
        except AdapterNotConfiguredError as e:
            raise HTTPException(500, str(e))

    return {
        "model": model.model_name,
        "results": results,
        "usage": {"latency_ms": round((time.time() - t0) * 1000, 2)},
    }


@app.get("/health")
def health():
    get_model()
    return {"status": "ok"}
