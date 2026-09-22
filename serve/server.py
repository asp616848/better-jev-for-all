"""
POST /v1/systemone — Phase 1 reference server, matching Jev's request shape
(PRD.md Section 1.2 / 6.1) as closely as we could verify from secondary
sources (no live Jev API access to diff against byte-for-byte — see the note
in Section 1.2's sourcing). Response field names (`results`, `usage`) are our
best-effort reconstruction, not independently confirmed against a real Jev
response body; treat this as "same request shape, best-effort response
shape" rather than a verified byte-for-byte contract.

Only the `choice` primitive is implemented, and only for the one schema the
Phase 1 checkpoint was actually trained on (see serve/inference.py). `score`
and `noul` return a 501 rather than a fabricated answer.
"""

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from serve.inference import ChoiceUnsupportedError, EncoderChoiceModel, load_default

app = FastAPI(title="ekvachan (Phase 1 reference server)")
_model: EncoderChoiceModel | None = None


def get_model() -> EncoderChoiceModel:
    global _model
    if _model is None:
        _model = load_default()
    return _model


class Question(BaseModel):
    type: str
    instructions: str | None = None
    options: list[str] | None = None
    levels: list[str] | None = None


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
        if q.type == "choice":
            if q.options is None:
                raise HTTPException(422, f"question '{key}': choice requires options")
            try:
                results[key] = model.predict_choice(req.state, q.options)
            except ChoiceUnsupportedError as e:
                raise HTTPException(501, str(e))
        else:
            raise HTTPException(501, f"question '{key}': primitive '{q.type}' not implemented in Phase 1")

    return {
        "model": model.model_name,
        "results": results,
        "usage": {"latency_ms": round((time.time() - t0) * 1000, 2)},
    }


@app.get("/health")
def health():
    get_model()
    return {"status": "ok"}
