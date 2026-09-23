"""
POST /v1/systemone — reference server, matching Jev's request shape
(PRD.md Section 1.2 / 6.1) as closely as we could verify from secondary
sources (no live Jev API access to diff against byte-for-byte — see the note
in Section 1.2's sourcing). Response field names (`results`, `usage`) are our
best-effort reconstruction, not independently confirmed against a real Jev
response body; treat this as "same request shape, best-effort response
shape" rather than a verified byte-for-byte contract.

Serves the DECODER checkpoint by default (`ekvachan-decoder-qwen-wideschema`
via `serve.inference.load_default_decoder`), not the original encoder --
updated 2026-09-23 to match PRD.md 14 Q4's decision that the decoder is the
primary architecture. `choice` now answers any 2-26 option request (PRD
13a.5's real ceiling), not one fixed schema. `score` and `noul` still return
501 -- training data for both exists now (`training/build_primitives_slice.py`),
but the wire contract doesn't yet know how to present either primitive's
output shape (a float for `noul`, a probability-weighted scale position for
`score`); see STATUS.md.
"""

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from serve.inference import ChoiceUnsupportedError, DecoderChoiceModel, load_default_decoder

app = FastAPI(title="ekvachan reference server")
_model: DecoderChoiceModel | None = None


def get_model() -> DecoderChoiceModel:
    global _model
    if _model is None:
        _model = load_default_decoder()
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
                results[key] = model.predict_choice(req.state, q.options, instructions=q.instructions)
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
