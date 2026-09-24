from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_TIMEOUT_S = 30.0


@dataclass
class Question:
    """One typed question in a /v1/systemone request.

    Mirrors `serve.server.Question`: `type` is one of "choice", "score",
    "noul" per the wire contract (PRD.md Section 1.2/6.1). As of 2026-09-24
    (PRD.md 5.2b / 14 Q4) all three primitives return a real answer against
    the server's default routing model, any 2-26 option/level count (PRD
    13a.5's ceiling) -- not just one fixed schema.

    `image`: optional base64-encoded image (a bare base64 string, or a
    `data:<mime>;base64,...` data URI -- both accepted). Valid on any
    primitive; a request that sets it is routed server-side to the
    vision-capable adapter and returns HTTP 501 if that checkpoint hasn't
    finished training yet, or HTTP 422 if the image doesn't decode. See
    `serve/server.py`'s `Question.image` docstring for the exact transport
    and size constraints.

    One thing still genuinely unresolved server-side, not this SDK's
    concern to paper over: a plain *text* request's adapter choice is an
    explicit, open server config value (`EKVACHAN_TEXT_ADAPTER`) pending a
    regression check -- an unconfigured server returns HTTP 500 for any
    text-only request until that's set. See `serve/inference.py`'s module
    docstring.
    """

    type: str
    instructions: Optional[str] = None
    options: Optional[List[str]] = None
    levels: Optional[List[str]] = None
    image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.instructions is not None:
            d["instructions"] = self.instructions
        if self.options is not None:
            d["options"] = self.options
        if self.levels is not None:
            d["levels"] = self.levels
        if self.image is not None:
            d["image"] = self.image
        return d


@dataclass
class SystemOneResponse:
    """Parsed response from POST /v1/systemone.

    Attributes:
        model: the model name that answered (server's `model.model_name`).
        results: dict keyed by question key. For a `choice` question that
            succeeded, the value looks like
            {"choice": str, "probabilities": {label: float, ...}, "confidence": float}.
        usage: e.g. {"latency_ms": float}.
        raw: the full, unparsed decoded JSON body, in case a future server
            version adds fields this client doesn't model yet.
    """

    model: str
    results: Dict[str, Any]
    usage: Dict[str, Any]
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SystemOneResponse":
        return cls(
            model=data.get("model", ""),
            results=data.get("results", {}),
            usage=data.get("usage", {}),
            raw=data,
        )


class EkVachanAPIError(Exception):
    """Raised when the server returns a non-2xx response.

    `status_code` is commonly:
      - 501: option/level count out of the 2-26 range, or an image-bearing
        request before the vision checkpoint has finished training.
      - 422: malformed request (e.g. a `choice` question missing `options`,
        a `score` question missing `levels`, or an `image` field that
        doesn't decode).
      - 500: the server's text-adapter routing config isn't set yet (see
        `Question`'s docstring) — a server misconfiguration, not a bad
        request.
    """

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"ekVachan API error {status_code}: {message}")


class EkVachanClient:
    """Minimal client for a self-hosted ekVachan server's /v1/systemone endpoint.

    No third-party dependencies — built on `urllib` from the standard
    library only, per the SDK's dependency-light design goal.

    Example:
        >>> from ekvachan_client import EkVachanClient, Question
        >>> client = EkVachanClient("http://localhost:8000")
        >>> response = client.system_one(
        ...     state=(
        ...         "Premise: The cat sat on the mat.\\n"
        ...         "Hypothesis: An animal was on the mat."
        ...     ),
        ...     questions={
        ...         "nli": Question(
        ...             type="choice",
        ...             options=["entailment", "neutral", "contradiction"],
        ...         ),
        ...     },
        ... )
        >>> response.results["nli"]["choice"]
        'entailment'

    As of 2026-09-24 (PRD.md 5.2b / 14 Q4) the default server model answers
    `choice` (any 2-26 options), `noul` (bare `P(Yes)` float), and `score`
    (probability-weighted level position) — not just one fixed schema. See
    `Question`'s own docstring for the two things that can still legitimately
    fail: an unconfigured server's text-adapter routing (HTTP 500), and an
    image-bearing request before the vision checkpoint has landed (HTTP 501).
    """

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def system_one(
        self,
        state: str,
        questions: Dict[str, Question],
        model: Optional[str] = None,
    ) -> SystemOneResponse:
        """POST /v1/systemone.

        `questions` maps an arbitrary caller-chosen key to a Question,
        mirroring `serve.server.SystemOneRequest` (`state`, `model`,
        `questions`) exactly.

        Raises EkVachanAPIError on any non-2xx response — most commonly a
        501 for `score`/`noul` or a `choice` option set the checkpoint
        wasn't trained on (see the module docstring).
        """
        payload: Dict[str, Any] = {
            "state": state,
            "questions": {key: q.to_dict() for key, q in questions.items()},
        }
        if model is not None:
            payload["model"] = model

        return self._post("/v1/systemone", payload)

    def choice(
        self,
        state: str,
        options: List[str],
        instructions: Optional[str] = None,
        image: Optional[str] = None,
        key: str = "choice",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper for a single `choice` question. Returns just
        that question's result dict, e.g. {"choice": "entailment",
        "probabilities": {...}, "confidence": 0.98}. `options` may be any
        2-26 item list (PRD.md 13a.5's ceiling) against today's default
        server model — no longer restricted to one fixed schema. Pass
        `image` (base64, see `Question`'s docstring) to route this question
        to the vision-capable adapter.
        """
        response = self.system_one(
            state=state,
            questions={
                key: Question(type="choice", instructions=instructions, options=options, image=image)
            },
            model=model,
        )
        return response.results[key]

    def noul(
        self,
        state: str,
        instructions: Optional[str] = None,
        image: Optional[str] = None,
        key: str = "noul",
        model: Optional[str] = None,
    ) -> float:
        """Convenience wrapper for a single `noul` (yes/no) question. Returns
        the bare `P(Yes)` float per PRD.md Section 1.2's wire contract — not
        a dict, matching the server's actual response shape for this
        primitive exactly.
        """
        response = self.system_one(
            state=state,
            questions={key: Question(type="noul", instructions=instructions, image=image)},
            model=model,
        )
        return response.results[key]

    def score(
        self,
        state: str,
        levels: List[str],
        instructions: Optional[str] = None,
        image: Optional[str] = None,
        key: str = "score",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper for a single `score` question. `levels` is
        sent, and answered, in exactly the order given — the server never
        shuffles or sorts it (an ordinal `score` scale's semantics depend on
        that order; see `serve/inference.py`'s `RoutingDecoderModel.
        predict_score` docstring for why). Returns
        {"score": float, "probabilities": {...}, "confidence": float} where
        `score` is the probability-weighted level position, not the argmax
        index — it can land between levels.
        """
        response = self.system_one(
            state=state,
            questions={
                key: Question(type="score", instructions=instructions, levels=levels, image=image)
            },
            model=model,
        )
        return response.results[key]

    def health(self) -> Dict[str, Any]:
        """GET /health."""
        url = f"{self.base_url}/health"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise EkVachanAPIError(e.code, e.read().decode("utf-8", errors="replace")) from e

    def _post(self, path: str, payload: Dict[str, Any]) -> SystemOneResponse:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise EkVachanAPIError(e.code, str(detail)) from e

        return SystemOneResponse.from_json(data)
