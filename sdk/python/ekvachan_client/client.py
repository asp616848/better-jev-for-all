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

    Mirrors `serve.server.Question` exactly: `type` is one of "choice",
    "score", "noul" per the wire contract (PRD.md Section 1.2/6.1).

    Today's Phase 1 reference server only returns a real answer for
    `type="choice"` with `options == ["entailment", "neutral",
    "contradiction"]`; every other combination (any `score`/`noul`
    question, or a `choice` with different options) currently raises
    HTTP 501. See the package docstring and PRD.md Section 10a / 13a.3.
    """

    type: str
    instructions: Optional[str] = None
    options: Optional[List[str]] = None
    levels: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.instructions is not None:
            d["instructions"] = self.instructions
        if self.options is not None:
            d["options"] = self.options
        if self.levels is not None:
            d["levels"] = self.levels
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
      - 501: primitive/option-set not implemented by the current checkpoint
        (the expected outcome for anything but `choice` with
        options == ["entailment", "neutral", "contradiction"] today).
      - 422: malformed request (e.g. a `choice` question missing `options`).
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

    Only this one question shape (`choice` over exactly
    `["entailment", "neutral", "contradiction"]`) will succeed against
    today's Phase 1 checkpoint; anything else raises EkVachanAPIError with
    status_code == 501. See the package docstring for why.
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
        key: str = "choice",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper for the single most common case today: one
        `choice` question. Returns just that question's result dict, e.g.
        {"choice": "entailment", "probabilities": {...}, "confidence": 0.98}.

        Only options == ["entailment", "neutral", "contradiction"] will
        succeed against the Phase 1 checkpoint; any other option list
        raises EkVachanAPIError(status_code=501).
        """
        response = self.system_one(
            state=state,
            questions={key: Question(type="choice", instructions=instructions, options=options)},
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
