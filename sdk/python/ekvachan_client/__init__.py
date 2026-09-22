"""
ekvachan_client — a minimal Python client for ekVachan's /v1/systemone endpoint.

Mirrors the request/response shape implemented by serve/server.py (see
PRD.md Section 6.1/6.3 for the target contract this wire shape follows).
Zero third-party runtime dependencies: uses only the Python standard
library (urllib).

HONEST LIMITATION (Phase 1 reference server, as of 2026-09-22):
Only the `choice` primitive returns a real answer today, and only for the
exact option set ["entailment", "neutral", "contradiction"] the checkpoint
was trained on. Any other `choice` options, or any `score`/`noul`
question, gets HTTP 501 from the server. This client sends the full
choice/score/noul wire shape (that's the target contract the server's
Pydantic models already accept), but don't expect anything outside that
one schema to succeed against the current checkpoint. See PRD.md
Section 10a / 13a.3.
"""

from .client import EkVachanAPIError, EkVachanClient, Question, SystemOneResponse

__all__ = [
    "EkVachanAPIError",
    "EkVachanClient",
    "Question",
    "SystemOneResponse",
]

__version__ = "0.1.0"
