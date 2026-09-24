"""
ekvachan_client — a minimal Python client for ekVachan's /v1/systemone endpoint.

Mirrors the request/response shape implemented by serve/server.py (see
PRD.md Section 6.1/6.3 for the target contract this wire shape follows).
Zero third-party runtime dependencies: uses only the Python standard
library (urllib).

STATUS (updated 2026-09-24, PRD.md 5.2b / 14 Q4): all three primitives
(`choice`, `noul`, `score`) return a real answer against the server's
default routing model, any 2-26 option/level count (PRD 13a.5's ceiling),
text or image. Two things can still legitimately fail, both by design, not
by omission:

- A plain-text request against an unconfigured server returns HTTP 500 —
  which of two named LoRA adapters answers a text-only request is an
  explicit open config value (`EKVACHAN_TEXT_ADAPTER`) pending a regression
  check (see `serve/inference.py`'s module docstring); this SDK does not
  paper over that with a guessed default either.
- An image-bearing request before `checkpoints/ekvachan-decoder-qwen-vision`
  has finished training returns HTTP 501.

See PRD.md Section 10a / 13a.3 / 13a.10 / 13a.11 for the underlying
checkpoint history.
"""

from .client import EkVachanAPIError, EkVachanClient, Question, SystemOneResponse

__all__ = [
    "EkVachanAPIError",
    "EkVachanClient",
    "Question",
    "SystemOneResponse",
]

__version__ = "0.1.0"
