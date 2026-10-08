"""Shared Gemini client construction.

Every Gemini call in this project goes through `gemini_client()`, which pins a
hard per-request HTTP timeout and disables the SDK's own retries so the
call-site loops are the only retry mechanism (bounded, never infinite).
"""
from __future__ import annotations

import os

import dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import PROJECT_ROOT, RETRY_BACKOFF_CAP_S

dotenv.load_dotenv(PROJECT_ROOT / ".env")

# Free-tier throttling, model overload and transport timeouts are all retryable
# by waiting; anything else (bad key, bad model name, 400) is not.
TRANSIENT_WORDS = (
    "high demand",
    "quota",
    "rate",
    "deadline",
    "unavailable",
    "timeout",
    "timed out",
    "try again",
    "busy",
    "connection",
    "reset",
)


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy it into the .env file in the "
            "project root (see .env.example) and try again."
        )
    return key


def gemini_client(timeout_ms: int) -> genai.Client:
    return genai.Client(
        api_key=api_key(),
        http_options=types.HttpOptions(
            timeout=int(timeout_ms),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def is_transient(exc: Exception) -> bool:
    # A read timeout can arrive with an empty message, so classify transport
    # failures by type rather than by wording.
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if type(exc).__module__.split(".")[0] == "httpx":
        return True
    if isinstance(exc, genai_errors.APIError):
        if exc.code in (408, 429, 500, 502, 503, 504):
            return True
        message = str(exc.message).lower()
    else:
        message = str(exc).lower()
    return any(word in message for word in TRANSIENT_WORDS)


def backoff(attempt: int) -> float:
    """Linear backoff with a cap, so total waiting time stays bounded."""
    return min(5.0 * (attempt + 1), RETRY_BACKOFF_CAP_S)
