"""Gemini Flash grounded answer generation."""
from __future__ import annotations

import re
import time

from src.config import (
    GENERATE_MODEL,
    GENERATE_TIMEOUT_MS,
    GENERATE_TOTAL_BUDGET_S,
    GEMINI_ATTEMPTS_PER_MODEL,
)
from src.llm_client import backoff, gemini_client, is_transient

FALLBACK_MODEL = "gemini-flash-lite-latest"
MODELS = (GENERATE_MODEL, FALLBACK_MODEL)

USED_MARKER = re.compile(r"\[used:\s*([\d,\s]+)\]", re.IGNORECASE)

SYSTEM_PROMPT = """You are the CodeCraft Academy assistant. Answer ONLY from the
retrieved context blocks provided below. The context blocks are the single source
of truth.

Rules:
- Use only the supplied retrieved context. Do not use outside knowledge.
- Do not guess. Do not hallucinate. Do not invent fees, timings, faculty,
  courses, or policies.
- Preserve exact numerical values (fees, percentages, durations, times, dates)
  exactly as written in the context.
- If the context does not contain sufficient information to answer, reply with
  exactly: NOT_FOUND
- For a broad or overview question, summarise the relevant facts that ARE
  present in the context (for example the list of courses, their fees,
  timings, faculty or facilities) instead of replying NOT_FOUND. Only reply
  NOT_FOUND when the context holds no facts related to the question at all.
- Every answer MUST end with one final line listing the context block numbers
  your answer is actually drawn from, formatted exactly as: [used: 1, 3]
  List only blocks you really relied on, and never list a block whose text does
  not appear in your answer. This line is mandatory on every non-NOT_FOUND
  answer.
- Never state a filename, page number or source name yourself: citations are
  built from the context metadata, so inventing one is an error.
- Keep the answer concise and factual.
"""


def _client():
    return gemini_client(GENERATE_TIMEOUT_MS)


def build_context(top_chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(top_chunks, start=1):
        meta = c["metadata"]
        blocks.append(
            f"[Context {i}] (source: {meta['source']}, page: {meta['page']}, "
            f"section: {meta['section']})\n{c['text']}"
        )
    return "\n\n".join(blocks)


NO_ANSWER_MESSAGE = (
    "I couldn't find information about that in the provided CodeCraft Academy "
    "documents."
)


BUSY_MESSAGE = (
    "Sorry — the answer service is busy or slow to respond (Gemini rate limit "
    "or timeout). Please try again in a minute."
)


def _fatal_message(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    detail = str(getattr(exc, "message", exc)).strip().splitlines()
    return (
        "Sorry, Gemini could not answer this request"
        + (f" (HTTP {code})" if code else "")
        + f": {detail[0][:200] if detail else 'unknown error'}"
    )


def _call(prompt: str) -> str:
    """Grounded generation with bounded retries and a hard wall-clock budget."""
    client = _client()
    deadline = time.monotonic() + GENERATE_TOTAL_BUDGET_S
    last_err: Exception | None = None
    for attempt in range(GEMINI_ATTEMPTS_PER_MODEL):
        for model in MODELS:
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                text = (resp.text or "").strip()
                if text:
                    return text
                last_err = RuntimeError("Gemini returned an empty response.")
            except Exception as exc:  # APIError, OSError, httpx transport error
                if not is_transient(exc):
                    raise RuntimeError(_fatal_message(exc)) from exc
                last_err = exc
        wait = backoff(attempt)
        if attempt + 1 >= GEMINI_ATTEMPTS_PER_MODEL or time.monotonic() + wait >= deadline:
            break
        time.sleep(wait)
    raise RuntimeError(BUSY_MESSAGE) from last_err


def parse_used(text: str) -> tuple[str, list[int]]:
    """Strip the mandatory [used: n] line and return (answer, indices)."""
    match = USED_MARKER.search(text)
    cleaned = USED_MARKER.sub("", text).strip()
    if not match:
        return cleaned, []
    return cleaned, [int(n) for n in re.findall(r"\d+", match.group(1))]


def generate_answer(
    question: str, top_chunks: list[dict], resolved: str | None = None
) -> tuple[str, list[int]]:
    """Return (answer_text, used_context_indices). used is empty for NOT_FOUND."""
    if not top_chunks:
        return NO_ANSWER_MESSAGE, []
    context = build_context(top_chunks)
    asked = question if not resolved or resolved == question else (
        f"{question}\n(Understood as: {resolved})"
    )
    prompt = (
        f"{SYSTEM_PROMPT}\n\nRetrieved context:\n{context}\n\n"
        f"User question: {asked}\n\nAnswer:"
    )
    text, used = parse_used(_call(prompt))

    if text.upper().startswith("NOT_FOUND") or len(text) < 3:
        return NO_ANSWER_MESSAGE, []
    return text, used
