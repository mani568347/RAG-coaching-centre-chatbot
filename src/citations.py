"""Citation formatting.

Page numbers and filenames are taken ONLY from chunk metadata captured during
ingestion (real Docling/pdfplumber provenance). Nothing here invents pages.
"""
from __future__ import annotations

import re

NO_ANSWER_MESSAGE = (
    "I couldn't find information about that in the provided CodeCraft Academy "
    "documents."
)

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "your", "are", "was",
    "will", "can", "you", "our", "its", "into", "course", "courses", "classes",
    "student", "students", "also", "all", "any", "them", "they", "than", "then",
    "have", "has", "not", "but", "out", "about", "which", "what", "when", "who",
    "how", "much", "many", "does", "did", "is", "of", "to", "in", "on",
    "at", "by", "as", "or", "a", "an", "it", "isn", "don", "please",
}

_TOKEN = re.compile(r"[a-z0-9₹]+(?:\.\d+)?")


def _is_no_answer(answer: str) -> bool:
    stripped = answer.strip().upper()
    return stripped.startswith("NOT_FOUND") or stripped == NO_ANSWER_MESSAGE.upper()


def _body(text: str) -> str:
    """Chunk text without its contextual-retrieval prefix line."""
    head, sep, rest = text.partition("\n\n")
    return rest if sep and head.lower().startswith("context:") else text


def _tokens(text: str) -> set[str]:
    out = set()
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.strip(".,'\"")
        if len(tok) > 2 and tok not in STOPWORDS:
            out.add(tok)
    return out


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,.]*", text))


def evidence_chunks(answer: str, top_chunks: list[dict]) -> list[dict]:
    """Deterministic fallback: chunks whose own wording the answer reuses.

    Used only when the model did not return a valid `[used: n]` line, so the
    citation set is still derived from retrieved evidence rather than from the
    whole Top 5.
    """
    if not top_chunks:
        return []
    ans_tokens = _tokens(answer)
    ans_numbers = _numbers(answer)
    scored = []
    for chunk in top_chunks:
        body = _body(chunk["text"])
        tokens = _tokens(body)
        numbers = _numbers(body)
        lexical = len(ans_tokens & tokens) / max(1, len(ans_tokens))
        numeric = len(ans_numbers & numbers) / max(1, len(ans_numbers))
        scored.append((lexical + numeric, chunk))
    best = max(scored, key=lambda sc: sc[0])[0]
    if best <= 0.0:
        return top_chunks[:1]  # nothing overlaps: cite only the top candidate
    keep = [chunk for score, chunk in scored if score >= max(0.15, best * 0.5)]
    return keep


def used_chunks(
    top_chunks: list[dict], used_indices: list[int], answer: str = ""
) -> list[dict]:
    """Map the model's [used: n] references back to chunks.

    Invalid or missing references fall back to evidence-matched chunks, never to
    the full Top 5 (which would over-attribute sources).
    """
    if not top_chunks:
        return []
    picked = []
    for n in dict.fromkeys(used_indices):  # dedupe, keep order
        if 1 <= n <= len(top_chunks):
            picked.append(top_chunks[n - 1])
    if picked:
        return picked
    return evidence_chunks(answer, top_chunks)


def citation_block(
    top_chunks: list[dict], used_indices: list[int], answer: str = ""
) -> str:
    """Build the 'Source(s): <file> — Page <n>' block from real metadata."""
    pairs: list[tuple[str, int]] = []
    for chunk in used_chunks(top_chunks, used_indices, answer):
        meta = chunk["metadata"]
        pair = (meta["source"], int(meta["page"]))
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        return ""
    header = "Source:" if len(pairs) == 1 else "Sources:"
    lines = [header]
    for source, page in pairs:
        lines.append(f"{source} — Page {page}")
    return "\n".join(lines)


def final_answer(answer: str, top_chunks: list[dict], used_indices: list[int]) -> str:
    """Compose the user-facing answer with citations (or the no-answer text)."""
    if not top_chunks or _is_no_answer(answer):
        return NO_ANSWER_MESSAGE
    citations = citation_block(top_chunks, used_indices, answer)
    if not citations:
        return NO_ANSWER_MESSAGE
    return f"{answer}\n\n{citations}"
