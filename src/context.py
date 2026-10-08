"""Conversational context: turn follow-up / vague questions into a
standalone, retrieval-friendly query.

This runs BEFORE the existing retrieval pipeline. It never answers the
question and never adds facts — it only rewrites the user's wording so the
unchanged hybrid-search -> RRF -> rerank -> generate stages can find the
right chunks.
"""
from __future__ import annotations

import re

from src.generator import FALLBACK_MODEL, GENERATE_MODEL, _client

# Follow-up pronouns only matter when there is history; these patterns detect
# broad "discovery" questions that name no concrete document topic.
DISCOVERY_PATTERNS = [
    r"\btell\s+(me|us)\s+about\b",
    r"\bwhat\s+(courses|programs|classes|batches|offers?)\b",
    r"\bdo\s+you\s+(offer|have|teach|provide|run)\b",
    r"\bwhat\s+can\s+i\s+(learn|study|do|take)\b",
    r"\babout\s+the\s+(academy|institute|college|school)\b",
    r"\bwhat\s+(services|facilities|support)\b",
    r"\bwhich\s+courses\b",
    r"\bcourse\s+(list|offered|availability)\b",
    r"\bprograms?\s+(available|offered)\b",
]

# A question that already names a document topic is self-sufficient.
SPECIFIC_MARKERS = re.compile(
    r"\b(fee|fees|cost|price|duration|timing|timings|syllabus|faculty|"
    r"refund|discount|installment|installments|admission|scholarship|"
    r"mentor|placement|lab|contact|who teaches|experience)\b",
    re.IGNORECASE,
)

# The knowledge base's own subject headings — used only to point retrieval at
# the right documents, never stated to the user as an answer.
KB_TOPICS = (
    "courses and their durations, tuition and registration fees, syllabus "
    "topics, batch timings, faculty members, admission process and required "
    "documents, refund/discount/installment policies, and facilities such as "
    "project mentoring, computer lab and placement support"
)

# Broad "what do you have?" questions name no topic, so retrieval drifts to
# whichever document happens to contain the word "academy". For those we search
# a fixed description of the corpus instead of guessing — this string is never
# shown to the user, it only drives the unchanged retrieval stages.
DISCOVERY_QUERY = (
    "CodeCraft Academy 2026 courses, durations, tuition fees, syllabus topics, "
    "batch timings, faculty, admission process and facilities"
)

# Vocabulary of a purely vague enquiry: function words plus generic asking/naming
# words. Any token outside this set means the user named a real subject, and that
# subject must survive untouched for the retrieval and no-answer stages.
GENERIC_WORDS = {
    # function words
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "at", "with",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "i", "we", "you", "your", "my", "me", "us", "it", "its", "this", "that",
    "there", "here", "what", "whats", "which", "who", "how", "any", "all",
    "much", "many", "about", "please", "just", "so",
    # vague enquiry words
    "tell", "say", "share", "give", "show", "know", "learn", "study", "take",
    "offer", "offers", "offered", "provide", "teach", "run", "have", "get",
    "explain", "describe", "overview", "general", "everything", "something",
    "info", "information", "details", "detail", "mean", "means",
    # topic-category words (not a specific course or technology)
    "course", "courses", "class", "classes", "program", "programs", "programme",
    "programmes", "batch", "batches", "option", "options", "type", "types",
    "kind", "kinds", "list", "curriculum", "topic", "topics", "subject",
    "subjects", "service", "services", "facility", "facilities", "support",
    "academy", "academies", "codecraft", "institute", "institution", "college",
    "school", "centre", "center", "coaching", "training",
    "available", "different", "various",
}

CONDENSE_PROMPT = f"""You prepare search queries for a chatbot that answers questions about
CodeCraft Academy, an AI & coding coaching centre, from its 2026 documents.
Those documents cover: {KB_TOPICS}.

Rewrite the user's latest message into ONE standalone search query.

Rules:
- Resolve pronouns and follow-ups ("it", "that course", "how much is AI and ML")
  using the conversation history, so the query makes sense on its own.
- If the question is vague or asks for an overview, make it concrete by naming
  the relevant document topics it is really asking about.
- Keep any course name, number or wording the user already gave.
- Do NOT answer the question. Do NOT add facts, prices, dates or course names
  that the user or history did not mention.
- If the question is already clear and self-contained, return it unchanged.

Conversation history (most recent last):
{{history}}

Latest user message: {{question}}

Return only the rewritten search query on a single line.
"""


def is_discovery(question: str) -> bool:
    """A broad overview question that names no concrete document topic.

    Questions like "Do you offer a Java course?" match the patterns but DO name a
    subject, and rewriting them would erase that subject before the no-answer
    guard could see it, so they must be left alone.
    """
    if SPECIFIC_MARKERS.search(question):
        return False
    lowered = question.lower()
    if not any(re.search(p, lowered) for p in DISCOVERY_PATTERNS):
        return False
    tokens = re.findall(r"[a-z0-9]+", lowered)
    return all(t in GENERIC_WORDS for t in tokens)


def needs_rewrite(question: str, history: list[dict] | None) -> bool:
    """Only rewrite when the question genuinely needs help.

    Follow-ups with history always do; a first turn does only when it is a
    broad discovery question that names no concrete document topic.
    """
    return bool(history) or is_discovery(question)


def _format_history(history: list[dict], max_turns: int = 3) -> str:
    if not history:
        return "(none)"
    recent = history[-max_turns * 2 :]
    return "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:400]}"
        for m in recent
    )


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """Return a standalone query for retrieval; original text on any failure."""
    if not needs_rewrite(question, history):
        return question
    if not history and is_discovery(question):
        # deterministic: no LLM call, no guessing about wording
        return DISCOVERY_QUERY
    prompt = CONDENSE_PROMPT.format(
        history=_format_history(history), question=question
    )
    try:
        client = _client()
        for model in (GENERATE_MODEL, FALLBACK_MODEL):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                break
            except Exception:
                resp = None
        text = (resp.text or "").strip() if resp else ""
    except Exception:
        return question  # never let context-rewriting break the chatbot
    rewritten = text.splitlines()[0].strip() if text else ""
    rewritten = re.sub(r"^['\"`\s]+|['\"`\s]+$", "", rewritten)
    rewritten = re.sub(r"^(query|rewritten query)\s*[:\-]\s*", "", rewritten, flags=re.I)
    return rewritten if 3 < len(rewritten) < 300 else question
