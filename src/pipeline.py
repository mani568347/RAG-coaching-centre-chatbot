"""End-to-end query pipeline: (optional context rewrite) -> hybrid retrieval
-> RRF -> rerank -> Gemini.

The retrieval architecture itself is unchanged; `rewrite_query` only turns a
follow-up or vague question into a standalone search query before Stage 1.
"""
from __future__ import annotations

from src.citations import final_answer
from src.config import RERANK_MIN_SCORE, RERANK_TOP_N
from src.context import rewrite_query
from src.generator import NO_ANSWER_MESSAGE, generate_answer
from src.hybrid_search import hybrid_search
from src.reranker import rerank


def answer_question(
    question: str, chat_history: list[dict] | None = None, verbose: bool = False
) -> dict:
    """Run the full RAG pipeline for one user question."""
    search_query = rewrite_query(question, chat_history)

    candidates = hybrid_search(search_query)      # Dense + BM25 -> RRF -> Top 20
    if not candidates:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "candidates": [],
            "top_chunks": [],
            "search_query": search_query,
            "refused_by": "no_candidates",
        }
    top = rerank(search_query, candidates, top_n=RERANK_TOP_N)  # BGE -> Top 5

    if not top or top[0].get("rerank_score", 0.0) < RERANK_MIN_SCORE:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "candidates": candidates,
            "top_chunks": top,
            "search_query": search_query,
            "refused_by": "rerank_gate",
        }

    raw, used = generate_answer(question, top, resolved=search_query)
    answer = final_answer(raw, top, used)
    return {
        "answer": answer,
        "candidates": candidates,
        "top_chunks": top,
        "used_indices": used,
        "search_query": search_query,
        "refused_by": "generation" if answer == NO_ANSWER_MESSAGE else None,
    }
