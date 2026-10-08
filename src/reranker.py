"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

Real reranking: every (user query, candidate chunk) pair is passed through
the cross-encoder, which outputs a relevance logit; sigmoid converts it to
a 0-1 score used to re-order the Top 20 hybrid candidates down to Top 5.
"""
from __future__ import annotations

import math

_MODEL = None
_TOKENIZER = None
_DEVICE = None

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"


def _load():
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _TOKENIZER = AutoTokenizer.from_pretrained(RERANKER_NAME)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(RERANKER_NAME)
    _MODEL.to(_DEVICE)
    _MODEL.eval()


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """Score candidates against the query and return the Top N."""
    if not candidates:
        return []
    _load()
    import torch

    pairs = [[query, c["text"]] for c in candidates]
    with torch.no_grad():
        batch = _TOKENIZER(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(_DEVICE)
        logits = _MODEL(**batch).logits.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]

    ranked = sorted(
        zip(candidates, scores), key=lambda cs: cs[1], reverse=True
    )[:top_n]
    out = []
    for hit, score in ranked:
        item = dict(hit)
        item["rerank_score"] = float(score)
        out.append(item)
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
