"""Hybrid retrieval: dense (Chroma) + keyword (BM25) fused with RRF."""
from __future__ import annotations

from src.config import (
    RRF_FUSE_TOP,
    RRF_K,
    TOP_K_DENSE,
    TOP_K_KEYWORD,
)
from src.embeddings import dense_search
from src.keyword_search import keyword_search


def reciprocal_rank_fusion(
    *rankings: list[dict], k: int = RRF_K, top_n: int = RRF_FUSE_TOP
) -> list[dict]:
    """Fuse multiple ranked hit lists into one using Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (k + rank(d))
    """
    fused: dict[str, float] = {}
    best_hit: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit["id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best_hit:
                best_hit[cid] = hit
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    results = []
    for cid, score in ordered:
        hit = dict(best_hit[cid])
        hit["rrf_score"] = score
        results.append(hit)
    return results


def hybrid_search(query: str) -> list[dict]:
    """Dense + keyword retrieval fused by RRF, returning Top 20 candidates."""
    dense = dense_search(query, TOP_K_DENSE)
    keyword = keyword_search(query, TOP_K_KEYWORD)
    return reciprocal_rank_fusion(dense, keyword, top_n=RRF_FUSE_TOP)
