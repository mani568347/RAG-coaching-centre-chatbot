"""BM25 keyword retrieval over the ingested chunks.

The BM25 index is built once during ingestion and persisted to disk, so the
Streamlit app does not rebuild it on every start.
"""
from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "or",
    "in", "on", "for", "with", "what", "how", "when", "who", "which", "do",
    "does", "did", "i", "my", "me", "you", "it", "its", "at", "by", "be",
    "from", "this", "that", "there", "they", "them", "their", "as", "am",
    "about", "can", "will", "would", "should",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9&+]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def build_index(chunks: list[dict]) -> None:
    """Build and persist the BM25 index from stored chunk dicts."""
    corpus = [tokenize(c["text"]) for c in chunks]
    payload = {
        "ids": [c["id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metadatas": [c["metadata"] for c in chunks],
        "corpus": corpus,
    }
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as fh:
        pickle.dump(payload, fh)


_CACHE: dict = {"key": None, "payload": None, "bm25": None}


def _load():
    if not BM25_INDEX_PATH.exists():
        raise RuntimeError(
            "Keyword (BM25) index not found. Run `python ingest.py` first."
        )
    stat = BM25_INDEX_PATH.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    if _CACHE["key"] != key:  # re-ingest writes a new pickle -> rebuild once
        with open(BM25_INDEX_PATH, "rb") as fh:
            payload = pickle.load(fh)
        _CACHE.update(key=key, payload=payload, bm25=BM25Okapi(payload["corpus"]))
    return _CACHE["payload"], _CACHE["bm25"]


def keyword_search(query: str, top_k: int) -> list[dict]:
    """Return top_k chunks ranked by BM25: [{id, text, metadata, score}]"""
    data, bm25 = _load()
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "id": data["ids"][i],
            "text": data["texts"][i],
            "metadata": data["metadatas"][i],
            "score": float(scores[i]),
        }
        for i in order
        if scores[i] > 0
    ]
