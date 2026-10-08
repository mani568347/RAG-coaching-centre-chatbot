"""Gemini embeddings + persistent local Chroma vector store."""
from __future__ import annotations

import time

import chromadb
import chromadb.config

from src.chunker import Chunk
from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBED_DIM,
    EMBED_MODEL,
    EMBED_TIMEOUT_MS,
    EMBED_TOTAL_BUDGET_S,
    GEMINI_ATTEMPTS_PER_MODEL,
)
from src.llm_client import backoff, gemini_client, is_transient


def _client():
    return gemini_client(EMBED_TIMEOUT_MS)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with Gemini Embedding (order preserved, one vector each).

    One request per text: `gemini-embedding-2` aggregates a batched `contents`
    list into a single vector (verified live — 3 inputs return 1 embedding), so
    batching cannot produce per-chunk vectors.
    """
    client = _client()
    return [_embed_one(client, text) for text in texts]


def _embed_one(client, text: str) -> list[float]:
    deadline = time.monotonic() + EMBED_TOTAL_BUDGET_S
    last_err: Exception | None = None
    for attempt in range(GEMINI_ATTEMPTS_PER_MODEL):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config={"output_dimensionality": EMBED_DIM},
            )
        except Exception as exc:  # APIError, transport timeout, connection reset
            if not is_transient(exc):
                raise RuntimeError(
                    "Sorry, Gemini embedding failed: "
                    f"{str(getattr(exc, 'message', exc)).splitlines()[0][:200]}"
                ) from exc
            last_err = exc
            wait = backoff(attempt)
            if (
                attempt + 1 >= GEMINI_ATTEMPTS_PER_MODEL
                or time.monotonic() + wait >= deadline
            ):
                break
            time.sleep(wait)
            continue
        embeddings = resp.embeddings or []
        if len(embeddings) != 1 or not embeddings[0].values:
            raise RuntimeError(
                f"Gemini returned {len(embeddings)} vectors for one text."
            )
        vector = list(embeddings[0].values)
        if len(vector) != EMBED_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(vector)}."
            )
        return vector
    raise RuntimeError(
        "Sorry — the embedding service is busy or slow to respond (Gemini rate "
        "limit or timeout). Please try again in a minute."
    ) from last_err


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


_CHROMA_CLIENT = None


def _chroma():
    # One client per process: opening chroma.sqlite3 repeatedly can raise
    # WinError 22 (file in use) when another process holds the store.
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _CHROMA_CLIENT = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False, is_persistent=True
                ),
            )
        except OSError as exc:
            raise RuntimeError(
                "The local Chroma store is busy (another process has "
                f"data/chroma_db open: {exc}). Close the other Python process "
                "(Streamlit app, ingest.py or run_eval.py), then try again."
            ) from exc
    return _CHROMA_CLIENT


def store_chunks(chunks: list[Chunk]) -> int:
    """Embed chunks and upsert them into the persistent Chroma collection."""
    if not chunks:
        raise RuntimeError("No chunks to store.")
    vectors = embed_texts([c.text for c in chunks])
    client = _chroma()
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = [c.metadata() for c in chunks]
    batch = 100
    for start in range(0, len(chunks), batch):
        collection.upsert(
            ids=ids[start : start + batch],
            documents=docs[start : start + batch],
            embeddings=vectors[start : start + batch],
            metadatas=metas[start : start + batch],
        )
    return len(chunks)


def collection_ready() -> bool:
    if not CHROMA_DIR.exists():
        return False
    try:
        client = _chroma()
        col = client.get_collection(CHROMA_COLLECTION)
        return col.count() > 0
    except Exception:
        return False


def fetch_all_chunks() -> list[dict]:
    """Return every stored chunk: {id, text, metadata}."""
    client = _chroma()
    col = client.get_or_create_collection(
        name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    data = col.get(include=["documents", "metadatas"])
    out = []
    for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        out.append({"id": cid, "text": doc, "metadata": meta})
    return out


def dense_search(query: str, top_k: int) -> list[dict]:
    """Vector similarity search in Chroma. Returns [{id, text, metadata, score}]"""
    client = _chroma()
    col = client.get_collection(CHROMA_COLLECTION)
    qvec = embed_query(query)
    res = col.query(
        query_embeddings=[qvec],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"id": cid, "text": doc, "metadata": meta, "score": 1.0 - dist})
    return hits
