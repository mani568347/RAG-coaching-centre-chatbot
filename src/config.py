"""Shared configuration for the CodeCraft Academy RAG chatbot."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "bm25_index.pkl"

CHROMA_COLLECTION = "codecraft_chunks"

# Embedding / generation models (Gemini).
# The spec named gemini-embedding-2; `client.models.list()` plus a real
# embed_content call confirm it is GA and accepts output_dimensionality=768,
# so it is the default. RAG_EMBED_MODEL=gemini-embedding-001 falls back to the
# previous model (re-ingest is required when switching — the stored vectors
# must come from the same model that embeds queries).
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "gemini-embedding-2").strip()
EMBED_DIM = 768
# Gemini Flash (evergreen alias); RAG_GENERATE_MODEL lets an eval run pick
# another Flash alias when the default one is rate-limited.
GENERATE_MODEL = os.getenv("RAG_GENERATE_MODEL", "gemini-flash-latest").strip()

# Gemini reliability. Under free-tier throttling a single request once took
# ~163 s; these bound every call so a request fails fast and the retry loops
# terminate instead of hanging the UI.
GENERATE_TIMEOUT_MS = int(os.getenv("RAG_GENERATE_TIMEOUT_MS", "60000"))
EMBED_TIMEOUT_MS = int(os.getenv("RAG_EMBED_TIMEOUT_MS", "45000"))
# Requests attempted **per model**; SDK-level retries are disabled so this loop
# is the only retry mechanism (avoids attempts multiplying). 3 rather than 2
# because a single "Server disconnected without sending a response" from Gemini
# otherwise aborts the whole turn.
GEMINI_ATTEMPTS_PER_MODEL = int(os.getenv("RAG_GEMINI_ATTEMPTS", "3"))
GENERATE_TOTAL_BUDGET_S = float(os.getenv("RAG_GENERATE_BUDGET_S", "150"))
# Wall-clock ceiling for embedding one text (query + each ingested chunk), so
# the extra attempts stay bounded instead of hanging.
EMBED_TOTAL_BUDGET_S = float(os.getenv("RAG_EMBED_BUDGET_S", "90"))
RETRY_BACKOFF_CAP_S = 15.0

# Chunking targets
CHUNK_TOKENS = 200          # ~200 tokens
CHUNK_CHARS = 1000          # heuristic ceiling: 1 token ~= 4 chars
CHUNK_OVERLAP = 0           # required: zero overlap

# Retrieval funnel
TOP_K_DENSE = 20
TOP_K_KEYWORD = 20
RRF_K = 60
RRF_FUSE_TOP = 20           # hybrid retrieval produces Top 20
RERANK_TOP_N = 5            # after reranking keep Top 5

# No-answer guard. This is a NOISE FLOOR, not a classifier: the best-chunk
# rerank scores of answerable and unsupported questions overlap, so no
# threshold can separate them. Measured over the 30 evaluation questions, the
# 18-question regression and extra weak phrasings:
#   answerable : 0.0459 ("full stack fee kitna hai"), 0.1368 ("Who teaches
#                Data Science?"), 0.1970 ("How much is AI and ML?") … 0.9968
#   unsupported: 0.0006 … 0.1627 ("What were the 2025 batch timings?")
# A threshold high enough to block the 0.1627 unsupported question (0.20, and
# then 0.18) demonstrably refuses *valid* questions instead, so 0.01 only skips
# the Gemini call when nothing remotely matches. The semantic no-answer
# decision is made by grounded generation (NOT_FOUND), which refused all 6
# unsupported questions: 5 caught here (scores <= 0.0077) and "2025 batch
# timings" caught by the LLM at 0.1627 — zero hallucinations.
RERANK_MIN_SCORE = float(os.getenv("RAG_RERANK_MIN_SCORE", "0.01"))
