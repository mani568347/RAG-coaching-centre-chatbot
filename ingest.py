"""One-shot ingestion: PDFs -> Docling/pdfplumber -> chunks -> Gemini
embeddings -> persistent Chroma -> BM25 index.

Run once (and again only when documents change):

    python ingest.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.chunker import chunk_document_map
from src.embeddings import fetch_all_chunks, store_chunks
from src.keyword_search import build_index
from src.pdf_parser import DOCUMENTS_DIR, parse_all


def main() -> int:
    print(f"[1/4] Parsing PDFs in {DOCUMENTS_DIR} with Docling + pdfplumber ...")
    t0 = time.time()
    try:
        doc_map = parse_all()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR while parsing: {exc}")
        return 1
    print(f"      Parsed {len(doc_map)} documents in {time.time() - t0:.1f}s")

    print("[2/4] Chunking (~200 tokens, 0 overlap, contextual prefixes) ...")
    chunks = chunk_document_map(doc_map)
    if not chunks:
        print("ERROR: chunking produced no chunks (empty or corrupted PDFs?).")
        return 1
    avg = sum(c.token_estimate for c in chunks) / len(chunks)
    print(f"      {len(chunks)} chunks, average ~{avg:.0f} tokens/chunk")

    print("[3/4] Embedding with Gemini and storing in Chroma (data/chroma_db) ...")
    t0 = time.time()
    try:
        stored = store_chunks(chunks)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"      Stored {stored} chunks in {time.time() - t0:.1f}s")

    print("[4/4] Building persistent BM25 keyword index ...")
    build_index(fetch_all_chunks())
    print(f"      BM25 index written to data/bm25_index.pkl")

    print("\nIngestion complete. Start the chatbot with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
