# CodeCraft Academy — RAG Chatbot

A retrieval-augmented chatbot for the fictional **CodeCraft Academy** coaching
centre. It answers questions **only** from five official 2026 PDF documents
(fees, syllabus, timings, faculty, admissions/facilities), cites the exact
source file and page number for every factual answer, and clearly refuses
questions the documents cannot support (e.g. "What is the Java course fee?").

Built as a learning-portfolio project: every classic RAG component is
implemented explicitly, with no framework hiding the pipeline.

---

## 1. RAG architecture

```text
PDF Documents
      ↓
Docling PDF Parsing (structure, headings, tables, page numbers, OCR)
      ↓
pdfplumber for relevant tables (exact fee rows, ₹ values)
      ↓
Document structure + page information
      ↓
~200-token chunks, 0 overlap (section-aware)
      ↓
Contextual retrieval prefix ("Context: from the ... document, section ...")
      ↓
Gemini Embedding (gemini-embedding-2, 768-d)
      ↓
Chroma Vector Database (persistent, data/chroma_db)
      ↓
                 User Question
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
        Dense Search       BM25 Keyword Search
              ↓                 ↓
              └────────┬────────┘
                       ↓
        Reciprocal Rank Fusion (RRF) → Top 20
                       ↓
        BAAI/bge-reranker-v2-m3 cross-encoder → Top 5
                       ↓
        Gemini Flash (grounded, citation-enforced prompt)
                       ↓
        Grounded Answer + Source/Page citation
        (or the strict NO-ANSWER refusal)
```

## 2. Technology stack

| Role | Technology | Where |
|---|---|---|
| PDF parsing | **Docling** | `src/pdf_parser.py` |
| Table/fee extraction | **pdfplumber** | `src/pdf_parser.py` |
| Chunking | custom, ~200 tokens / 0 overlap | `src/chunker.py` |
| Embeddings | **Gemini Embedding** (`gemini-embedding-2`, see §18) | `src/embeddings.py` |
| Vector DB | **Chroma** (local, persistent) | `data/chroma_db/` |
| Keyword search | **BM25** (`rank_bm25`) | `src/keyword_search.py` |
| Fusion | **RRF** (k=60) | `src/hybrid_search.py` |
| Reranker | **BAAI/bge-reranker-v2-m3** (transformers) | `src/reranker.py` |
| LLM | **Gemini Flash** | `src/generator.py` |
| Gemini client (timeouts, bounded retries) | **google-genai** | `src/llm_client.py` |
| UI | **Streamlit** | `app.py` |

## 3. Folder structure

```text
rag-coaching-chatbot/
├── documents/              # the 5 knowledge-base PDFs (only source of truth)
├── data/
│   ├── chroma_db/          # persistent vector store (created by ingest)
│   └── bm25_index.pkl      # persistent keyword index (created by ingest)
├── src/
│   ├── config.py           # models, paths, chunk/retrieval settings
│   ├── pdf_parser.py       # Docling + pdfplumber
│   ├── chunker.py          # ~200-token chunks + contextual prefixes
│   ├── embeddings.py       # Gemini embeddings + Chroma store/search
│   ├── keyword_search.py   # BM25
│   ├── hybrid_search.py    # Dense + Keyword → RRF → Top 20
│   ├── reranker.py         # bge-reranker-v2-m3 → Top 5
│   ├── context.py          # follow-up / discovery query rewriting
│   ├── generator.py        # Gemini Flash grounded answers
│   ├── llm_client.py       # shared Gemini client: timeouts + bounded retries
│   ├── citations.py        # source/page citation + no-answer logic
│   └── pipeline.py         # full query pipeline wiring
├── scripts/
│   ├── ask.py              # CLI: ask questions without the UI (--chat for follow-ups)
│   └── run_eval.py         # evaluation harness
├── eval/                   # 30-question dataset + reports
├── tests/
│   ├── questions.txt       # the 10 required smoke-test questions
│   └── followup.txt        # 5-turn pronoun-resolution conversation
├── ingest.py               # one-time knowledge-base build
├── app.py                  # Streamlit chatbot
├── requirements.txt
├── .env.example
└── .gitignore
```

## 4. Installation

Requires **Python 3.10+** and a Google AI Studio API key.

```bash
# 0) clone the repository, then enter it
git clone <your-repository-url>.git RAG_chatbot
cd RAG_chatbot

# 1) create and activate a virtual environment
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

```bash
# 2) install dependencies
pip install -r requirements.txt
```

> Tip: `torch` is large (~200 MB CPU build). If it is already installed on
> your system you can create the venv with
> `python -m venv --system-site-packages venv` to reuse it.

## 5. Environment variables

Create a file named `.env` in the project root:

```env
GEMINI_API_KEY=your_key_here
```

Get a free key at <https://aistudio.google.com/apikey>.
Never hardcode the key in source files and never commit `.env`
(already in `.gitignore`).

Everything else is optional — the defaults in `src/config.py` are what the
project ships with, and each is read from the environment so a demo or an
evaluation run can be retuned without editing code:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — (required) | Google AI Studio key for embeddings + generation. |
| `RAG_EMBED_MODEL` | `gemini-embedding-2` | Embedding model for **both** documents and queries (§18). Changing it means re-running `python ingest.py`, because the stored vectors must come from the same model that embeds the question. |
| `RAG_GENERATE_MODEL` | `gemini-flash-latest` | Answer-generation model. Set it to e.g. `gemini-flash-lite-latest` when the default alias is rate-limited; `src/generator.py` already falls back to that alias automatically. |
| `RAG_GENERATE_TIMEOUT_MS` | `60000` | Hard client-side timeout per generation request (milliseconds). |
| `RAG_EMBED_TIMEOUT_MS` | `45000` | Hard client-side timeout per embedding request (milliseconds). |
| `RAG_GEMINI_ATTEMPTS` | `3` | Requests attempted **per model** before giving up (generation tries this many times for each of the two models). SDK-level retries are disabled, so this is the only retry mechanism. |
| `RAG_GENERATE_BUDGET_S` | `150` | Wall-clock budget for one generation; once it is spent the request fails with a clean message instead of hanging. |
| `RAG_EMBED_BUDGET_S` | `90` | Wall-clock budget for embedding one text (the query, or one ingested chunk); bounds the same retry loop on the embedding side. |
| `RAG_RERANK_MIN_SCORE` | `0.01` | No-answer gate (§10). A noise floor, **not** a valid/unsupported classifier — those two populations' scores overlap, so raising it only refuses valid questions. The semantic refusal is made by grounded generation. |

## 6. PDF ingestion (run once)

```bash
python ingest.py
```

This performs Stages 1–5 of the pipeline offline:

1. parses all PDFs in `documents/` with **Docling**, and re-extracts ruled
   tables with **pdfplumber** (exact `₹` fee values, header + rows);
2. cuts the text into **~200-token chunks with 0 overlap** (tables are never
   split mid-row);
3. prepends a **contextual-retrieval line** to every chunk
   (`Context: from the 01 course fee structure (2026) document, section ...`);
4. embeds each chunk with **Gemini Embedding** and upserts into the
   **persistent Chroma** database in `data/chroma_db/`;
5. builds and pickles the **BM25 keyword index** to `data/bm25_index.pkl`.

Re-run it only when the documents change. The Streamlit app **never**
re-creates embeddings — it only opens the existing database.

## 7. Chroma database

Chroma runs fully locally (`chromadb.PersistentClient`) — the embeddings live
in `data/chroma_db/` on disk and survive application restarts. **This database
and `data/bm25_index.pkl` are generated locally by `python ingest.py`; they are
build artifacts, not source, and are never committed** (the whole `data/` tree
is in `.gitignore`). A fresh clone therefore contains no index until you run
ingestion once. The collection
`codecraft_chunks` holds **24 chunks from the 5 PDFs** and stores, per chunk:
the contextualized text, the 768-d Gemini vector, and metadata (real values,
taken verbatim from the ingested collection):

```python
{"source": "01_course_fee_structure_2026.pdf", "page": 1,
 "section": "AI & Coding Courses - Fee Structure 2026",
 "document_type": "fees", "chunk_id": "01_course_fee_structure_2026_p1_c0"}
```

Other `document_type` values are `syllabus`, `timings`, `faculty` and
`admissions_facilities`.

**One process at a time.** Chroma's local SQLite store is single-writer, and on
Windows a second process opening the same `data/chroma_db` while another holds
it can fail with `WinError 22 (file in use)`. Normal usage is already safe —
the Streamlit app only *reads*, and ingestion is a one-off — but do not run
`python ingest.py` while `app.py` or `scripts/run_eval.py` is live. If it
happens anyway, the app reports it as a readable message
(`The local Chroma store is busy …`) instead of a stack trace; close the other
Python process and retry.

## 8. Hybrid search

- **Dense**: the question is embedded with the *same* `EMBED_MODEL` constant
  that ingested the documents (one value in `src/config.py`, so a query can
  never be embedded by a different model than the stored vectors) and compared
  to them by cosine similarity (Chroma `query`).
- **Keyword**: BM25 over the chunk texts with light stopword removal — this
  is what catches exact tokens like course names, `₹48,000`, faculty names.
  The unpickled corpus and the `BM25Okapi` object are built once per process
  and reused (invalidated by the index file's mtime + size, so a re-ingest
  takes effect immediately); measured 0.29 ms per query versus 1.41 ms when
  rebuilding per query, with byte-identical rankings.

Neither alone is used; both are required for the fusion step.

## 9. RRF (Reciprocal Rank Fusion)

The two ranked lists (top 20 each) are merged by score

```text
RRF(d) = Σ_lists  1 / (60 + rank_of_d_in_list)
```

so a chunk that appears in both lists rises to the top. The fused list keeps
the **Top 20** candidates.

## 10. BGE reranking

`BAAI/bge-reranker-v2-m3` (downloaded from Hugging Face, run locally via
`transformers`) scores every `(question, chunk)` pair through a real
cross-encoder; sigmoid-normalised scores re-order the Top 20 down to the
**Top 5** chunks that are handed to the LLM.

The best rerank score also gates generation: if nothing scores above
`RERANK_MIN_SCORE`, the refusal message is returned *before* Gemini is called.
That gate is a **noise floor, not a classifier**, and `0.01` is what the
measurements support. Best-chunk scores per question (30 evaluation questions
+ 18 regression phrasings + deliberately weak code-mixed phrasings; identical
for both candidate embedding models):

| Population | Best-chunk rerank score |
|---|---|
| answerable | **0.0459** (*"full stack fee kitna hai"*), **0.1368** (*"Who teaches Data Science?"*), **0.1970** (*"How much is AI and ML?"*) … 0.9968 |
| unsupported | 0.0006 … **0.1627** (*"What were the 2025 batch timings?"*) |

**The two ranges overlap**, so no threshold can separate them. Three values
were tried and falsified end to end: `0.20` (first chosen from the 30-question
set alone) refused *"How much is AI and ML?"*; `0.18` then refused *"Who teaches
Data Science?"*, which the faculty document answers explicitly
(`Ms. Ananya Rao | Data Science | 7 years`); `0.10` recovered that one but still
refused the code-mixed *"full stack fee kitna hai"*, whose ₹42,000 answer is
sitting in the fee table. A gate that refuses answerable questions damages the
reason the chatbot exists, so the floor stays low — it only saves a Gemini call
when nothing remotely matches (5 of the 6 unsupported questions score ≤ 0.0077).

The semantic no-answer decision is therefore made by the **grounded generator**:
`generate_answer` returns `NOT_FOUND` when the Top 5 chunks cannot support the
question, and that is what refuses *"What were the 2025 batch timings?"*
(0.1627 — through the gate) and the rephrased *"Java course fee"* (0.0360). Measured split at
`0.01`: 5 unsupported refused by the gate, 1 by the LLM, 0 hallucinations, and
every answerable question answered. `RAG_RERANK_MIN_SCORE` in `.env` overrides
the floor without editing code.

`refused_by` is reported by `src/pipeline.py` (`rerank_gate`, `generation` or
`no_candidates`), so a refusal can be attributed instead of taken on faith.

## 11. Gemini generation

The Top 5 chunks are passed to **Gemini Flash** with a strict prompt: use only
the supplied context, never outside knowledge, preserve exact numbers, and
reply `NOT_FOUND` when the evidence is insufficient. Every non-`NOT_FOUND`
answer must also end with a mandatory line naming the context blocks it relied
on — `[used: 1, 3]` — which the code parses off before showing the answer.

Generation is bounded end to end: a 60 s client-side timeout per request, at
most 3 attempts on each of the two Flash aliases, and a 150 s wall-clock budget
(§5). Embeddings use the same loop with a 45 s timeout and a 90 s budget per
text. When the budget is spent the user sees
*"Sorry — the answer service is busy or slow to respond…"* — never a stack
trace, and never an infinite hang.

## 12. Citation system

Citations are assembled **from chunk metadata only** — never generated by the
LLM — so a page number can only be a page number that was actually read from
the PDF:

```text
The AI & ML course tuition fee is ₹48,000 and the duration is 6 months.

Source:
01_course_fee_structure_2026.pdf — Page 1
```

Multiple documents produce a `Sources:` list.

The set of cited chunks is the set that supports the answer, nothing more:

1. the `[used: n]` indices are validated against the Top 5 (out-of-range and
   duplicate numbers are dropped);
2. if the marker is missing or every index is invalid, `src/citations.py`
   scores each Top-5 chunk by how much of the answer's own wording and numbers
   it contains, and cites the chunks that match (falling back to the single
   best candidate when nothing matches at all).

The earlier behaviour — citing the whole Top 5 whenever the model forgot the
marker — is gone, because it over-attributed sources. A filename or page
number that the model makes up cannot reach the output either: the answer text
is shown as-is, but the `Source:` block is built from stored metadata fields,
so an invented file simply never appears.

## 13. No-answer behaviour

```text
User:  What is the Java course fee?
Bot:   I couldn't find information about that in the provided CodeCraft
       Academy documents.
```

Guaranteed by two layers: the rerank-score gate and the `NOT_FOUND` rule in
the generation prompt. The model never guesses.

## 14. Running the chatbot

```bash
streamlit run app.py
```

Open <http://localhost:8501>. The first screen shows a welcome note and six
suggested actions (**View all courses, Course fees, Syllabus, Faculty, Batch
timings, Admission process**) that each send a real question through the same
RAG pipeline — nothing about them is hardcoded as an answer. The chat supports
history, follow-up questions, citations, and a **Clear chat** button.

### Conversational follow-ups

A follow-up such as "Who teaches it?" is first resolved into a standalone
search query using the last few turns (`src/context.py`, running *before*
retrieval — the pipeline itself is unchanged), so "it" correctly means the
course just discussed. Broad discovery questions ("Tell me about the academy",
"What programs are available?") are matched to a fixed description of the
corpus instead, because they name no topic at all. Neither rewrite adds facts:
the answers still come only from retrieved chunks, with citations.

The rewrite is skipped entirely for a first-turn question that already names a
topic, so single-question behaviour is identical to before.

CLI (no UI), useful while developing:

```bash
python scripts/ask.py "What is the AI and ML course fee?"
python scripts/ask.py --file tests/questions.txt

# one conversation, follow-ups resolved from history:
python scripts/ask.py --chat --file tests/followup.txt
```

## 15. Testing & evaluation

Required smoke tests live in `tests/questions.txt` (the last one must refuse).
The full evaluation dataset is `eval/questions.json` (30 questions: fees,
syllabus, timings, faculty, admissions, paraphrases, and unanswerables):

```bash
python scripts/run_eval.py
```

It writes `eval/results_<timestamp>.json` (per-question retrieved chunks,
final answer, citation correctness) and `eval/report.md` with aggregate
metrics: retrieval quality (Top 20), reranking quality (Top 5), answer
correctness, citation correctness, numerical accuracy, hallucination rate,
and unsupported-question handling.

`report.md` always describes the **most recent** run; the timestamped JSON
files are never overwritten and earlier reports are kept side by side as
`eval/report_<run-id>.md`, so a result can be traced back to the code that
produced it. The retrieval-only analyses used to tune the gate
(`eval/retrieval_ab_<model>.json`) are generated artifacts and are gitignored.

## 16. Common errors

| Symptom | Cause / fix |
|---|---|
| `GEMINI_API_KEY is not set` | Create `.env` (see §5), restart the app. |
| `The knowledge base has not been ingested yet` | Run `python ingest.py` once. |
| `Keyword (BM25) index not found` | Same — `python ingest.py` builds it. |
| `gemini-2.0-flash no longer available` | Google retires pinned models; `config.py` uses the evergreen `gemini-flash-latest` by default. Set `RAG_GENERATE_MODEL` in `.env` to override without touching code. |
| `high demand` / `quota exceeded` (429) | Free tier is rate-limited (≈20 req/min plus a daily per-model cap, which resets around 13:30 IST). The client retries a bounded number of times with backoff and then fails with *"Sorry — the answer service is busy or slow to respond…"*; it no longer hangs. Wait for the quota window and ask again. |
| A request seems to take a long time | Every Gemini call now has a hard ceiling (60 s generation, 45 s embedding, 150 s total budget — §5), so a request fails inside that window instead of blocking indefinitely. |
| `The local Chroma store is busy (another process has data/chroma_db open …)` | Windows file lock (`WinError 22`): only one process may hold `data/chroma_db`. Close the Streamlit app / evaluation run before `python ingest.py` (see §7). |
| `Embedding dimension mismatch` / poor retrieval right after changing `RAG_EMBED_MODEL` | The stored vectors were built by a different model. Re-run `python ingest.py` so documents and queries use the same embedding model (back up `data/chroma_db` first if you want to compare). |
| First run is slow | Docling downloads layout/table models once; the reranker downloads `bge-reranker-v2-m3` (~2.3 GB) once. |
| Corrupted/empty PDF | `ingest.py` prints a clear error naming the file. |
| Windows console `UnicodeEncodeError` | run with `python -X utf8 ...` (the app itself is unaffected). |

## 17. Design notes

- **No LangChain/LlamaIndex** — each pipeline stage is a small, readable
  Python module so the mechanics (RRF, cross-encoder scoring, contextual
  prefixes) stay visible.
- Chunking targets ~200 tokens (≈1000 chars, the PDFs' natural sections fit
  inside that; average chunk is ~74 tokens) with **zero overlap**; table rows
  are atomic units and table chunks repeat the header row for self-containment.
- The fee PDF encodes `₹` with a glyph that plain extractors render as `n`
  (`n48,000`); `src/pdf_parser.py` repairs currency values and one
  column-boundary artefact **only inside table cells**, so numbers reach the
  database exactly as printed.

## 18. Which embedding model — and why

The project specification asked for `gemini-embedding-2`. Rather than assume
the name is wrong or blindly swap it in, it was verified against the live API
with the installed SDK (`google-genai` 2.22.0):

```python
client = genai.Client()
[n.name for n in client.models.list(config={"page_size": 200}) if "embedding" in n.name]
# ['models/gemini-embedding-001', 'models/gemini-embedding-2', 'models/gemini-embedding-2-preview']

client.models.embed_content(model="gemini-embedding-2", contents=["hello"],
                            config={"output_dimensionality": 768})   # -> 768 floats, HTTP 200
```

So `gemini-embedding-2` **is** available and is now the default for both
documents and queries (`EMBED_MODEL` in `src/config.py`). The verification also
found one behavioural difference worth knowing about:

- `gemini-embedding-001` returns **one vector per input** when `contents` is a
  list (3 texts → 3 embeddings).
- `gemini-embedding-2` returns **one vector for the whole list** (3 texts → 1
  aggregated embedding).

Batching therefore cannot produce per-chunk vectors, so `embed_texts()` sends
**one request per text** — 24 chunks are ingested in ~16 s, and a query is a
single request exactly as before.

Both models were then measured on the same 30 questions through the real
retrieval path (hybrid search → RRF → BGE rerank, stored in
`eval/retrieval_ab_<model>.json`):

| Measure | `gemini-embedding-001` | `gemini-embedding-2` |
|---|---|---|
| expected chunk in Top 20 | 24/24 | 24/24 |
| expected chunk in Top 5 | 24/24 | 24/24 |
| weakest answerable rerank score (of these 30) | 0.2394 | 0.2394 |
| strongest unsupported rerank score (of these 30) | 0.1627 | 0.1627 |
| dense rank-1 on probe questions | correct | correct, with wider cosine margins (e.g. 0.818 vs 0.791 on the fee question) |

The two models are equivalent on this corpus (the tiny 24-chunk store means
BM25 + reranking dominate the outcome), so the spec-named model was kept.
To compare or roll back: set `RAG_EMBED_MODEL=gemini-embedding-001` in `.env`
and re-run `python ingest.py`. The pre-migration store is left untouched at
`data/chroma_db_backup_embed001/` (inside the ignored `data/` directory).
