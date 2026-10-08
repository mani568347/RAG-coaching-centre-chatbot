"""CodeCraft Academy — Streamlit RAG chatbot.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="CodeCraft Academy", page_icon="🎓", layout="centered")

SUGGESTIONS = {
    "View all courses": "What courses do you offer?",
    "Course fees": "What are the fees for all courses?",
    "Syllabus": "What is the syllabus of the courses?",
    "Faculty": "Who are the faculty members?",
    "Batch timings": "What are the batch timings?",
    "Admission process": "What is the admission process?",
}


def _precheck() -> str | None:
    """Return a human-readable fatal problem, or None if the app can run."""
    import os

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return (
            "GEMINI_API_KEY is not set. Create a `.env` file in the project "
            "root containing `GEMINI_API_KEY=your_key` (see `.env.example`), "
            "then restart the app."
        )
    from src.embeddings import collection_ready

    if not collection_ready():
        return (
            "The knowledge base has not been ingested yet. Run "
            "`python ingest.py` in a terminal, then reload this page."
        )
    return None


def ask(question: str) -> None:
    """Render one user turn and its answer, appending both to history."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    with st.chat_message("assistant"):
        try:
            from src.pipeline import answer_question

            history = list(st.session_state.messages[:-1])  # turns before this one
            with st.spinner("Searching the academy documents ..."):
                result = answer_question(question, chat_history=history)
            answer = result["answer"]
            st.markdown(answer)

            top = result.get("top_chunks") or []
            if top and "couldn't find" not in answer:
                with st.expander("Retrieval details (debug)"):
                    if result.get("search_query") != question:
                        st.caption(
                            f"Understood as: _{result['search_query']}_"
                        )
                    for i, c in enumerate(top, start=1):
                        m = c["metadata"]
                        st.markdown(
                            f"**{i}.** {m['source']} — page {m['page']} — "
                            f"section *{m['section']}* — rerank score "
                            f"{c.get('rerank_score', 0):.3f}"
                        )
                        st.text(c["text"].split("\n\n", 1)[-1][:300])
        except Exception as exc:  # surface friendly errors, never stack traces
            message = str(exc).strip()
            known = message.startswith(
                ("Sorry", "GEMINI_API_KEY", "The local Chroma store", "Keyword (BM25)")
            )
            answer = message if known else f"Sorry, something went wrong: {message}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


st.title("CodeCraft Academy")
st.caption("AI & Coding Assistant — RAG over the academy's official 2026 documents")

problem = _precheck()
if problem:
    st.warning(problem)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []          # [{"role":..., "content":...}]

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

with st.sidebar:
    st.subheader("You can ask about")
    st.markdown(
        "- Course fees & durations\n- Syllabus topics\n- Batch timings\n"
        "- Faculty details\n- Admission process & documents\n- Refund, discount, "
        "installment policies\n- Facilities (lab, mentoring, placements)"
    )
    st.divider()
    st.caption(
        "Answers come only from the 5 academy PDFs; every factual answer cites "
        "its source file and page. Follow-ups like \"who teaches it?\" use the "
        "conversation so far."
    )

if not st.session_state.messages:
    st.subheader("Hi! I'm the CodeCraft Academy assistant.")
    st.markdown(
        "I can answer questions about our **2026 courses, fees, syllabus, batch "
        "timings, faculty, admissions and facilities** — straight from the "
        "academy's official documents, with the source file and page cited."
    )
    st.markdown("Pick a starting point, or just type your question below:")
    labels = list(SUGGESTIONS)
    for row in range(0, len(labels), 3):
        cols = st.columns(3)
        for col, label in zip(cols, labels[row : row + 3]):
            if col.button(label, use_container_width=True):
                ask(SUGGESTIONS[label])
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about fees, timings, syllabus, faculty, admissions ...")

if user_input:
    ask(user_input)
