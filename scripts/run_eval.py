"""Evaluation harness for the CodeCraft Academy RAG chatbot.

    python scripts/run_eval.py

Runs every question in eval/questions.json through the real pipeline and
scores: retrieval quality, reranking quality, answer correctness, citation
correctness, numerical accuracy, hallucination rate and unsupported-question
handling. Results: eval/results_<timestamp>.json + eval/report.md
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.pipeline import answer_question  # noqa: E402

NO_ANSWER_TOKEN = "couldn't find"


def _expected_chunk_ids(item: dict, id_to_meta: dict[str, dict]) -> set[str]:
    if not item["expected_source"]:
        return set()
    return {
        cid
        for cid, m in id_to_meta.items()
        if m["source"] == item["expected_source"] and m["page"] == item["expected_page"]
    }


def evaluate() -> dict:
    items = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))

    # Ground truth chunk ids from the Chroma store
    from src.embeddings import fetch_all_chunks
    from src.keyword_search import keyword_search

    stored = fetch_all_chunks()
    id_to_meta = {c["id"]: c["metadata"] for c in stored}

    rows = []
    for item in items:
        q = item["question"]
        expected_ids = _expected_chunk_ids(item, id_to_meta)
        # retrieval quality: keyword-side proxy of whether expected doc is findable
        expanded = f"{q} " + " ".join(item.get("expected_answer_contains") or [])
        kw_ids = {h["id"] for h in keyword_search(expanded, 20)}
        retrieval_hit = bool(expected_ids & kw_ids) if expected_ids else None

        t0 = time.time()
        try:
            result = answer_question(q)
        except Exception as exc:
            rows.append({**item, "error": str(exc)})
            time.sleep(3)
            continue
        answer = result["answer"]
        candidates = result["candidates"]
        top = result["top_chunks"]

        in_top20 = bool(expected_ids & {c["id"] for c in candidates}) if expected_ids else None
        in_top5 = bool(expected_ids & {c["id"] for c in top}) if expected_ids else None

        answered = NO_ANSWER_TOKEN not in answer.lower()
        if item["type"] == "unanswerable":
            correct = not answered  # strict: must refuse
            hallucinated = answered
            citation_ok = None
            numerical_ok = None
        else:
            correct = answered and all(
                tok.lower() in answer.lower() for tok in item["expected_answer_contains"]
            )
            hallucinated = False
            citation_ok = answered and (
                f"{item['expected_source']} — Page {item['expected_page']}" in answer
            )
            numerical_ok = (
                all(tok.lower() in answer.lower() for tok in item["expected_answer_contains"])
                if answered
                else False
            )

        rows.append(
            {
                **item,
                "final_answer": answer,
                "retrieval_keyword_hit": retrieval_hit,
                "expected_in_top20": in_top20,
                "expected_in_top5": in_top5,
                "answer_correct": correct,
                "citation_correct": citation_ok,
                "numerical_exact": numerical_ok,
                "hallucinated": hallucinated,
                "refused_by": result.get("refused_by"),
                "best_rerank": max(
                    (c.get("rerank_score", 0.0) for c in top), default=None
                ),
                "latency_s": round(time.time() - t0, 1),
            }
        )
        print(
            f"[{item['id']:>2}] correct={correct} cite={citation_ok} "
            f"top20={in_top20} top5={in_top5} :: {q[:60]}"
        )
        time.sleep(3)  # stay under free-tier request rate

    answerable = [r for r in rows if r["type"] != "unanswerable" and "error" not in r]
    unanswerable = [r for r in rows if r["type"] == "unanswerable" and "error" not in r]
    n = len(answerable)
    summary = {
        "total_questions": len(rows),
        "errors": len([r for r in rows if "error" in r]),
        "answerable": n,
        "unanswerable": len(unanswerable),
        "retrieval_top20_quality": _rate(answerable, "expected_in_top20"),
        "rerank_top5_quality": _rate(answerable, "expected_in_top5"),
        "answer_correctness": _rate(answerable, "answer_correct"),
        "citation_correctness": _rate(answerable, "citation_correct"),
        "numerical_accuracy": _rate(answerable, "numerical_exact"),
        "hallucination_rate": _rate(unanswerable, "hallucinated"),
        "unsupported_question_handling": _rate(unanswerable, "answer_correct"),
    }
    return {"summary": summary, "rows": rows}


def _numeric_tokens(answer: str) -> list[str]:
    return re.findall(r"\d[\d,]*\.?\d*", answer)


def _rate(rows, key) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return "n/a"
    return f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)"


def write_reports(data: dict) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (ROOT / "eval" / f"results_{ts}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = ["# CodeCraft Academy RAG — Evaluation Report", ""]
    md.append(f"Run: {ts}\n")
    md.append("| Metric | Score |")
    md.append("|---|---|")
    for k, v in data["summary"].items():
        md.append(f"| {k.replace('_', ' ')} | {v} |")
    md.append("\n## Per-question results\n")
    md.append("| # | Question | Top20 | Top5 | Correct | Citation | Numbers | Answer |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in data["rows"]:
        if "error" in r:
            md.append(f"| {r['id']} | {r['question'][:50]} | - | - | ERROR: {r['error'][:40]} | - | - | - |")
            continue
        md.append(
            f"| {r['id']} | {r['question'][:50]} | {r['expected_in_top20']} | "
            f"{r['expected_in_top5']} | {r['answer_correct']} | {r['citation_correct']} | "
            f"{r['numerical_exact']} | {r['final_answer'][:70].replace(chr(10), ' ')} |"
        )
    (ROOT / "eval" / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote eval/results_{ts}.json and eval/report.md")


if __name__ == "__main__":
    data = evaluate()
    print("\n===== SUMMARY =====")
    for k, v in data["summary"].items():
        print(f"{k:35s} {v}")
    write_reports(data)
