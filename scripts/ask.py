"""CLI test harness: ask questions through the full pipeline without Streamlit.

    python scripts/ask.py "What is the AI and ML course fee?"
    python scripts/ask.py --file tests/questions.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> int:
    args = sys.argv[1:]
    questions: list[str] = []
    shared_history = False
    if args and args[0] == "--chat":
        # every question becomes a turn of ONE conversation
        shared_history = True
        args = args[1:]
    if args and args[0] == "--file":
        questions = [
            line.strip()
            for line in Path(args[1]).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        questions = args

    if not questions:
        print(
            'usage: python scripts/ask.py [--chat] "question" ["question2" ...]'
            "  |  --file questions.txt"
        )
        return 1

    from src.pipeline import answer_question

    history: list[dict] = []
    for q in questions:
        print("\n" + "=" * 70)
        print("Q:", q)
        try:
            result = answer_question(
                q, chat_history=history if shared_history else None
            )
        except Exception as exc:
            print("ERROR:", exc)
            continue
        print("A:", result["answer"])
        if shared_history and result.get("search_query") != q:
            print("(searched as:", result["search_query"], ")")
        print(f"(candidates={len(result['candidates'])}, "
              f"top={len(result['top_chunks'])}, "
              f"best_rerank={result['top_chunks'][0]['rerank_score']:.3f})"
              if result["top_chunks"] else "(no candidates)")
        if shared_history:
            history += [
                {"role": "user", "content": q},
                {"role": "assistant", "content": result["answer"]},
            ]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
