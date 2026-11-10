import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .rag_chain import build_rag_chain, get_llm
from .few_shot_chain import build_few_shot_rag_chain
from .few_shot_no_cot import build_few_shot_rag_chain as build_few_shot_no_cot_chain
from .prompts import LLM_STRUCTURED_SYSTEM


DIRECT_LLM_SYSTEM = """You are a Building Technology assistant.

Answer the user's question directly using your own knowledge.
Do not use any external context or retrieval.
If you are uncertain, say so clearly.

Output format (keep the section titles exactly as written below):

1) Conclusion
2) Explanation
3) Confidence / Limits

Language rule:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
"""


def _extract_answer(out: Any) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for k in ("answer", "result", "output", "output_text", "text"):
            if k in out and isinstance(out[k], str):
                return out[k]
        return json.dumps(out, ensure_ascii=False, indent=2)
    return str(out)


def run_llm(question: str) -> Dict[str, Any]:
    llm = get_llm()
    prompt = [
        ("system", DIRECT_LLM_SYSTEM),
        ("human", f"Question: {question}"),
    ]
    out = llm.invoke(prompt)

    answer = getattr(out, "content", None)
    if not isinstance(answer, str):
        answer = str(out)

    return {
        "mode": "llm",
        "question": question,
        "answer": answer,
        "sources": [],
    }


def run_llm_structured(question: str) -> Dict[str, Any]:
    llm = get_llm()
    out = llm.invoke(
        [
            ("system", LLM_STRUCTURED_SYSTEM),
            ("human", f"Question: {question}"),
        ]
    )
    answer = getattr(out, "content", None)
    return {
        "mode": "llm_structured",
        "question": question,
        "answer": answer if isinstance(answer, str) else str(out),
        "sources": [],
    }


def run_rag(question: str) -> Dict[str, Any]:
    rag = build_rag_chain()
    out = rag.invoke({"input": question})
    return {
        "mode": "rag",
        "question": question,
        "answer": _extract_answer(out),
        "citations": out.get("citations", []) if isinstance(out, dict) else [],
    }


def run_fewshot(question: str, num_examples: int = 3) -> Dict[str, Any]:
    chain = build_few_shot_rag_chain(num_examples=num_examples)
    out = chain.invoke({"input": question})

    return {
        "mode": f"fewshot_{num_examples}",
        "question": question,
        "answer": _extract_answer(out),
        "citations": out.get("citations", []) if isinstance(out, dict) else [],
        "sources": out.get("sources", []) if isinstance(out, dict) else [],
        "num_examples": num_examples,
    }


def run_fewshot_no_cot(question: str, num_examples: int = 3) -> Dict[str, Any]:
    chain = build_few_shot_no_cot_chain(num_examples=num_examples)
    out = chain.invoke({"input": question})
    return {
        "mode": f"fewshot_no_cot_{num_examples}",
        "question": question,
        "answer": _extract_answer(out),
        "citations": out.get("citations", []) if isinstance(out, dict) else [],
        "sources": out.get("sources", []) if isinstance(out, dict) else [],
        "num_examples": num_examples,
    }


def run_agent(question: str, max_steps: int = 4) -> Dict[str, Any]:
    # lazy import so llm/rag/fewshot modes still work without langgraph installed
    from .agent_graph import build_agent_graph, extract_sources

    agent = build_agent_graph(max_steps=max_steps)
    out = agent.invoke({"question": question})

    return {
        "mode": "agent",
        "question": question,
        "answer": _extract_answer(out.get("answer", "")),
        "sources": extract_sources(out.get("docs", [])),
        "steps": out.get("steps", []),
        "iterations": out.get("iterations", 0),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["llm", "llm_structured", "rag", "fewshot", "fewshot_no_cot", "agent"],
        default="fewshot",
        help="Select run mode",
    )
    ap.add_argument("--q", required=True, help="Input question (natural language)")
    ap.add_argument(
        "--fewshot-n",
        type=int,
        default=3,
        help="Number of few-shot examples to use when mode=fewshot",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Max ReAct steps when mode=agent",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (useful for evaluation/logging)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Append results to a jsonl file",
    )
    ap.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieval sources (for debugging)",
    )

    args = ap.parse_args()

    if args.mode == "llm":
        result = run_llm(args.q)
    elif args.mode == "llm_structured":
        result = run_llm_structured(args.q)
    elif args.mode == "rag":
        result = run_rag(args.q)
    elif args.mode == "agent":
        result = run_agent(args.q, max_steps=args.max_steps)
    elif args.mode == "fewshot_no_cot":
        result = run_fewshot_no_cot(args.q, num_examples=args.fewshot_n)
    else:
        result = run_fewshot(args.q, num_examples=args.fewshot_n)

    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    if not args.show_sources:
        result.pop("sources", None)
        result.pop("citations", None)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()