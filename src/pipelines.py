from __future__ import annotations

import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from .citations import format_context, parse_citation_labels, resolve_citations
from .config import settings
from .pricing import estimate_cost
from .prompts import LLM_STRUCTURED_SYSTEM, LLM_SYSTEM
from .rag_chain import RAG_PROMPT, build_retriever, get_llm


def _empty_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_cny": 0.0,
    }


def _usage_from(response: Any) -> dict[str, Any]:
    meta = getattr(response, "usage_metadata", None) or {}
    inp = int(meta.get("input_tokens", 0) or 0)
    out = int(meta.get("output_tokens", 0) or 0)
    total = int(meta.get("total_tokens", inp + out) or (inp + out))
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
        "cost_cny": estimate_cost(inp, out),
    }


def _invoke(messages: Any) -> tuple[str, dict[str, Any]]:
    response = get_llm().invoke(messages)
    content = getattr(response, "content", None)
    if not isinstance(content, str):
        content = str(response)
    return content, _usage_from(response)


def _fewshot_messages(
    system: str,
    examples: list[dict[str, str]],
    num_examples: int,
    question: str,
    context: str,
) -> list[Any]:
    selected = examples[: max(0, num_examples)]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Question: {question}\n\nContext:\n{context}"),
            ("ai", "{answer}"),
        ]
    )
    messages: list[Any] = [("system", system)]
    if selected:
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=selected,
            example_prompt=example_prompt,
        )
        messages.extend(few_shot_prompt.format_messages())
    messages.append(("human", f"Question: {question}\n\nContext:\n{context}"))
    return messages


def answer_llm(question: str, structured: bool = True) -> dict[str, Any]:
    start = time.perf_counter()
    system = LLM_STRUCTURED_SYSTEM if structured else LLM_SYSTEM
    messages = [("system", system), ("human", f"Question: {question}")]
    answer, usage = _invoke(messages)
    return {
        "answer": answer,
        "context": "",
        "citations": [],
        "retrieved": [],
        "usage": usage,
        "latency_sec": round(time.perf_counter() - start, 4),
        "mode": "llm_structured" if structured else "llm",
    }


def answer_rag(
    question: str,
    k: int | None = None,
    fewshot_n: int = 0,
    no_cot: bool = False,
) -> dict[str, Any]:
    from .few_shot_chain import FEW_SHOT_EXAMPLES, SYSTEM_TEXT as FEWSHOT_SYSTEM
    from .few_shot_no_cot import FEW_SHOT_EXAMPLES as NO_COT_EXAMPLES
    from .few_shot_no_cot import SYSTEM_TEXT as NO_COT_SYSTEM

    start = time.perf_counter()
    docs = build_retriever(k=k).invoke(question)
    context, citations = format_context(docs)

    if no_cot or fewshot_n > 0:
        examples = NO_COT_EXAMPLES if no_cot else FEW_SHOT_EXAMPLES
        system = NO_COT_SYSTEM if no_cot else FEWSHOT_SYSTEM
        messages = _fewshot_messages(system, examples, fewshot_n, question, context)
    else:
        messages = RAG_PROMPT.format_messages(input=question, context=context)

    answer, usage = _invoke(messages)
    cited, unknown = resolve_citations(parse_citation_labels(answer), citations)

    if no_cot:
        mode = f"fewshot_no_cot_{fewshot_n}"
    elif fewshot_n > 0:
        mode = f"fewshot_{fewshot_n}"
    elif k is not None and k != settings.k:
        mode = f"rag_k{k}"
    else:
        mode = "rag_0shot"

    return {
        "answer": answer,
        "context": context,
        "citations": citations,
        "cited": cited,
        "unknown_citations": unknown,
        "retrieved": citations,
        "usage": usage,
        "latency_sec": round(time.perf_counter() - start, 4),
        "mode": mode,
    }


def answer_agent(
    question: str,
    max_steps: int = 4,
    max_total_chunks: int = 18,
) -> dict[str, Any]:
    from .agent_graph import build_agent_graph

    start = time.perf_counter()
    graph = build_agent_graph(
        max_steps=max_steps,
        max_total_chunks=max_total_chunks,
    )
    out = graph.invoke({"question": question})

    citations = out.get("citations", [])
    context = out.get("context", "")
    usage = dict(out.get("usage", _empty_usage()))
    usage.setdefault(
        "cost_cny",
        estimate_cost(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        ),
    )
    return {
        "answer": out.get("answer", ""),
        "context": context,
        "citations": citations,
        "retrieved": citations,
        "usage": usage,
        "latency_sec": round(time.perf_counter() - start, 4),
        "mode": "agent",
        "steps": out.get("steps", []),
        "iterations": out.get("iterations", 0),
    }
