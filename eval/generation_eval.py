from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI

from src.citations import parse_citation_labels
from src.config import settings
from src.prompts import (
    CITATION_SYSTEM,
    FAITHFULNESS_SYSTEM,
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
)

from .cache import ResponseCache
from .case_loader import EvalCase

_REFUSAL_LITERAL = "insufficient_context"
_REFUSAL_PHRASES = (
    "insufficient context",
    "not enough information",
    "cannot determine",
    "cannot be determined",
    "cannot answer",
    "no evidence",
    "不足以确定",
    "不足以回答",
    "资料不足",
    "没有足够",
    "无法确定",
    "无法回答",
)


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def detect_refusal(answer: str) -> bool:
    text = (answer or "").casefold()
    if _REFUSAL_LITERAL in text:
        return True
    return any(phrase in text for phrase in _REFUSAL_PHRASES)


@lru_cache(maxsize=1)
def get_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.judge_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.judge_temperature,
        max_retries=settings.max_retries,
    )


def _parse_response(content: Any) -> dict[str, Any]:
    data = _extract_json(content if isinstance(content, str) else str(content))
    if not data:
        return {"error": "judge_parse_failed", "raw": str(content)[:500]}
    return data


def _call_judge(
    messages: list[Any],
    cache: ResponseCache | None,
    key_parts: dict[str, Any],
    llm: Any = None,
) -> dict[str, Any]:
    model = llm or get_judge_llm()

    def fn() -> dict[str, Any]:
        response = model.invoke(messages)
        return _parse_response(getattr(response, "content", response))

    if cache is None:
        return fn()
    return cache.get_or_call(key_parts, fn)


def _clamp(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except Exception:
        return 0.0


def judge_quality(
    case: EvalCase,
    answer: str,
    llm: Any = None,
    cache: ResponseCache | None = None,
) -> dict[str, Any]:
    messages = [
        ("system", JUDGE_SYSTEM),
        (
            "human",
            JUDGE_USER_TEMPLATE.format(
                question=case.question,
                gold_answer=case.gold_answer,
                key_points=json.dumps(case.key_points, ensure_ascii=False),
                answer=answer,
            ),
        ),
    ]
    key_parts = {
        "kind": "quality",
        "model": settings.judge_model,
        "case_id": case.id,
        "answer": answer,
    }
    data = _call_judge(messages, cache, key_parts, llm=llm)
    return {
        "completeness": _clamp(data.get("completeness")),
        "answer_relevance": _clamp(data.get("answer_relevance")),
        "judge_reason": str(data.get("reason", data.get("error", ""))),
    }


def judge_faithfulness(
    answer: str,
    context: str,
    llm: Any = None,
    cache: ResponseCache | None = None,
) -> dict[str, Any]:
    messages = [
        ("system", FAITHFULNESS_SYSTEM),
        ("human", f"Context:\n{context}\n\nAnswer:\n{answer}"),
    ]
    key_parts = {
        "kind": "faithfulness",
        "model": settings.judge_model,
        "answer": answer,
        "context": context,
    }
    data = _call_judge(messages, cache, key_parts, llm=llm)

    claims = data.get("claims", [])
    total = int(data.get("total_claims", len(claims)) or 0)
    supported = int(data.get("supported_claims", 0) or 0)
    if total <= 0:
        total = len(claims)
    if total <= 0:
        total = 1
    score = _clamp(supported / total)
    return {
        "supported_claims": supported,
        "total_claims": total,
        "score": score,
        "claims": claims,
    }


def judge_citations(
    answer: str,
    citations: list[dict[str, Any]],
    llm: Any = None,
    cache: ResponseCache | None = None,
) -> dict[str, Any]:
    cited_labels = parse_citation_labels(answer)
    context = "\n\n".join(
        f"[{c.get('label')}] {c.get('source_basename')} p.{c.get('page')}\n{c.get('snippet', '')}"
        for c in citations
    )
    messages = [
        ("system", CITATION_SYSTEM),
        (
            "human",
            f"Answer:\n{answer}\n\nCitations:\n{context}\n\n"
            f"Cited labels: {json.dumps(cited_labels)}",
        ),
    ]
    key_parts = {
        "kind": "citations",
        "model": settings.judge_model,
        "answer": answer,
        "labels": cited_labels,
    }
    data = _call_judge(messages, cache, key_parts, llm=llm)

    supported = [str(x) for x in data.get("supported_labels", [])]
    known = {c.get("label") for c in citations}
    valid_supported = [x for x in supported if x in known]
    precision = (
        round(len(valid_supported) / len(cited_labels), 4) if cited_labels else 0.0
    )
    return {
        "cited_labels": cited_labels,
        "supported_labels": valid_supported,
        "precision": precision,
        "unsupported": data.get("unsupported", []),
    }
