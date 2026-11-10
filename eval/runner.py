from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import settings
from src.pipelines import answer_agent, answer_llm, answer_rag
from src.textnorm import normalize_text

from .cache import ResponseCache
from .case_loader import EvalCase, load_cases
from .generation_eval import (
    detect_refusal,
    judge_citations,
    judge_faithfulness,
    judge_quality,
)
from .metrics import bootstrap_mean_ci, question_level_means

METHODS: tuple[str, ...] = (
    "llm",
    "llm_structured",
    "rag_0shot",
    "fewshot_1",
    "fewshot_3",
    "fewshot_5",
    "fewshot_no_cot_3",
    "rag_k18",
    "agent",
)

SMOKE_METHODS: tuple[str, ...] = ("rag_0shot", "fewshot_1", "agent")


def call_method(method: str, question: str) -> dict[str, Any]:
    if method == "llm":
        return answer_llm(question, structured=False)
    if method == "llm_structured":
        return answer_llm(question, structured=True)
    if method == "rag_0shot":
        return answer_rag(question)
    if method == "fewshot_1":
        return answer_rag(question, fewshot_n=1)
    if method == "fewshot_3":
        return answer_rag(question, fewshot_n=3)
    if method == "fewshot_5":
        return answer_rag(question, fewshot_n=5)
    if method == "fewshot_no_cot_3":
        return answer_rag(question, fewshot_n=3, no_cot=True)
    if method == "rag_k18":
        return answer_rag(question, k=18)
    if method == "agent":
        return answer_agent(question, max_steps=4)
    raise ValueError(f"unsupported method: {method}")


def key_point_coverage(answer: str, key_points: Sequence[str]) -> float:
    if not key_points:
        return 0.0
    text = normalize_text(answer)
    matched = sum(1 for kp in key_points if normalize_text(kp) in text)
    return round(matched / len(key_points), 4)


def _mean(values: Sequence[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return round(statistics.pstdev(values), 4) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return round(ordered[idx], 4)


def run_benchmark(
    cases: Sequence[EvalCase],
    methods: Sequence[str] = METHODS,
    repeats: int = 3,
    out_dir: str | Path | None = None,
    reuse_responses: bool = False,
) -> list[dict[str, Any]]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir) if out_dir else Path("results") / run_id
    out.mkdir(parents=True, exist_ok=True)

    cache = ResponseCache(settings.cache_dir)
    rows: list[dict[str, Any]] = []

    for case in cases:
        print(f"[bench] {case.id} ({case.type})")
        for method in methods:
            for repeat in range(repeats):
                started = time.perf_counter()
                try:
                    key_parts = {
                        "kind": "generation",
                        "method": method,
                        "question": case.question,
                        "model": settings.deepseek_model,
                        "temperature": settings.temperature,
                    }
                    result = cache.get_or_call(
                        key_parts,
                        lambda: call_method(method, case.question),
                        enabled=reuse_responses,
                    )
                    answer = str(result.get("answer", ""))
                    context = str(result.get("context", ""))
                    citations = result.get("citations", []) or []

                    if case.type == "unanswerable":
                        quality = {"completeness": None, "answer_relevance": None, "judge_reason": ""}
                        faithfulness = {"score": None, "supported_claims": 0, "total_claims": 0}
                        citation = {"precision": None, "cited_labels": [], "supported_labels": []}
                    else:
                        quality = judge_quality(case, answer, cache=cache)
                        faithfulness = (
                            judge_faithfulness(answer, context, cache=cache)
                            if context
                            else {"score": None, "supported_claims": 0, "total_claims": 0}
                        )
                        citation = (
                            judge_citations(answer, citations, cache=cache)
                            if citations
                            else {
                                "precision": 0.0,
                                "cited_labels": [],
                                "supported_labels": [],
                            }
                        )

                    row = {
                        "run_id": run_id,
                        "case_id": case.id,
                        "question": case.question,
                        "type": case.type,
                        "difficulty": case.difficulty,
                        "method": method,
                        "repeat": repeat,
                        "answer": answer,
                        "citations": citations,
                        "usage": result.get("usage", {}),
                        "latency_sec": round(time.perf_counter() - started, 4),
                        "completeness": quality.get("completeness"),
                        "answer_relevance": quality.get("answer_relevance"),
                        "key_point_coverage": key_point_coverage(answer, case.key_points),
                        "faithfulness": faithfulness.get("score"),
                        "citation_precision": citation.get("precision"),
                        "refused": detect_refusal(answer),
                        "judge_reason": quality.get("judge_reason", ""),
                        "error": "",
                    }
                except Exception as exc:  # keep the run alive on API failures
                    row = {
                        "run_id": run_id,
                        "case_id": case.id,
                        "question": case.question,
                        "type": case.type,
                        "difficulty": case.difficulty,
                        "method": method,
                        "repeat": repeat,
                        "answer": "",
                        "citations": [],
                        "usage": {},
                        "latency_sec": round(time.perf_counter() - started, 4),
                        "completeness": None,
                        "answer_relevance": None,
                        "key_point_coverage": 0.0,
                        "faithfulness": None,
                        "citation_precision": None,
                        "refused": False,
                        "judge_reason": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
                print(
                    f"  {method:18s} rep={repeat} "
                    f"comp={row['completeness']} faith={row['faithfulness']} "
                    f"err={bool(row['error'])}"
                )
                time.sleep(0.1)

    raw_path = out / "raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = aggregate(rows)
    by_type = aggregate_by_type(rows)

    from .report import write_report

    write_report(rows, summary, by_type, out)
    print(f"[bench] saved {out}")
    return rows


def _metric_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method")), []).append(row)

    out: list[dict[str, Any]] = []
    for method, items in grouped.items():
        answerable = [r for r in items if r.get("type") != "unanswerable"]
        unanswerable = [r for r in items if r.get("type") == "unanswerable"]
        summary: dict[str, Any] = {
            "method": method,
            "n": len(answerable),
            "n_rows": len(items),
        }

        for key in (
            "completeness",
            "answer_relevance",
            "key_point_coverage",
            "faithfulness",
            "citation_precision",
        ):
            exclude_refused = key in {"faithfulness", "citation_precision"}
            values = question_level_means(
                answerable, key, exclude_refused=exclude_refused
            )
            summary[f"{key}_mean"] = _mean(values)
            summary[f"{key}_std"] = _std(values)
            summary[f"{key}_median"] = _median(values)
            summary[f"{key}_p95"] = _p95(values)
            lo, hi = bootstrap_mean_ci(values)
            summary[f"{key}_ci_low"] = lo
            summary[f"{key}_ci_high"] = hi
            summary[f"{key}_n"] = len(values)

        latency_values = _metric_values(answerable, "latency_sec")
        summary["latency_sec_mean"] = _mean(latency_values)
        summary["latency_sec_median"] = _median(latency_values)
        summary["latency_sec_p95"] = _p95(latency_values)
        summary["latency_sec_n"] = len(latency_values)
        summary["n_questions"] = len(
            question_level_means(answerable, "completeness")
        )

        costs = [
            float((r.get("usage") or {}).get("cost_cny", 0.0) or 0.0)
            for r in items
        ]
        summary["cost_total_cny"] = round(sum(costs), 4)
        summary["cost_mean_cny"] = _mean(costs)
        summary["refusal_rate_unanswerable"] = _mean(
            [1.0 if r.get("refused") else 0.0 for r in unanswerable]
        )
        summary["false_refusal_rate_answerable"] = _mean(
            [1.0 if r.get("refused") else 0.0 for r in answerable]
        )
        summary["error_rate"] = _mean(
            [1.0 if r.get("error") else 0.0 for r in items]
        )
        out.append(summary)

    order = {m: i for i, m in enumerate(METHODS)}
    out.sort(key=lambda r: order.get(str(r["method"]), 999))
    return out


def aggregate_by_type(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("method")), str(row.get("type")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (method, case_type), items in grouped.items():
        entry: dict[str, Any] = {
            "method": method,
            "type": case_type,
            "n_rows": len(items),
        }
        for metric in (
            "completeness",
            "answer_relevance",
            "key_point_coverage",
            "faithfulness",
            "citation_precision",
        ):
            exclude_refused = metric in {"faithfulness", "citation_precision"}
            values = question_level_means(
                items, metric, exclude_refused=exclude_refused
            )
            entry[f"{metric}_mean"] = _mean(values)
            entry[f"{metric}_median"] = _median(values)
            entry[f"{metric}_p95"] = _p95(values)

        latency_values = _metric_values(items, "latency_sec")
        entry["latency_sec_mean"] = _mean(latency_values)
        entry["latency_sec_median"] = _median(latency_values)
        entry["latency_sec_p95"] = _p95(latency_values)
        entry["n_questions"] = len(question_level_means(items, "completeness"))
        entry["refusal_rate"] = _mean(
            [1.0 if r.get("refused") else 0.0 for r in items]
        )
        out.append(entry)

    order = {m: i for i, m in enumerate(METHODS)}
    out.sort(key=lambda r: (order.get(str(r["method"]), 999), str(r["type"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the RAG benchmark")
    ap.add_argument("--num-cases", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse-responses", action="store_true")
    args = ap.parse_args()

    cases = load_cases("eval/cases.jsonl")
    methods = list(args.methods)
    repeats = args.repeats
    num_cases = args.num_cases

    if args.smoke:
        cases = cases[:2]
        repeats = 1
        methods = list(SMOKE_METHODS)

    if num_cases > 0:
        cases = cases[:num_cases]

    print(f"[bench] cases={len(cases)} methods={methods} repeats={repeats}")
    run_benchmark(
        cases,
        methods=methods,
        repeats=repeats,
        out_dir=args.out_dir or None,
        reuse_responses=args.reuse_responses,
    )


if __name__ == "__main__":
    main()
