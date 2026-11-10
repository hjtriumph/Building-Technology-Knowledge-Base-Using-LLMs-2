from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.textnorm import contains_quote

CASE_TYPES = {"single_hop", "multi_hop", "method", "numeric", "unanswerable"}
DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class EvalCase:
    id: str
    question: str
    type: str
    difficulty: str
    gold_answer: str
    key_points: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


def _require(raw: dict[str, Any], key: str, case_id: str) -> Any:
    if key not in raw:
        raise ValueError(f"case {case_id}: missing field {key}")
    return raw[key]


def load_cases(path: str = "eval/cases.jsonl") -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()

    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            case_id = str(_require(raw, "id", f"line {line_no}"))
            if case_id in seen:
                raise ValueError(f"duplicate case id: {case_id}")
            seen.add(case_id)

            case_type = str(_require(raw, "type", case_id))
            if case_type not in CASE_TYPES:
                raise ValueError(f"case {case_id}: bad type {case_type}")

            difficulty = str(raw.get("difficulty", "medium"))
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"case {case_id}: bad difficulty {difficulty}")

            evidence = raw.get("evidence", [])
            if not isinstance(evidence, list):
                raise ValueError(f"case {case_id}: evidence must be a list")
            for ev in evidence:
                for key in ("source_substring", "quote"):
                    if not ev.get(key):
                        raise ValueError(f"case {case_id}: evidence missing {key}")

            if case_type == "unanswerable" and evidence:
                raise ValueError(f"case {case_id}: unanswerable case must have no evidence")
            if case_type != "unanswerable" and not evidence:
                raise ValueError(f"case {case_id}: answerable case needs evidence")

            cases.append(
                EvalCase(
                    id=case_id,
                    question=str(_require(raw, "question", case_id)),
                    type=case_type,
                    difficulty=difficulty,
                    gold_answer=str(_require(raw, "gold_answer", case_id)),
                    key_points=[str(x) for x in raw.get("key_points", [])],
                    evidence=evidence,
                    notes=str(raw.get("notes", "")),
                )
            )

    return cases


def resolve_evidence_chunks(case: EvalCase, chunk_index: list[dict[str, Any]]) -> set[str]:
    """Map evidence quotes to current chunk ids (source substring + page + quote)."""
    matched: set[str] = set()

    for ev in case.evidence:
        source_sub = str(ev.get("source_substring", "")).lower()
        page = ev.get("page")
        quote = str(ev.get("quote", ""))

        for chunk in chunk_index:
            basename = str(chunk.get("source_basename", "")).lower()
            if source_sub and source_sub not in basename:
                continue
            chunk_page = chunk.get("page")
            if page is not None:
                if chunk_page is None or int(chunk_page) + 1 != int(page):
                    continue
            if contains_quote(str(chunk.get("text", "")), quote):
                matched.add(str(chunk["chunk_id"]))

    return matched
