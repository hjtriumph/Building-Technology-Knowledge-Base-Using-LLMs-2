from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

_LABEL_RE = re.compile(r"\[C(\d+)\]")


def format_context(docs: list[Document]) -> tuple[str, list[dict[str, Any]]]:
    """Format retrieved chunks with [C1]..[Cn] labels and return the mapping."""
    parts: list[str] = []
    citations: list[dict[str, Any]] = []

    for i, doc in enumerate(docs or [], start=1):
        md = doc.metadata or {}
        label = f"C{i}"
        source = str(md.get("source", ""))
        basename = md.get("source_basename") or Path(source).name or "unknown"
        raw_page = md.get("page")
        # PyPDFLoader pages are 0-based; citations use human-readable 1-based pages.
        page = None if raw_page is None else int(raw_page) + 1
        chunk_id = md.get("chunk_id")

        header = f"[{label}] Source: {basename}"
        if page is not None:
            header += f" | page={page}"

        parts.append(header + "\n" + (doc.page_content or ""))
        citations.append(
            {
                "label": label,
                "source": source,
                "source_basename": basename,
                "page": page,
                "chunk_id": chunk_id,
                "snippet": (doc.page_content or "")[:200],
            }
        )

    return "\n\n---\n\n".join(parts).strip(), citations


def parse_citation_labels(answer: str) -> list[str]:
    """Extract unique [Cn] labels in first-appearance order."""
    seen: set[str] = set()
    labels: list[str] = []
    for match in _LABEL_RE.finditer(answer or ""):
        label = f"C{int(match.group(1))}"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def resolve_citations(
    labels: list[str],
    citations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_label = {c.get("label"): c for c in citations or []}
    resolved = [by_label[label] for label in labels if label in by_label]
    unknown = [label for label in labels if label not in by_label]
    return resolved, unknown
