from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Sequence

from src.rag_chain import build_retriever, get_vectorstore

from .case_loader import load_cases, resolve_evidence_chunks
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k

KS = (1, 3, 6, 10)


def build_chunk_index() -> list[dict[str, Any]]:
    vs = get_vectorstore()
    data = vs._collection.get(include=["documents", "metadatas"])
    return [
        {
            "chunk_id": chunk_id,
            "source_basename": (meta or {}).get("source_basename", ""),
            "page": (meta or {}).get("page"),
            "text": text or "",
        }
        for chunk_id, text, meta in zip(
            data["ids"], data["documents"], data["metadatas"]
        )
    ]


def evaluate(
    cases: Sequence[Any],
    chunk_index: list[dict[str, Any]],
    search_type: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    retriever = build_retriever(k=max(KS), search_type=search_type)
    rows: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for case in cases:
        if case.type == "unanswerable":
            continue

        relevant = resolve_evidence_chunks(case, chunk_index)
        if not relevant:
            unresolved.append(case.id)
            continue

        docs = retriever.invoke(case.question)
        retrieved = [str((d.metadata or {}).get("chunk_id")) for d in docs]
        hits = set(retrieved) & relevant

        row: dict[str, Any] = {
            "case_id": case.id,
            "type": case.type,
            "search_type": search_type,
            "relevant_chunks": len(relevant),
            "retrieved_chunks": len(retrieved),
        }
        for k in KS:
            row[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
        row["mrr@10"] = mrr_at_k(relevant, retrieved, 10)
        row["ndcg@10"] = ndcg_at_k(relevant, retrieved, 10)
        row["context_precision"] = (
            round(len(hits) / len(retrieved), 4) if retrieved else 0.0
        )
        row["context_recall"] = round(len(hits) / len(relevant), 4)
        rows.append(row)

    return rows, unresolved


def summarize(rows: list[dict[str, Any]], search_type: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "search_type": search_type,
        "n_cases": len(rows),
    }
    keys = [f"recall@{k}" for k in KS] + [
        "mrr@10",
        "ndcg@10",
        "context_precision",
        "context_recall",
    ]
    for key in keys:
        out[key] = round(statistics.mean([float(r[key]) for r in rows]), 4) if rows else 0.0
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval benchmark")
    ap.add_argument("--out-dir", default="results/retrieval")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Fail if Recall@k is not monotonic non-decreasing in k",
    )
    args = ap.parse_args()

    cases = load_cases("eval/cases.jsonl")
    chunk_index = build_chunk_index()
    print(f"[retrieval] cases={len(cases)} chunks={len(chunk_index)}")

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for search_type in ("mmr", "similarity"):
        rows, unresolved = evaluate(cases, chunk_index, search_type)
        if unresolved:
            print(f"[retrieval] WARNING unresolved evidence: {unresolved}")
        summary = summarize(rows, search_type)
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"[retrieval] {search_type:10s} "
            f"R@1={summary['recall@1']:.3f} "
            f"R@6={summary['recall@6']:.3f} "
            f"R@10={summary['recall@10']:.3f} "
            f"MRR={summary['mrr@10']:.3f} "
            f"nDCG={summary['ndcg@10']:.3f}"
        )

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "summary.csv", summaries)
    write_csv(out_dir / "detail.csv", all_rows)
    print(f"[retrieval] saved {out_dir / 'summary.csv'}")

    if args.check:
        for summary in summaries:
            recalls = [summary[f"recall@{k}"] for k in KS]
            if recalls != sorted(recalls):
                raise SystemExit(f"non-monotonic recall for {summary['search_type']}: {recalls}")
        print("[retrieval] recall monotonicity OK")


if __name__ == "__main__":
    main()
