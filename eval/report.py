from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import paired_comparisons


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def _plot_quality_latency(summary: Sequence[dict[str, Any]], path: Path) -> None:
    if not summary:
        return
    names = [str(r["method"]) for r in summary]
    comp = [float(r.get("completeness_mean", 0.0)) for r in summary]
    lat = [float(r.get("latency_sec_mean", 0.0)) for r in summary]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(lat, comp, s=60)
    for x, y, name in zip(lat, comp, names):
        ax.annotate(name, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean latency (s)")
    ax.set_ylabel("Mean completeness")
    ax.set_title("Quality vs latency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_abstention(summary: Sequence[dict[str, Any]], path: Path) -> None:
    if not summary:
        return
    names = [str(r["method"]) for r in summary]
    unans = [float(r.get("refusal_rate_unanswerable", 0.0)) for r in summary]
    false = [float(r.get("false_refusal_rate_answerable", 0.0)) for r in summary]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - 0.2 for i in x], unans, width=0.4, label="Correct refusal (unanswerable)")
    ax.bar([i + 0.2 for i in x], false, width=0.4, label="False refusal (answerable)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Abstention behaviour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    rows: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    by_type: Sequence[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    paired = paired_comparisons(
        rows, "rag_0shot", ["fewshot_5", "rag_k18", "agent"]
    )
    paired += paired_comparisons(rows, "rag_k18", ["agent"])

    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "by_type.csv", by_type)
    write_csv(out_dir / "paired.csv", paired)

    overall_cols = [
        ("method", "Method"),
        ("n_questions", "Questions"),
        ("n", "Rows"),
        ("completeness_mean", "Completeness"),
        ("answer_relevance_mean", "Relevance"),
        ("faithfulness_mean", "Faithfulness"),
        ("citation_precision_mean", "Citation P"),
        ("key_point_coverage_mean", "Key points"),
        ("latency_sec_median", "Latency p50 s"),
        ("latency_sec_p95", "Latency p95 s"),
        ("cost_total_cny", "Cost CNY"),
        ("refusal_rate_unanswerable", "Refusal (unans)"),
        ("false_refusal_rate_answerable", "False refusal"),
        ("error_rate", "Error rate"),
    ]
    paired_cols = [
        ("baseline", "Baseline"),
        ("compare", "Compare"),
        ("metric", "Metric"),
        ("n", "Pairs"),
        ("mean_diff", "Mean diff"),
        ("ci_low", "CI low"),
        ("ci_high", "CI high"),
        ("p_value", "p"),
    ]
    type_cols = [
        ("method", "Method"),
        ("type", "Type"),
        ("n_questions", "Questions"),
        ("n_rows", "Rows"),
        ("completeness_mean", "Completeness"),
        ("faithfulness_mean", "Faithfulness"),
        ("refusal_rate", "Refusal"),
        ("latency_sec_median", "Latency p50 s"),
    ]

    current = rows[0].get("run_id", "") if rows else ""
    report = f"""# Benchmark report

Run: {current}
Rows: {len(rows)}
Answerable quality means exclude unanswerable cases; refusal rates are reported separately.

## Overall

{_table(summary, overall_cols)}

## Paired comparisons (completeness, question-level means)

{_table(paired, paired_cols)}

## By question type

{_table(by_type, type_cols)}

## Notes

- Generation temperature: 0.2; judge temperature: 0.0; retrieval: MMR, k=6, fetch_k=20.
- Quality metrics average the three repeats per question first, then bootstrap over questions (n≈30), not rows.
- Paired comparisons use question-level completeness and a paired bootstrap against the stated baseline.
- Faithfulness and citation precision exclude refused answers. Closed-book arms have no retrieved context, so faithfulness is not applicable (N/A).
- Latency is reported as median and p95 because one request in the raw data hung for several hours and would distort the mean.
- LLM judge and the evaluated model are both DeepSeek; use the human spot-check table for calibration.
- Costs are estimates from token counts and configured rates, not provider invoices.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    _plot_quality_latency(summary, out_dir / "quality_latency.png")
    _plot_abstention(summary, out_dir / "abstention.png")
