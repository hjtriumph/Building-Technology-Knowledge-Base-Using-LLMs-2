import csv
import json
from pathlib import Path

from eval.visualize import build_visuals

FIELDS = [
    "method",
    "n_questions",
    "completeness_mean",
    "answer_relevance_mean",
    "faithfulness_mean",
    "citation_precision_mean",
    "latency_sec_median",
    "cost_total_cny",
    "refusal_rate_unanswerable",
    "false_refusal_rate_answerable",
]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_visuals_creates_expected_files(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    summary = []
    for i, method in enumerate(["rag_0shot", "fewshot_5", "rag_k18", "agent"]):
        summary.append(
            {
                "method": method,
                "n_questions": 22,
                "completeness_mean": 0.7 + i * 0.05,
                "answer_relevance_mean": 0.8,
                "faithfulness_mean": 0.9,
                "citation_precision_mean": 0.5 + i * 0.03,
                "latency_sec_median": 7 + i,
                "cost_total_cny": 0.5 + i,
                "refusal_rate_unanswerable": 0.9,
                "false_refusal_rate_answerable": 0.05,
            }
        )
    _write(run / "summary.csv", summary)
    _write(
        run / "by_type.csv",
        [
            {"method": m, "type": t, "n_questions": 2, "completeness_mean": 0.8}
            for m in ["rag_0shot", "fewshot_5", "rag_k18", "agent"]
            for t in ["numeric", "multi_hop"]
        ],
    )
    _write(
        run / "paired.csv",
        [
            {
                "baseline": "rag_0shot",
                "compare": "rag_k18",
                "metric": "completeness",
                "n": 22,
                "mean_diff": 0.09,
                "ci_low": 0.01,
                "ci_high": 0.18,
                "p_value": 0.025,
            }
        ],
    )
    retrieval = tmp_path / "retrieval"
    _write(
        retrieval / "summary.csv",
        [
            {"search_type": "mmr", "recall@6": 0.37, "recall@10": 0.53},
            {"search_type": "similarity", "recall@6": 0.46, "recall@10": 0.67},
        ],
    )

    out = tmp_path / "visuals"
    build_visuals(run, out, retrieval_dir=retrieval)

    expected = [
        "radar.png",
        "by_type.png",
        "paired_ci.png",
        "quality_latency.png",
        "quality_cost_latency.png",
        "abstention.png",
        "retrieval.png",
        "dashboard.html",
    ]
    for name in expected:
        assert (out / name).exists(), name
    assert "rag_0shot" in (out / "dashboard.html").read_text(encoding="utf-8")
