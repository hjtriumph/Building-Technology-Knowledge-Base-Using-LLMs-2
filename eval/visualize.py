from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED = "#C30D1E"
DARK = "#434343"
GRAY = "#9CA3AF"
LIGHT = "#F2F2F2"
BLUE = "#1F90CC"
GREEN = "#047857"
ORANGE = "#FF6C00"

RADAR_METRICS = [
    ("completeness_mean", "Completeness"),
    ("answer_relevance_mean", "Relevance"),
    ("faithfulness_mean", "Faithfulness"),
    ("citation_precision_mean", "Citation"),
    ("refusal_rate_unanswerable", "Correct refusal"),
]
DECK_METHODS = ["rag_0shot", "fewshot_5", "rag_k18", "agent"]
COLORS = {"rag_0shot": GRAY, "fewshot_5": BLUE, "rag_k18": GREEN, "agent": RED}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return str(path)


def plot_radar(summary: Sequence[dict[str, Any]], out: Path) -> str | None:
    rows = [r for r in summary if r.get("method") in DECK_METHODS]
    if not rows:
        return None

    angles = [i / len(RADAR_METRICS) * 2 * math.pi for i in range(len(RADAR_METRICS))]
    angles += angles[:1]
    fig = plt.figure(figsize=(8.2, 6.0))
    ax = plt.subplot(111, polar=True)
    for row in rows:
        values = [min(1.0, max(0.0, _f(row, key))) for key, _ in RADAR_METRICS]
        values += values[:1]
        color = COLORS.get(str(row["method"]), GRAY)
        ax.plot(angles, values, color=color, linewidth=2, label=str(row["method"]))
        ax.fill(angles, values, color=color, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label in RADAR_METRICS], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color=GRAY)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=9)
    return _save(fig, out / "radar.png")


def plot_by_type(by_type: Sequence[dict[str, Any]], out: Path) -> str | None:
    if not by_type:
        return None
    types = ["numeric", "single_hop", "multi_hop", "method"]
    arms = DECK_METHODS
    lookup = {
        (str(r.get("method")), str(r.get("type"))): _f(r, "completeness_mean")
        for r in by_type
    }
    x = list(range(len(types)))
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for i, arm in enumerate(arms):
        values = [lookup.get((arm, t), 0.0) for t in types]
        positions = [xi - 0.4 + width / 2 + i * width for xi in x]
        bars = ax.bar(positions, values, width, label=arm, color=COLORS.get(arm, GRAY))
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=7.5,
                color=DARK,
            )
    ax.set_xticks(x, types)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Completeness")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out / "by_type.png")


def plot_paired(paired: Sequence[dict[str, Any]], out: Path) -> str | None:
    rows = [r for r in paired if r.get("baseline") and r.get("compare")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = list(range(len(rows)))[::-1]
    lo_all = [_f(r, "ci_low") for r in rows]
    hi_all = [_f(r, "ci_high") for r in rows]
    left = min(lo_all + [0.0]) - 0.02
    right = max(hi_all + [0.0]) + 0.06
    for yi, row in zip(y, rows):
        diff = _f(row, "mean_diff")
        lo = _f(row, "ci_low")
        hi = _f(row, "ci_high")
        color = GREEN if lo > 0 else GRAY
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.5)
        ax.plot(diff, yi, "o", color=color, markersize=8)
        ax.text(hi + 0.01, yi, f"p={_f(row, 'p_value'):.3f}", va="center", fontsize=9, color=DARK)
    ax.axvline(0, color=DARK, linewidth=1, linestyle="--")
    ax.set_yticks(y, [f"{r['compare']} vs {r['baseline']}" for r in rows])
    ax.set_xlim(left, right)
    ax.set_xlabel("Completeness difference (95% paired bootstrap CI)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, out / "paired_ci.png")


def plot_quality_latency(summary: Sequence[dict[str, Any]], out: Path) -> str | None:
    arms = DECK_METHODS + ["fewshot_no_cot_3"]
    rows = [r for r in summary if r.get("method") in arms]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for row in rows:
        name = str(row["method"])
        color = COLORS.get(name, BLUE if name == "fewshot_no_cot_3" else GRAY)
        x = _f(row, "latency_sec_median")
        y = _f(row, "completeness_mean")
        ax.scatter(x, y, s=90, color=color, zorder=3)
        ax.annotate(name, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9, color=DARK)
    ax.set_xlabel("Median latency (s)")
    ax.set_ylabel("Completeness")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out / "quality_latency.png")


def plot_quality_cost_latency(summary: Sequence[dict[str, Any]], out: Path) -> str | None:
    arms = DECK_METHODS + ["fewshot_no_cot_3"]
    rows = [r for r in summary if r.get("method") in arms]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for row in rows:
        name = str(row["method"])
        x = _f(row, "latency_sec_median")
        y = _f(row, "completeness_mean")
        cost = _f(row, "cost_total_cny")
        size = 70 + cost * 420
        color = COLORS.get(name, GRAY)
        ax.scatter(x, y, s=size, color=color, alpha=0.75, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(name, (x, y), xytext=(7, 5), textcoords="offset points", fontsize=8, color=DARK)
    ax.set_xlabel("Median latency (s)")
    ax.set_ylabel("Completeness")
    ax.set_title("Bubble size = total benchmark cost (CNY)", fontsize=10, color=GRAY)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out / "quality_cost_latency.png")


def plot_abstention(summary: Sequence[dict[str, Any]], out: Path) -> str | None:
    rows = [r for r in summary if r.get("method") in DECK_METHODS]
    if not rows:
        return None
    arms = [str(r["method"]) for r in rows]
    correct = [_f(r, "refusal_rate_unanswerable") for r in rows]
    false = [_f(r, "false_refusal_rate_answerable") for r in rows]
    x = list(range(len(arms)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    b1 = ax.bar([i - width / 2 for i in x], correct, width, label="Correct refusal (unanswerable)", color=GREEN)
    b2 = ax.bar([i + width / 2 for i in x], false, width, label="False refusal (answerable)", color=ORANGE)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=9,
                color=DARK,
            )
    ax.set_xticks(x, arms)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out / "abstention.png")


def plot_retrieval(retrieval: Sequence[dict[str, Any]], out: Path) -> str | None:
    rows = [r for r in retrieval if r.get("search_type")]
    if not rows:
        return None
    labels = ["Recall@6", "Recall@10"]
    x = range(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, row in enumerate(rows):
        values = [_f(row, "recall@6"), _f(row, "recall@10")]
        color = RED if row.get("search_type") == "mmr" else GRAY
        bars = ax.bar([xi - width / 2 + i * width for xi in x], values, width,
                      label=str(row["search_type"]), color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=10,
                color=DARK,
            )
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Recall")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out / "retrieval.png")


def _img(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{path.name}" />'


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_dashboard(
    run_dir: Path,
    out: Path,
    summary: Sequence[dict[str, Any]],
    by_type: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    human: dict[str, Any] | None,
) -> str:
    best = max(summary, key=lambda r: _f(r, "completeness_mean")) if summary else {}
    best_cite = max(summary, key=lambda r: _f(r, "citation_precision_mean")) if summary else {}
    total_cost = sum(_f(r, "cost_total_cny") for r in summary)

    def metric_cell(row: dict[str, Any], key: str) -> str:
        method = str(row.get("method", ""))
        if method in {"llm", "llm_structured"} and key in {
            "faithfulness_mean",
            "citation_precision_mean",
        }:
            return "N/A"
        if method == "fewshot_no_cot_3" and key == "citation_precision_mean":
            return "N/A"
        return f"{_f(row, key):.3f}"

    summary_rows = [
        [
            r.get("method", ""),
            r.get("n_questions", ""),
            metric_cell(r, "completeness_mean"),
            metric_cell(r, "answer_relevance_mean"),
            metric_cell(r, "faithfulness_mean"),
            metric_cell(r, "citation_precision_mean"),
            f"{_f(r, 'latency_sec_median'):.1f}s",
            f"¥{_f(r, 'cost_total_cny'):.2f}",
            f"{_f(r, 'refusal_rate_unanswerable'):.2f} / {_f(r, 'false_refusal_rate_answerable'):.2f}",
        ]
        for r in summary
    ]
    by_type_rows = [
        [
            r.get("method", ""),
            r.get("type", ""),
            r.get("n_questions", ""),
            f"{_f(r, 'completeness_mean'):.3f}",
            f"{_f(r, 'faithfulness_mean'):.3f}",
        ]
        for r in by_type
        if str(r.get("type")) != "unanswerable" and _f(r, "n_questions", 1) > 0
    ]
    paired_rows = [
        [
            r.get("compare", ""),
            r.get("baseline", ""),
            f"{_f(r, 'mean_diff'):+.3f}",
            f"[{_f(r, 'ci_low'):+.3f}, {_f(r, 'ci_high'):+.3f}]",
            f"{_f(r, 'p_value'):.3f}",
        ]
        for r in paired
    ]

    human_html = ""
    if human:
        human_html = f"""
        <div class="cards">
          <div class="card"><div class="num">{human.get('exact', 0):.3f}</div><div class="label">Completeness exact agreement</div></div>
          <div class="card"><div class="num">{human.get('weighted_kappa', 0):.3f}</div><div class="label">Completeness weighted kappa</div></div>
          <div class="card"><div class="num">{human.get('faithfulness', {}).get('exact', 0):.3f}</div><div class="label">Faithfulness exact agreement</div></div>
        </div>"""

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark dashboard · {run_dir.name}</title>
<style>
  :root {{ --red:#C30D1E; --dark:#434343; --muted:#6b7280; --line:#e5e7eb; --bg:#f6f7f9; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--dark); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
  header {{ background:var(--red); color:#fff; padding:28px 32px; }}
  header h1 {{ margin:0 0 4px; font-size:24px; }}
  header p {{ margin:0; opacity:.85; }}
  main {{ max-width:1200px; margin:0 auto; padding:24px 28px 60px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px; }}
  .card .num {{ font-size:26px; font-weight:700; color:var(--red); }}
  .card .label {{ font-size:12px; color:var(--muted); }}
  section {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px; margin:18px 0; }}
  h2 {{ font-size:17px; margin:0 0 12px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  img {{ width:100%; height:auto; display:block; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ background:var(--red); color:#fff; text-align:left; padding:8px; }}
  td {{ border-bottom:1px solid var(--line); padding:7px 8px; }}
  tr:nth-child(even) td {{ background:#fafafa; }}
  @media (max-width:900px) {{ .cards,.grid2 {{ grid-template-columns:1fr 1fr; }} }}
</style></head>
<body>
<header><h1>Benchmark dashboard</h1><p>Run {run_dir.name} · 30 questions × 9 arms × 3 repeats · 810 rows</p></header>
<main>
  <div class="cards">
    <div class="card"><div class="num">{best.get('method','N/A')}</div><div class="label">Best completeness</div></div>
    <div class="card"><div class="num">{_f(best, 'completeness_mean'):.3f}</div><div class="label">Best completeness score</div></div>
    <div class="card"><div class="num">{best_cite.get('method','N/A')}</div><div class="label">Best citation precision</div></div>
    <div class="card"><div class="num">¥{total_cost:.2f}</div><div class="label">Total benchmark cost</div></div>
  </div>
  <section><h2>Quality profile</h2><div class="grid2">{_img(out/'radar.png')}{_img(out/'by_type.png')}</div></section>
  <section><h2>Paired comparisons (completeness)</h2>{_img(out/'paired_ci.png')}</section>
  <section><h2>Quality vs latency vs cost</h2><div class="grid2">{_img(out/'quality_latency.png')}{_img(out/'quality_cost_latency.png')}</div></section>
  <section><h2>Abstention and retrieval</h2><div class="grid2">{_img(out/'abstention.png')}{_img(out/'retrieval.png')}</div></section>
  {human_html}
  <section><h2>Summary</h2>{_table(['Method','N','Completeness','Relevance','Faithfulness','Citation','Latency','Cost','Refusal / false'], summary_rows)}</section>
  <section><h2>By question type</h2>{_table(['Method','Type','N','Completeness','Faithfulness'], by_type_rows)}</section>
  <section><h2>Paired tests</h2>{_table(['Compare','Baseline','Mean diff','95% CI','p'], paired_rows)}</section>
</main></body></html>"""
    path = out / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def build_visuals(
    run_dir: str | Path,
    out_dir: str | Path,
    retrieval_dir: str | Path = "results/retrieval",
) -> dict[str, str | None]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = _read_csv(run / "summary.csv")
    by_type = _read_csv(run / "by_type.csv")
    paired = _read_csv(run / "paired.csv")
    retrieval = _read_csv(Path(retrieval_dir) / "summary.csv")
    human_path = run / "human_agreement.json"
    human = json.loads(human_path.read_text(encoding="utf-8")) if human_path.exists() else None

    paths: dict[str, str | None] = {
        "radar": plot_radar(summary, out),
        "by_type": plot_by_type(by_type, out),
        "paired_ci": plot_paired(paired, out),
        "quality_latency": plot_quality_latency(summary, out),
        "quality_cost_latency": plot_quality_cost_latency(summary, out),
        "abstention": plot_abstention(summary, out),
        "retrieval": plot_retrieval(retrieval, out),
    }
    paths["dashboard"] = build_dashboard(run, out, summary, by_type, paired, human)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Build benchmark visuals and one-page dashboard")
    ap.add_argument("--run", default="", help="results/<run-id>; defaults to latest")
    ap.add_argument("--out", default="", help="output dir; defaults to <run>/visuals")
    ap.add_argument("--retrieval", default="results/retrieval")
    args = ap.parse_args()

    if args.run:
        run = Path(args.run)
    else:
        candidates = sorted(Path("results").glob("*/summary.csv"))
        if not candidates:
            raise SystemExit("no results/*/summary.csv found")
        run = candidates[-1].parent

    out = Path(args.out) if args.out else run / "visuals"
    paths = build_visuals(run, out, retrieval_dir=args.retrieval)
    for name, path in paths.items():
        if path:
            print(f"{name}: {path}")
    print(f"dashboard: {out / 'dashboard.html'}")


if __name__ == "__main__":
    main()
