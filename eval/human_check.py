from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from .case_loader import load_cases


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def export_samples(
    raw_jsonl: str,
    out_csv: str,
    key_json: str,
    n: int = 25,
    seed: int = 42,
) -> None:
    rows = [
        json.loads(line)
        for line in Path(raw_jsonl).open(encoding="utf-8")
        if line.strip()
    ]
    gold = {c.id: c.gold_answer for c in load_cases("eval/cases.jsonl")}

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["type"])), []).append(row)

    rng = random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)
    keys = list(groups)
    rng.shuffle(keys)

    selected: list[dict[str, Any]] = []
    while len(selected) < n and any(groups[k] for k in keys):
        for key in keys:
            if groups[key] and len(selected) < n:
                selected.append(groups[key].pop())

    human_rows: list[dict[str, Any]] = []
    key_map: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(selected, start=1):
        sample_id = f"s{i:03d}"
        human_rows.append(
            {
                "sample_id": sample_id,
                "question": row["question"],
                "gold_answer": gold.get(str(row["case_id"]), ""),
                "answer": row["answer"],
                "citations": "\n".join(
                    f"[{c.get('label')}] {c.get('source_basename', '')} "
                    f"p.{c.get('page')}: {c.get('snippet', '')}"
                    for c in row.get("citations", [])
                ),
                "completeness_human": "",
                "faithfulness_human": "",
                "notes": "",
            }
        )
        key_map[sample_id] = {
            "method": row["method"],
            "case_id": row["case_id"],
            "type": row["type"],
            "completeness": row.get("completeness"),
            "faithfulness": row.get("faithfulness"),
        }

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(human_rows[0].keys()) if human_rows else [])
        writer.writeheader()
        writer.writerows(human_rows)

    Path(key_json).write_text(
        json.dumps(key_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[human] exported {len(human_rows)} samples to {out_csv}")


def _stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "exact": 0.0, "within_025": 0.0, "weighted_kappa": 0.0}

    n = len(pairs)
    exact = sum(1 for h, j in pairs if h == j) / n
    within = sum(1 for h, j in pairs if abs(h - j) <= 0.25) / n
    return {
        "n": n,
        "exact": round(exact, 4),
        "within_025": round(within, 4),
        "weighted_kappa": round(_quadratic_weighted_kappa(pairs), 4),
    }


def _quadratic_weighted_kappa(pairs: list[tuple[float, float]]) -> float:
    categories = sorted({round(h, 2) for h, _ in pairs} | {round(j, 2) for _, j in pairs})
    if len(categories) < 2:
        return 1.0 if all(h == j for h, j in pairs) else 0.0

    index = {value: i for i, value in enumerate(categories)}
    size = len(categories)
    observed = [[0.0] * size for _ in range(size)]
    for h, j in pairs:
        observed[index[round(h, 2)]][index[round(j, 2)]] += 1

    total = float(len(pairs))
    row_sums = [sum(row) / total for row in observed]
    col_sums = [sum(observed[i][j] for i in range(size)) / total for j in range(size)]
    expected = [[row_sums[i] * col_sums[j] for j in range(size)] for i in range(size)]

    numerator = 0.0
    denominator = 0.0
    denom = (size - 1) ** 2
    for i in range(size):
        for j in range(size):
            weight = ((i - j) ** 2) / denom
            numerator += weight * observed[i][j] / total
            denominator += weight * expected[i][j]

    if denominator == 0:
        return 1.0
    return 1.0 - numerator / denominator


def compute_agreement(human_csv: str, key_json: str) -> dict[str, Any]:
    key_map = json.loads(Path(key_json).read_text(encoding="utf-8"))
    completeness: list[tuple[float, float]] = []
    faithfulness: list[tuple[float, float]] = []

    with Path(human_csv).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            entry = key_map.get(row.get("sample_id", ""), {})
            hc = _num(row.get("completeness_human"))
            jc = _num(entry.get("completeness"))
            if hc is not None and jc is not None:
                completeness.append((hc, jc))

            hf = _num(row.get("faithfulness_human"))
            jf = _num(entry.get("faithfulness"))
            if hf is not None and jf is not None:
                faithfulness.append((hf, jf))

    out = _stats(completeness)
    out["faithfulness"] = _stats(faithfulness)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Human spot-check tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    export = sub.add_parser("export")
    export.add_argument("--raw", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--key", required=True)
    export.add_argument("--n", type=int, default=25)
    export.add_argument("--seed", type=int, default=42)

    agreement = sub.add_parser("agreement")
    agreement.add_argument("--human", required=True)
    agreement.add_argument("--key", required=True)

    args = ap.parse_args()
    if args.cmd == "export":
        export_samples(args.raw, args.out, args.key, n=args.n, seed=args.seed)
    else:
        print(json.dumps(compute_agreement(args.human, args.key), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
