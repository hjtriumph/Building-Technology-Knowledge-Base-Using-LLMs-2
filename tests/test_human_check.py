import csv
import json

from eval.human_check import compute_agreement


def test_agreement_perfect(tmp_path):
    human = tmp_path / "h.csv"
    with human.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "completeness_human", "faithfulness_human"]
        )
        writer.writeheader()
        writer.writerow(
            {"sample_id": "s1", "completeness_human": 1, "faithfulness_human": 1}
        )
    key = tmp_path / "k.json"
    key.write_text(
        json.dumps({"s1": {"completeness": 1.0, "faithfulness": 1.0}}),
        encoding="utf-8",
    )
    out = compute_agreement(str(human), str(key))
    assert out["exact"] == 1.0
    assert out["n"] == 1


def test_agreement_handles_utf8_bom(tmp_path):
    human = tmp_path / "h.csv"
    with human.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "completeness_human", "faithfulness_human"]
        )
        writer.writeheader()
        writer.writerow(
            {"sample_id": "s1", "completeness_human": 1, "faithfulness_human": 1}
        )
    key = tmp_path / "k.json"
    key.write_text(
        json.dumps({"s1": {"completeness": 1.0, "faithfulness": 1.0}}),
        encoding="utf-8",
    )
    out = compute_agreement(str(human), str(key))
    assert out["n"] == 1
