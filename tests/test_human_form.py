import csv

from eval.human_form import build_html

FIELDS = [
    "sample_id",
    "question",
    "gold_answer",
    "answer",
    "citations",
    "completeness_human",
    "faithfulness_human",
    "notes",
]


def test_build_html_embeds_rows_and_escapes_script(tmp_path):
    csv_path = tmp_path / "h.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s001",
                "question": "What is X?",
                "gold_answer": "Gold",
                "answer": "Answer",
                "citations": "[C1] x.pdf p.2: evil </script><script>alert(1)</script>",
                "completeness_human": "",
                "faithfulness_human": "",
                "notes": "",
            }
        )

    out = tmp_path / "form.html"
    build_html(str(csv_path), str(out))
    text = out.read_text(encoding="utf-8")

    assert "s001" in text
    assert "What is X?" in text
    assert "__DATA__" not in text
    assert "<\\/script>" in text
