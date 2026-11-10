import json

from eval.human_check import export_samples


def test_export_includes_citations(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "method": "rag_0shot",
                "type": "numeric",
                "case_id": "c001",
                "question": "q",
                "answer": "a [C1]",
                "citations": [
                    {
                        "label": "C1",
                        "source_basename": "x.pdf",
                        "page": 2,
                        "snippet": "supporting snippet",
                    }
                ],
                "completeness": 1.0,
                "faithfulness": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "h.csv"
    key = tmp_path / "k.json"
    export_samples(str(raw), str(out), str(key), n=1)
    text = out.read_text(encoding="utf-8")
    assert "citations" in text
    assert "[C1]" in text
    assert "x.pdf" in text
