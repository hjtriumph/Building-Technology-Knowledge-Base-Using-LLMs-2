from eval.case_loader import EvalCase, load_cases, resolve_evidence_chunks


def test_cases_schema_and_counts():
    cases = load_cases("eval/cases.jsonl")
    assert len(cases) == 30
    assert len({c.id for c in cases}) == 30
    assert sum(c.type == "unanswerable" for c in cases) == 8
    assert all(c.evidence for c in cases if c.type != "unanswerable")


def test_resolve_evidence_matches_normalized_quote():
    case = EvalCase(
        id="x",
        question="q",
        type="numeric",
        difficulty="easy",
        gold_answer="a",
        key_points=[],
        evidence=[
            {
                "source_substring": "a.pdf",
                "page": 1,
                "quote": "92.75 s to 45.80 s",
            }
        ],
        notes="",
    )
    idx = [
        {
            "chunk_id": "c1",
            "source_basename": "a.pdf",
            "page": 0,
            "text": "simulation time from 92.75 s to\n45.80 s",
        }
    ]
    assert resolve_evidence_chunks(case, idx) == {"c1"}
