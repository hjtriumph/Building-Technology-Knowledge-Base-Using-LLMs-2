from eval.case_loader import load_cases
from src import prompts


def test_demo_questions_do_not_match_test_cases():
    cases = load_cases("eval/cases.jsonl")
    for ex in prompts.COT_EXAMPLES:
        for c in cases:
            assert ex["question"].strip().lower() != c.question.strip().lower()


def test_examples_come_from_new_documents():
    allowed = {
        "Automated building energy modeling",
        "Economic viability",
        "Energetic comparison",
        "Experimental analysis of airborne",
        "HeatShift-HP",
        "Improving heat demand forecasting",
        "Operational regime dependence",
    }
    for ex in prompts.COT_EXAMPLES:
        assert any(a in ex["context"] for a in allowed)


def test_no_cot_examples_have_short_answers():
    assert all(len(ex["answer"]) <= 300 for ex in prompts.NO_COT_EXAMPLES)
