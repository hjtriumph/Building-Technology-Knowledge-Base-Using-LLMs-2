from langchain_core.messages import AIMessage

from eval.cache import ResponseCache
from eval.case_loader import EvalCase
from eval.generation_eval import detect_refusal, judge_quality


def test_cache_hit_avoids_second_call(tmp_path):
    cache = ResponseCache(tmp_path)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"v": 1}

    assert cache.get_or_call({"q": "x"}, fn)["v"] == 1
    assert cache.get_or_call({"q": "x"}, fn)["v"] == 1
    assert calls["n"] == 1


def test_disabled_cache_calls_every_time(tmp_path):
    cache = ResponseCache(tmp_path)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"v": calls["n"]}

    cache.get_or_call({"q": "x"}, fn, enabled=False)
    cache.get_or_call({"q": "x"}, fn, enabled=False)
    assert calls["n"] == 2


def test_detect_refusal_literal_and_semantic():
    assert detect_refusal("INSUFFICIENT_CONTEXT: no evidence")
    assert detect_refusal("现有资料不足以确定该问题的答案。")
    assert not detect_refusal("According to [C1], the answer is 4.91%.")


def test_judge_quality_parses_json():
    class FakeJudge:
        def invoke(self, messages):
            return AIMessage(
                content='{"completeness": 0.8, "answer_relevance": 0.9, "reason": "ok"}'
            )

    case = EvalCase(
        id="c1",
        question="q",
        type="numeric",
        difficulty="easy",
        gold_answer="4.91%",
        key_points=["4.91"],
        evidence=[],
        notes="",
    )
    out = judge_quality(case, "answer", llm=FakeJudge())
    assert out["completeness"] == 0.8
    assert out["answer_relevance"] == 0.9
