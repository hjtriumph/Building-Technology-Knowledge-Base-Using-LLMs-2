from src.agent_graph import _budget_exceeded


def test_budget_not_exceeded_below_limit():
    assert _budget_exceeded(["a", "b"], 18) is False


def test_budget_exceeded_at_or_above_limit():
    assert _budget_exceeded(["a"] * 18, 18) is True
