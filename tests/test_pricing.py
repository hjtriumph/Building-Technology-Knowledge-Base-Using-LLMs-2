from src.pricing import estimate_cost


def test_cost_uses_env_rates(monkeypatch):
    monkeypatch.setenv("PRICE_INPUT_PER_M", "2.0")
    monkeypatch.setenv("PRICE_OUTPUT_PER_M", "8.0")
    assert round(estimate_cost(1_000_000, 1_000_000), 4) == 10.0


def test_zero_tokens_costs_zero():
    assert estimate_cost(0, 0) == 0.0
