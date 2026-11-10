from __future__ import annotations

import os

_DEFAULT_INPUT_PER_M = 2.0
_DEFAULT_OUTPUT_PER_M = 8.0
_DEFAULT_CACHE_HIT_PER_M = 0.2


def _rate(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_hit_tokens: int = 0,
) -> float:
    """Estimate CNY cost from token counts using env-configured rates per 1M."""
    input_rate = _rate("PRICE_INPUT_PER_M", _DEFAULT_INPUT_PER_M)
    output_rate = _rate("PRICE_OUTPUT_PER_M", _DEFAULT_OUTPUT_PER_M)
    cache_rate = _rate("PRICE_CACHE_HIT_PER_M", _DEFAULT_CACHE_HIT_PER_M)

    cache_hit = max(0, min(int(cache_hit_tokens or 0), int(input_tokens or 0)))
    fresh_input = max(0, int(input_tokens or 0) - cache_hit)

    cost = (
        fresh_input / 1_000_000 * input_rate
        + cache_hit / 1_000_000 * cache_rate
        + max(0, int(output_tokens or 0)) / 1_000_000 * output_rate
    )
    return round(cost, 4)
