from eval.runner import aggregate, aggregate_by_type


def _row(method, score, typ="single_hop"):
    return {
        "method": method,
        "case_id": "c1",
        "type": typ,
        "completeness": score,
        "answer_relevance": score,
        "key_point_coverage": score,
        "faithfulness": score,
        "citation_precision": score,
        "latency_sec": 1.0,
        "usage": {"cost_cny": 0.01},
        "refused": False,
        "error": "",
    }


def test_aggregate_groups_by_method():
    out = aggregate([_row("a", 1.0), _row("a", 0.5), _row("b", 0.0)])
    by = {r["method"]: r for r in out}
    assert by["a"]["n"] == 2
    assert by["a"]["completeness_mean"] == 0.75


def test_aggregate_by_type_keeps_type_rows():
    out = aggregate_by_type([_row("a", 1.0, "multi_hop")])
    assert out[0]["type"] == "multi_hop"
