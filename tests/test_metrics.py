from eval.metrics import (
    bootstrap_mean_ci,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_diff,
    recall_at_k,
)


def test_recall_mrr_ndcg_basic():
    rel = {"a", "b"}
    got = ["x", "a", "b"]
    assert recall_at_k(rel, got, 2) == 0.5
    assert recall_at_k(rel, got, 3) == 1.0
    assert mrr_at_k(rel, got, 3) == 0.5
    assert 0 < ndcg_at_k(rel, got, 3) <= 1.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k({"a"}, ["a"], 5) == 1.0


def test_bootstrap_ci_contains_mean():
    lo, hi = bootstrap_mean_ci([1, 1, 1, 1], n_boot=200)
    assert lo == hi == 1.0


def test_paired_diff_reports_interval():
    out = paired_bootstrap_diff([1, 1, 1], [0, 0, 0], n_boot=200)
    assert out["mean_diff"] == 1.0
    assert out["ci_low"] <= 1.0 <= out["ci_high"]


def _row(case_id, method, value, refused=False, case_type="numeric"):
    return {
        "case_id": case_id,
        "method": method,
        "type": case_type,
        "completeness": value,
        "refused": refused,
    }


def test_question_level_means_averages_repeats():
    from eval.metrics import question_level_means

    rows = [
        _row("c1", "a", 1.0),
        _row("c1", "a", 0.5),
        _row("c2", "a", 1.0),
    ]
    assert question_level_means(rows, "completeness") == [0.75, 1.0]


def test_question_level_means_excludes_refused():
    from eval.metrics import question_level_means

    rows = [
        _row("c1", "a", 1.0),
        _row("c1", "a", 0.0, refused=True),
    ]
    assert question_level_means(rows, "completeness", exclude_refused=True) == [1.0]


def test_paired_comparisons_align_cases():
    from eval.metrics import paired_comparisons

    rows = [
        _row("c1", "a", 1.0),
        _row("c2", "a", 1.0),
        _row("c1", "b", 0.0),
        _row("c2", "b", 1.0),
    ]
    out = paired_comparisons(rows, baseline="b", compares=["a"], metric="completeness")
    assert out[0]["n"] == 2
    assert out[0]["mean_diff"] == 0.5
