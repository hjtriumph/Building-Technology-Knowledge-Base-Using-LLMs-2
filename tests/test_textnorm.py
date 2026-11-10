from src.textnorm import normalize_text, contains_quote


def test_normalize_ligatures_and_dashes():
    assert "fi" in normalize_text("eﬃcient")
    assert normalize_text("24–48 h") == normalize_text("24-48 h")


def test_contains_quote_tolerates_hyphenation_and_whitespace():
    hay = "the simu-\nlation time was reduced from 92.75 s"
    assert contains_quote(hay, "simulation time was reduced from 92.75 s")


def test_contains_quote_is_case_insensitive():
    assert contains_quote("NRMSE improved", "nrmse improved")
