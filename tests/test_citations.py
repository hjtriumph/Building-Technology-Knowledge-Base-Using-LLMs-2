from langchain_core.documents import Document

from src.citations import format_context, parse_citation_labels, resolve_citations


def test_format_context_labels_in_order():
    docs = [
        Document(
            page_content="alpha",
            metadata={"source_basename": "a.pdf", "page": 1, "chunk_id": "x1"},
        ),
        Document(
            page_content="beta",
            metadata={"source_basename": "b.pdf", "page": 2, "chunk_id": "x2"},
        ),
    ]
    ctx, cites = format_context(docs)
    assert "[C1]" in ctx and "[C2]" in ctx
    assert cites[0]["label"] == "C1" and cites[0]["source_basename"] == "a.pdf"


def test_parse_and_resolve_keeps_unknown_labels():
    cites = [{"label": "C1", "source_basename": "a.pdf", "page": 1, "chunk_id": "x1"}]
    labels = parse_citation_labels("见 [C1] 与 [C9]")
    resolved, unknown = resolve_citations(labels, cites)
    assert [c["label"] for c in resolved] == ["C1"]
    assert unknown == ["C9"]
