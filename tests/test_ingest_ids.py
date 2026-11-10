from langchain_core.documents import Document

from src.ingest import stable_chunk_id, split_documents


def test_same_chunk_same_id():
    assert stable_chunk_id("a.pdf", 1, "hello") == stable_chunk_id("a.pdf", 1, "hello")


def test_different_page_or_source_changes_id():
    assert stable_chunk_id("a.pdf", 1, "hello") != stable_chunk_id("a.pdf", 2, "hello")
    assert stable_chunk_id("a.pdf", 1, "hello") != stable_chunk_id("b.pdf", 1, "hello")


def test_split_sets_required_metadata_and_respects_size():
    docs = [Document(page_content="word " * 500, metadata={"source_basename": "a.pdf", "page": 3})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["chunk_id"] for c in chunks)
    assert all(c.metadata["source_basename"] == "a.pdf" for c in chunks)
    assert all(c.metadata["page"] == 3 for c in chunks)
