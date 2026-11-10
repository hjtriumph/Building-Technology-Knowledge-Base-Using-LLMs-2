from langchain_core.messages import AIMessage

from src import pipelines


class FakeLLM:
    def invoke(self, messages):
        return AIMessage(
            content="1) Conclusion\nok [C1]",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        )


def test_answer_llm_returns_usage(monkeypatch):
    monkeypatch.setattr(pipelines, "get_llm", lambda: FakeLLM())
    out = pipelines.answer_llm("q", structured=True)
    assert out["usage"]["input_tokens"] == 100
    assert out["usage"]["output_tokens"] == 20
    assert out["usage"]["cost_cny"] > 0


def test_answer_rag_returns_citations(monkeypatch):
    class FakeRetriever:
        def invoke(self, q):
            from langchain_core.documents import Document

            return [
                Document(
                    page_content="alpha",
                    metadata={"source_basename": "a.pdf", "page": 1, "chunk_id": "x1"},
                )
            ]

    monkeypatch.setattr(pipelines, "build_retriever", lambda k=None: FakeRetriever())
    monkeypatch.setattr(pipelines, "get_llm", lambda: FakeLLM())
    out = pipelines.answer_rag("q")
    assert out["citations"][0]["label"] == "C1"
    assert "[C1]" in out["answer"]
