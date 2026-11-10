import random
import subprocess
import sys
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import src.ingest as ingest


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.rng = random.Random(42)

    def _vec(self):
        return [self.rng.uniform(-0.2, 0.3) for _ in range(1024)]

    def embed_documents(self, texts):
        return [self._vec() for _ in texts]

    def embed_query(self, text):
        return self._vec()


def test_large_collection_persists_and_reopens(tmp_path, monkeypatch):
    persist_dir = tmp_path / "知识库目录"
    persist_dir.mkdir()
    monkeypatch.setattr(
        ingest,
        "settings",
        SimpleNamespace(
            chroma_dir=str(persist_dir),
            collection_name="kkk",
            embedding_model="fake",
        ),
    )
    monkeypatch.setattr(ingest, "HuggingFaceEmbeddings", lambda model_name=None: FakeEmbeddings())

    chunks = [
        Document(
            page_content=f"doc {i} occupancy heating",
            metadata={"chunk_id": f"id{i}", "source_basename": "a.pdf", "page": 1},
        )
        for i in range(1410)
    ]

    vs = ingest.build_vectorstore(chunks, reset=True)
    assert vs._collection.count() == 1410
    del vs

    reader = """
import random, sys
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
class Fake(Embeddings):
    def __init__(self):
        self.rng = random.Random(42)
    def _vec(self):
        return [self.rng.uniform(-0.2, 0.3) for _ in range(1024)]
    def embed_documents(self, texts):
        return [self._vec() for _ in texts]
    def embed_query(self, text):
        return self._vec()
vs = Chroma(persist_directory=sys.argv[1], embedding_function=Fake(), collection_name="kkk")
print(vs._collection.count())
print(len(vs.similarity_search("doc", k=1)))
"""
    result = subprocess.run(
        [sys.executable, "-c", reader, str(persist_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "1410" in result.stdout
