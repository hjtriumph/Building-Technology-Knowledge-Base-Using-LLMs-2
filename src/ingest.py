from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings
from .textnorm import normalize_text

# Chroma 1.5.8 fails to persist the HNSW segment on Windows when the persist
# path contains non-ASCII characters and the default sync threshold (1000) is
# crossed mid-write. Keeping the threshold above the collection size avoids the
# broken incremental sync; the index is written correctly on close.
HNSW_COLLECTION_METADATA = {
    "hnsw:sync_threshold": 10000,
    "hnsw:batch_size": 1000,
}


def stable_chunk_id(source_basename: str, page: int | None, text: str) -> str:
    raw = f"{source_basename}|{page}|{normalize_text(text)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_documents(docs_dir: str) -> list[Document]:
    """Load PDF / txt / md documents under docs_dir."""
    docs: list[Document] = []

    for root, dirs, files in os.walk(docs_dir):
        dirs.sort()
        for fname in sorted(files):
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".pdf":
                loader = PyPDFLoader(path)
                pages = loader.load()
                title = ""
                for d in pages:
                    md = d.metadata or {}
                    title = title or str(md.get("title") or "").strip()
                for d in pages:
                    d.metadata.update(
                        {
                            "source": path,
                            "source_type": "pdf",
                            "source_basename": Path(path).name,
                            "doc_title": title or Path(path).stem,
                        }
                    )
                docs.extend(pages)

    for glob in ["**/*.txt", "**/*.md"]:
        loader = DirectoryLoader(
            docs_dir,
            glob=glob,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )
        text_docs = loader.load()
        for d in text_docs:
            src = str(d.metadata.get("source", ""))
            d.metadata.update(
                {
                    "source_type": "text",
                    "source_basename": Path(src).name if src else "unknown",
                    "doc_title": Path(src).stem if src else "unknown",
                }
            )
        docs.extend(text_docs)

    return docs


def _split_raw(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _dedupe_chunks(chunks: list[Document]) -> tuple[list[Document], int]:
    seen: set[str] = set()
    out: list[Document] = []
    duplicates = 0

    for chunk in chunks:
        md = chunk.metadata or {}
        basename = md.get("source_basename") or Path(str(md.get("source", ""))).name
        page = md.get("page")
        chunk_id = stable_chunk_id(basename, page, chunk.page_content or "")

        if chunk_id in seen:
            duplicates += 1
            continue
        seen.add(chunk_id)

        chunk.metadata.update(
            {
                "source_basename": basename,
                "page": page,
                "chunk_id": chunk_id,
                "chunk_index": len(out),
            }
        )
        out.append(chunk)

    return out, duplicates


def split_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    chunks, _ = _dedupe_chunks(_split_raw(docs, chunk_size, chunk_overlap))
    return chunks


def build_vectorstore(chunks: list[Document], reset: bool = False) -> Chroma:
    print(f"[ingest] 加载 embedding 模型: {settings.embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    os.makedirs(settings.chroma_dir, exist_ok=True)

    vs = Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
        collection_name=settings.collection_name,
        collection_metadata=HNSW_COLLECTION_METADATA,
    )

    if reset:
        print("[ingest] 清空现有 collection")
        try:
            vs.delete_collection()
        except Exception:
            pass
        vs = Chroma(
            persist_directory=settings.chroma_dir,
            embedding_function=embeddings,
            collection_name=settings.collection_name,
            collection_metadata=HNSW_COLLECTION_METADATA,
        )

    if not chunks:
        return vs

    texts = [c.page_content or "" for c in chunks]
    ids = [c.metadata["chunk_id"] for c in chunks]
    metadatas = [c.metadata or {} for c in chunks]

    print(f"[ingest] 向量化 {len(chunks)} 个 chunk")
    vectors = embeddings.embed_documents(texts)

    batch = 256
    for start in tqdm(range(0, len(chunks), batch), desc="upsert"):
        end = start + batch
        vs._collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end],
        )

    return vs


def ingest(reset: bool = False) -> dict:
    if not os.path.isdir(settings.docs_dir):
        raise FileNotFoundError(f"找不到文档目录: {settings.docs_dir}")

    print(f"[ingest] 从目录加载文档: {settings.docs_dir}")
    docs = load_documents(settings.docs_dir)
    if not docs:
        raise RuntimeError("没有找到任何文档！请把 PDF/MD/TXT 放到 ./data/docs")

    print(f"[ingest] 原始 Document 数量（PDF 为页数）: {len(docs)}")

    raw_chunks = _split_raw(docs, settings.chunk_size, settings.chunk_overlap)
    chunks, duplicate_ids = _dedupe_chunks(raw_chunks)
    print(f"[ingest] 切分后 chunk 数量: {len(chunks)}（去重 {duplicate_ids}）")

    vs = build_vectorstore(chunks, reset=reset)

    source_basenames = sorted({c.metadata["source_basename"] for c in chunks})
    report = {
        "document_count": len(source_basenames),
        "page_count": len(docs),
        "chunk_count": int(vs._collection.count()),
        "duplicate_ids": duplicate_ids,
        "embedding_model": settings.embedding_model,
        "collection_name": settings.collection_name,
        "source_basenames": source_basenames,
    }

    report_path = Path(settings.chroma_dir).parent / "ingest_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ingest] 向量库构建完成，chunk 总数: {report['chunk_count']}")
    print(f"[ingest] 报告: {report_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Chroma knowledge base")
    ap.add_argument("--reset", action="store_true", help="Delete the existing collection first")
    args = ap.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
