# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        mapping = {
            "胃痛反酸": [1.0, 0.0, 0.0],
            "胃痛反酸常见于胃炎或胃食管反流。": [1.0, 0.0, 0.0],
            "咳嗽发热需要警惕肺部感染。": [0.0, 1.0, 0.0],
            "皮疹瘙痒常见于过敏性皮炎。": [0.0, 0.0, 1.0],
        }
        return mapping.get(str(text), [0.1, 0.1, 0.1])


def _doc(text: str, *, department: str, source_file: str, row: int):
    from langchain.schema import Document  # type: ignore

    return Document(
        page_content=text,
        metadata={
            "department": department,
            "title": text,
            "row": row,
            "source_file": source_file,
            "chunk_id": f"{source_file}:{row}:0",
        },
    )


def _build_store(tmp_path: Path):
    from app.rag.faiss_store import FaissHNSWStore

    return FaissHNSWStore(
        persist_dir=tmp_path / "kb_store",
        embedding_function=_FakeEmbeddings(),
        collection_name="pytest_kb",
    )


def test_faiss_store_roundtrip_add_search_and_reload(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    store.add_documents(
        [
            _doc("胃痛反酸常见于胃炎或胃食管反流。", department="内科", source_file="a.md", row=1),
            _doc("咳嗽发热需要警惕肺部感染。", department="呼吸科", source_file="b.md", row=2),
        ]
    )
    store.persist()

    reloaded = _build_store(tmp_path)
    hits = reloaded.similarity_search_with_score("胃痛反酸", k=1)

    assert reloaded.count() == 2
    assert len(hits) == 1
    assert hits[0][0].page_content == "胃痛反酸常见于胃炎或胃食管反流。"
    assert hits[0][0].metadata["source_file"] == "a.md"
    assert isinstance(hits[0][1], float)


def test_faiss_store_search_respects_department_filter(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    store.add_documents(
        [
            _doc("胃痛反酸常见于胃炎或胃食管反流。", department="内科", source_file="a.md", row=1),
            _doc("皮疹瘙痒常见于过敏性皮炎。", department="皮肤科", source_file="c.md", row=3),
        ]
    )
    store.persist()

    hits = store.similarity_search_with_score("皮疹瘙痒常见于过敏性皮炎。", k=5, filter={"department": "皮肤科"})

    assert len(hits) == 1
    assert hits[0][0].metadata["department"] == "皮肤科"
    assert hits[0][0].metadata["source_file"] == "c.md"


def test_faiss_store_get_documents_returns_metadata_rows(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    store.add_documents(
        [
            _doc("胃痛反酸常见于胃炎或胃食管反流。", department="内科", source_file="a.md", row=1),
            _doc("咳嗽发热需要警惕肺部感染。", department="呼吸科", source_file="b.md", row=2),
        ]
    )
    store.persist()

    rows = store.get_documents(where={"department": "呼吸科"})

    assert rows["documents"] == ["咳嗽发热需要警惕肺部感染。"]
    assert rows["metadatas"][0]["department"] == "呼吸科"
    assert rows["metadatas"][0]["row"] == 2
