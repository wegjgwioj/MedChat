# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import app.rag.ingest_kb as ingest_kb


@dataclass
class _FakeEmbeddingInfo:
    provider_used: str = "fake"
    model_name: str = "fake-embed"
    device: str = "cpu"


class _FakeEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeOpenSearchStore:
    def __init__(self) -> None:
        self.batches = []

    def bulk_upsert(self, docs):
        payload = [dict(doc) for doc in docs]
        self.batches.append(payload)
        return len(payload)

    def count(self) -> int:
        return sum(len(batch) for batch in self.batches)


class _ForbiddenFaissStore:
    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("Faiss store should not be used when RAG_BACKEND=opensearch")


def test_build_and_persist_store_uses_opensearch_bulk_and_preserves_metadata(tmp_path: Path, monkeypatch):
    kb_dir = tmp_path / "kb"
    persist_dir = tmp_path / "persist"
    kb_dir.mkdir()
    persist_dir.mkdir()
    (kb_dir / "qa.csv").write_text("ignored", encoding="utf-8")

    fake_store = _FakeOpenSearchStore()

    monkeypatch.setenv("RAG_BACKEND", "opensearch")
    monkeypatch.setenv("OPENSEARCH_URL", "http://localhost:9200")
    monkeypatch.setenv("RAG_INGEST_BATCH_SIZE", "1")
    monkeypatch.setattr(ingest_kb, "make_embeddings", lambda: (_FakeEmbedder(), _FakeEmbeddingInfo()))
    monkeypatch.setattr(ingest_kb, "get_opensearch_store", lambda: fake_store)
    monkeypatch.setattr(ingest_kb, "FaissHNSWStore", _ForbiddenFaissStore)
    monkeypatch.setattr(
        ingest_kb,
        "_iter_csv_docs",
        lambda *args, **kwargs: iter(
            [
                (
                    7,
                    ingest_kb.RawDoc(
                        text="发热伴咽痛，建议先补液休息。",
                        metadata={
                            "source": "qa.csv",
                            "source_file": "qa.csv",
                            "department": "内科",
                            "title": "感冒问答",
                            "row": 7,
                        },
                    ),
                )
            ]
        ),
    )

    count = ingest_kb.build_and_persist_store(
        kb_dir=kb_dir,
        persist_dir=persist_dir,
        collection_name="medical_kb",
        chunk_size=800,
        chunk_overlap=100,
    )

    assert count == 1
    assert len(fake_store.batches) == 1

    doc = fake_store.batches[0][0]
    assert doc["text"] == "发热伴咽痛，建议先补液休息。"
    assert doc["embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert doc["department"] == "内科"
    assert doc["source_file"] == "qa.csv"
    assert doc["row"] == 7
    assert doc["chunk_id"] == "qa.csv:7:0"
