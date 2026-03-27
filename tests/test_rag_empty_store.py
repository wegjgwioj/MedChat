# -*- coding: utf-8 -*-

from __future__ import annotations


def test_retrieve_returns_empty_when_collection_is_empty(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    class FakeCollection:
        def count(self) -> int:
            return 0

    class FakeVS:
        _collection = FakeCollection()

    monkeypatch.setattr(rag_core, "get_vectordb", lambda: FakeVS())
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)

    assert rag_core.retrieve("咳嗽", top_k=3) == []
