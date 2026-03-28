# -*- coding: utf-8 -*-

from __future__ import annotations

from tests.conftest import FakeRagStore


def test_retrieve_returns_empty_when_store_is_empty(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    monkeypatch.setattr(rag_core, "get_vectordb", lambda: FakeRagStore(count=0))
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)

    assert rag_core.retrieve("咳嗽", top_k=3) == []
