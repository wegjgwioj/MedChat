# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import pytest

import app.rag.rag_core as rag_core


@dataclass
class _FakeEmbeddingInfo:
    device: str = "cpu"
    model_name: str = "fake-embed"


class _FakeStore:
    class config:
        index_name = "medical_kb"
        url = "http://localhost:9200"

    def count(self) -> int:
        return 42


def test_get_stats_reports_opensearch_backend(monkeypatch):
    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_BACKEND", "opensearch")
    monkeypatch.setenv("OPENSEARCH_URL", "http://localhost:9200")
    monkeypatch.setattr(rag_core, "get_embedder", lambda: (object(), _FakeEmbeddingInfo()))
    monkeypatch.setattr(rag_core, "get_opensearch_store", lambda: _FakeStore())

    stats = rag_core.get_stats()

    assert stats.backend == "opensearch"
    assert stats.collection == "medical_kb"
    assert stats.count == 42
    assert stats.persist_dir == "http://localhost:9200"
    assert stats.embed_model == "fake-embed"


def test_get_opensearch_store_raises_clear_error_when_url_missing(monkeypatch):
    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_BACKEND", "opensearch")
    monkeypatch.setenv("OPENSEARCH_URL", "")

    with pytest.raises(RuntimeError, match="OPENSEARCH_URL"):
        rag_core.get_opensearch_store()
