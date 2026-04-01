# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

import app.rag.ingest_kb as ingest_kb
import app.rag.rag_core as rag_core


class _FakeEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4]


def test_retrieve_defaults_to_opensearch_and_fails_without_url(monkeypatch):
    rag_core.clear_runtime_state()
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.delenv("OPENSEARCH_URL", raising=False)
    monkeypatch.setattr(rag_core, "_rewrite_query_slimming", lambda query: query)
    monkeypatch.setattr(rag_core, "get_embedder", lambda: (_FakeEmbedder(), None))
    monkeypatch.setattr(rag_core, "_lookup_cache", lambda **kwargs: (None, {"cache_hit": False, "cache_mode": None, "cache_backend": None}))

    with pytest.raises(RuntimeError, match="OPENSEARCH_URL"):
        rag_core.retrieve("发热咽痛", top_k=1, use_rerank=False)


def test_readme_alignment_rejects_non_opensearch_backend(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "faiss")

    with pytest.raises(RuntimeError, match="OpenSearch"):
        rag_core._env_backend()

    with pytest.raises(RuntimeError, match="OpenSearch"):
        ingest_kb._env_backend()


def test_faiss_store_module_is_removed():
    assert not Path("app/rag/faiss_store.py").exists()
