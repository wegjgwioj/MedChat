# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import app.rag.rag_core as rag_core


@dataclass
class _FakeEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _FakeStore:
    def search_hybrid(self, *, query: str, query_vector: List[float], top_k: int, department: str | None = None) -> Dict[str, Any]:
        return {
            "hits": [
                {
                    "_score": 3.2,
                    "_source": {
                        "chunk_id": "chunk-1",
                        "text": "发热伴咽痛，建议先补液休息。",
                        "source": "qa.csv",
                        "source_file": "qa.csv",
                        "department": "内科",
                        "title": "感冒问答",
                        "row": 7,
                    },
                    "_explanation_meta": {"bm25_score": 2.1, "knn_score": 0.88},
                }
            ],
            "meta": {
                "backend": "opensearch",
                "hybrid_mode": "app_rrf",
                "bm25_hits": 1,
                "knn_hits": 1,
                "fusion_hits": 1,
                "took_ms": 12,
                "index_name": "medical_kb",
            },
        }


def test_retrieve_uses_opensearch_hybrid_and_preserves_evidence_contract(monkeypatch):
    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_BACKEND", "opensearch")
    monkeypatch.setattr(rag_core, "_rewrite_query_slimming", lambda query: query)
    monkeypatch.setattr(rag_core, "get_embedder", lambda: (_FakeEmbedder(), None))
    monkeypatch.setattr(rag_core, "get_opensearch_store", lambda: _FakeStore())
    monkeypatch.setattr(rag_core, "_lookup_cache", lambda **kwargs: (None, {"cache_hit": False, "cache_mode": None, "cache_backend": None}))
    monkeypatch.setattr(rag_core, "_store_cache", lambda **kwargs: None)
    monkeypatch.setattr(rag_core, "_apply_rerank", lambda query, items: items)
    monkeypatch.setattr(rag_core, "_apply_score_thresholds", lambda items, use_rerank: items)

    items = rag_core.retrieve("发热咽痛", top_k=3, use_rerank=False)

    assert len(items) == 1
    assert items[0]["eid"] == "E1"
    assert items[0]["chunk_id"] == "chunk-1"
    assert items[0]["source"] == "qa.csv"
    assert items[0]["metadata"]["department"] == "内科"

    meta = rag_core.get_last_retrieval_meta()
    assert meta["backend"] == "opensearch"
    assert meta["bm25_hits"] == 1
    assert meta["knn_hits"] == 1
    assert meta["hybrid_mode"] == "app_rrf"
