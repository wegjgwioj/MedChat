# -*- coding: utf-8 -*-

from __future__ import annotations


class _FakeDoc:
    def __init__(self, text: str, source_file: str):
        self.page_content = text
        self.metadata = {
            "department": "内科",
            "title": text,
            "row": 1,
            "source_file": source_file,
        }


def _fake_vector_docs():
    return [
        (_FakeDoc("证据A", "a.md"), 0.10),
        (_FakeDoc("证据B", "b.md"), 0.20),
        (_FakeDoc("证据C", "c.md"), 0.30),
    ]


def test_retrieve_filters_out_items_below_rerank_threshold(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    class FakeCollection:
        def count(self) -> int:
            return 3

    class FakeVS:
        _collection = FakeCollection()

    monkeypatch.setenv("RAG_RERANK_MIN_SCORE", "0.80")
    monkeypatch.setattr(rag_core, "get_vectordb", lambda: FakeVS())
    monkeypatch.setattr(rag_core, "_vector_search", lambda query, top_n, department=None: _fake_vector_docs())
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: True)

    def fake_apply_rerank(query: str, items):
        items[0]["rerank_score"] = 0.91
        items[1]["rerank_score"] = 0.79
        items[2]["rerank_score"] = 0.88
        return [items[0], items[2], items[1]]

    monkeypatch.setattr(rag_core, "_apply_rerank", fake_apply_rerank)

    evidence = rag_core.retrieve("咳嗽", top_k=3, use_rerank=True)

    assert [item["text"] for item in evidence] == ["证据A", "证据C"]
    assert [item["eid"] for item in evidence] == ["E1", "E2"]
    assert all(float(item["rerank_score"]) >= 0.80 for item in evidence)


def test_retrieve_returns_empty_when_all_items_below_rerank_threshold(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    class FakeCollection:
        def count(self) -> int:
            return 2

    class FakeVS:
        _collection = FakeCollection()

    monkeypatch.setenv("RAG_RERANK_MIN_SCORE", "0.95")
    monkeypatch.setattr(rag_core, "get_vectordb", lambda: FakeVS())
    monkeypatch.setattr(rag_core, "_vector_search", lambda query, top_n, department=None: _fake_vector_docs()[:2])
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: True)

    def fake_apply_rerank(query: str, items):
        items[0]["rerank_score"] = 0.70
        items[1]["rerank_score"] = 0.80
        return items

    monkeypatch.setattr(rag_core, "_apply_rerank", fake_apply_rerank)

    assert rag_core.retrieve("咳嗽", top_k=2, use_rerank=True) == []


def test_retrieve_filters_out_items_above_vector_score_threshold_without_rerank(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    class FakeCollection:
        def count(self) -> int:
            return 2

    class FakeVS:
        _collection = FakeCollection()

    monkeypatch.setenv("RAG_VECTOR_MAX_SCORE", "0.15")
    monkeypatch.setattr(rag_core, "get_vectordb", lambda: FakeVS())
    monkeypatch.setattr(rag_core, "_vector_search", lambda query, top_n, department=None: _fake_vector_docs()[:2])
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)

    evidence = rag_core.retrieve("咳嗽", top_k=2, use_rerank=False)

    assert len(evidence) == 1
    assert evidence[0]["text"] == "证据A"
    assert evidence[0]["eid"] == "E1"
    assert float(evidence[0]["score"]) <= 0.15
