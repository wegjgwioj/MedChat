# -*- coding: utf-8 -*-

from __future__ import annotations

from tests.conftest import FakeRagStore


class _FakeDoc:
    def __init__(self, text: str, source_file: str):
        self.page_content = text
        self.metadata = {
            "department": "内科",
            "title": text,
            "row": 1,
            "source_file": source_file,
        }


def _fake_store():
    return FakeRagStore(
        rows=[
            {
                "page_content": "饮食清淡，注意休息。",
                "metadata": {"department": "内科", "title": "饮食清淡，注意休息。", "row": 1, "source_file": "a.md"},
            },
            {
                "page_content": "胃痛反酸常见于胃炎或胃食管反流。",
                "metadata": {"department": "内科", "title": "胃痛反酸常见于胃炎或胃食管反流。", "row": 1, "source_file": "b.md"},
            },
        ]
    )


def test_hybrid_merge_promotes_sparse_match_when_dense_ties(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "1")
    monkeypatch.setenv("RAG_HYBRID_ALPHA", "0.25")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)
    monkeypatch.setattr(
        rag_core,
        "_vector_search",
        lambda query, top_n, department=None: [
            (_FakeDoc("饮食清淡，注意休息。", "a.md"), 0.05),
            (_FakeDoc("胃痛反酸常见于胃炎或胃食管反流。", "b.md"), 0.08),
        ],
    )

    evidence = rag_core.retrieve("胃痛反酸", top_k=2, use_rerank=False)

    assert [item["text"] for item in evidence] == [
        "胃痛反酸常见于胃炎或胃食管反流。",
        "饮食清淡，注意休息。",
    ]

    meta = rag_core.get_last_retrieval_meta()
    assert meta["hybrid_enabled"] is True
    assert meta["cache_hit"] is False


def test_hybrid_merge_uses_sparse_candidates_outside_dense_results(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "1")
    monkeypatch.setenv("RAG_HYBRID_ALPHA", "0.5")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)
    monkeypatch.setattr(
        rag_core,
        "_vector_search",
        lambda query, top_n, department=None: [
            (_FakeDoc("饮食清淡，注意休息。", "dense.md"), 0.03),
        ],
    )
    monkeypatch.setattr(
        rag_core,
        "_sparse_search",
        lambda query, top_n, department=None: [
            (_FakeDoc("胃痛反酸常见于胃炎或胃食管反流。", "sparse.md"), 0.95),
        ],
    )

    evidence = rag_core.retrieve("胃痛反酸", top_k=2, use_rerank=False)
    texts = [item["text"] for item in evidence]
    sources = [item["source"] for item in evidence]

    assert "胃痛反酸常见于胃炎或胃食管反流。" in texts
    assert "sparse.md" in sources

    meta = rag_core.get_last_retrieval_meta()
    assert meta["dense_hits"] == 1
    assert meta["sparse_hits"] == 1


def test_hybrid_disabled_does_not_call_sparse_search(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "0")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)
    monkeypatch.setattr(
        rag_core,
        "_vector_search",
        lambda query, top_n, department=None: [
            (_FakeDoc("饮食清淡，注意休息。", "dense.md"), 0.03),
        ],
    )
    monkeypatch.setattr(
        rag_core,
        "_sparse_search",
        lambda query, top_n, department=None: (_ for _ in ()).throw(RuntimeError("sparse should not run")),
    )

    evidence = rag_core.retrieve("胃痛反酸", top_k=1, use_rerank=False)

    assert [item["source"] for item in evidence] == ["dense.md"]


def test_hybrid_merge_can_be_disabled_by_env(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "0")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)
    monkeypatch.setattr(
        rag_core,
        "_vector_search",
        lambda query, top_n, department=None: [
            (_FakeDoc("饮食清淡，注意休息。", "a.md"), 0.05),
            (_FakeDoc("胃痛反酸常见于胃炎或胃食管反流。", "b.md"), 0.08),
        ],
    )

    evidence = rag_core.retrieve("胃痛反酸", top_k=2, use_rerank=False)

    assert [item["text"] for item in evidence] == [
        "饮食清淡，注意休息。",
        "胃痛反酸常见于胃炎或胃食管反流。",
    ]


def test_hybrid_merge_keeps_evidence_contract_fields(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "1")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)
    monkeypatch.setattr(
        rag_core,
        "_vector_search",
        lambda query, top_n, department=None: [
            (_FakeDoc("胃痛反酸常见于胃炎或胃食管反流。", "b.md"), 0.08),
        ],
    )

    evidence = rag_core.retrieve("胃痛反酸", top_k=1, use_rerank=False)

    assert evidence == [
        {
            "eid": "E1",
            "text": "胃痛反酸常见于胃炎或胃食管反流。",
            "source": "b.md",
            "chunk_id": "b.md:1:1",
            "score": 0.08,
            "rerank_score": None,
            "metadata": {
                "department": "内科",
                "title": "胃痛反酸常见于胃炎或胃食管反流。",
                "row": 1,
                "source_file": "b.md",
            },
        }
    ]
