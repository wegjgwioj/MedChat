# -*- coding: utf-8 -*-

from __future__ import annotations


def test_rag_retrieve_api_includes_evidence_quality(monkeypatch):
    from fastapi.testclient import TestClient

    import app.rag.rag_core as rag_core
    from app.api_server import app

    class FakeStats:
        collection = "kb"
        count = 10
        persist_dir = "app/rag/kb_store"
        device = "cpu"
        embed_model = "fake-embed"
        rerank_model = "fake-rerank"
        updated_at = "2026-03-27 12:00:00"

    monkeypatch.delenv("TRIAGE_API_KEY", raising=False)
    monkeypatch.setattr(
        rag_core,
        "retrieve",
        lambda *args, **kwargs: [
            {
                "eid": "E1",
                "text": "测试证据",
                "source": "kb.md",
                "chunk_id": "kb:1",
                "score": 0.12,
                "rerank_score": 0.91,
                "metadata": {"department": "内科", "title": "测试", "row": 1, "source_file": "kb.md"},
            }
        ],
    )
    monkeypatch.setattr(rag_core, "get_stats", lambda: FakeStats())
    monkeypatch.setattr(
        rag_core,
        "get_last_retrieval_meta",
        lambda: {
            "search_query": "咳嗽怎么办",
            "cache_hit": True,
            "cache_mode": "semantic",
            "hybrid_enabled": True,
        },
    )

    client = TestClient(app)
    resp = client.post("/v1/rag/retrieve", json={"query": "咳嗽怎么办", "top_k": 5, "top_n": 30, "use_rerank": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["evidence_quality"]["level"] == "low"
    assert body["evidence_quality"]["reason"] == "too_few_hits"
    assert body["evidence_quality"]["count"] == 1
    assert body["retrieval_meta"]["cache_hit"] is True
    assert body["retrieval_meta"]["cache_mode"] == "semantic"
    assert body["retrieval_meta"]["hybrid_enabled"] is True
    assert body["retrieval_meta"]["search_query"] == "咳嗽怎么办"


def test_eval_meddg_summary_includes_evidence_quality_counts():
    from scripts.eval_meddg_e2e import TurnResult, summarize

    turns = [
        TurnResult(
            dialog_idx=0,
            turn_idx=0,
            user_text="a",
            mode="answer",
            latency_ms=120.0,
            citations=1,
            rag_hits=2,
            rag_latency_ms=30.0,
            planner_strategy="triage",
            evidence_quality_level="ok",
            evidence_quality_reason="ok",
            error=None,
        ),
        TurnResult(
            dialog_idx=0,
            turn_idx=1,
            user_text="b",
            mode="answer",
            latency_ms=150.0,
            citations=0,
            rag_hits=1,
            rag_latency_ms=35.0,
            planner_strategy="triage",
            evidence_quality_level="low",
            evidence_quality_reason="too_few_hits",
            error=None,
        ),
    ]

    summary = summarize(turns)

    assert summary["evidence_quality_counts"] == {"ok": 1, "low": 1}
    assert summary["low_evidence_rate"] == 0.5


def test_eval_rag_summary_includes_evidence_quality_counts():
    from scripts.eval_rag_quality import RagCase, summarize_cases

    cases = [
        RagCase(
            idx=0,
            query="q1",
            reference="r1",
            top_k=5,
            latency_ms=80.0,
            evidence_count=2,
            max_similarity=0.2,
            hit=True,
            evidence_quality_level="ok",
            evidence_quality_reason="ok",
            error=None,
        ),
        RagCase(
            idx=1,
            query="q2",
            reference="r2",
            top_k=5,
            latency_ms=100.0,
            evidence_count=1,
            max_similarity=0.05,
            hit=False,
            evidence_quality_level="low",
            evidence_quality_reason="too_few_hits",
            error=None,
        ),
    ]

    summary = summarize_cases(cases, sim_threshold=0.08, top_k=5)

    assert summary["evidence_quality_counts"] == {"ok": 1, "low": 1}
    assert summary["low_evidence_rate"] == 0.5
