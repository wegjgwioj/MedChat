from pathlib import Path
from types import SimpleNamespace


def test_triage_payload_marks_low_evidence_and_adds_followups(monkeypatch):
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.triage_service import _normalize_answer_schema, triage_step_build_payload

    evidence = [
        {
            "eid": "E1",
            "text": "仅有一条本地资料。",
            "source": "kb.md",
            "chunk_id": "kb:1",
            "score": 0.21,
            "rerank_score": 0.92,
            "metadata": {"department": "内科", "title": "测试", "row": 1, "source_file": "kb.md"},
        }
    ]

    normalized = _normalize_answer_schema(
        {
            "triage_level": "ROUTINE",
            "reasoning": "现有信息有限。",
            "uncertainty": "",
            "key_questions": [],
        },
        evidence_list=evidence,
    )

    payload = triage_step_build_payload(
        answer_json=normalized,
        evidence_list=evidence,
        rag_query="咳嗽怎么办",
        mode="safe",
        created_at="2026-03-27 10:00:00",
        rag_status="ok",
        trace=[],
    )

    assert "low_local_evidence" in normalized["uncertainty"]
    assert normalized["key_questions"], "low evidence should trigger follow-up questions"

    evidence_quality = payload["meta"].get("evidence_quality")
    assert isinstance(evidence_quality, dict)
    assert evidence_quality.get("level") == "low"
    assert evidence_quality.get("reason") == "too_few_hits"
    assert evidence_quality.get("count") == 1


def test_agent_chat_v2_low_evidence_skips_llm_and_returns_cautious_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import run_chat_v2_turn
    import app.agent.graph as graph
    import app.rag.retriever as retriever
    from app.agent.storage_sqlite import SqliteSessionStore

    graph._GRAPH = None
    monkeypatch.setattr(graph, "_STORE", SqliteSessionStore(Path(tmp_path) / "agent.sqlite3"))

    calls = {"llm": 0}

    def fake_llm(*args, **kwargs):
        calls["llm"] += 1
        return "根据[E1]资料可以明确判断问题不大。"

    def fake_retrieve(*args, **kwargs):
        return [
            {
                "eid": "E1",
                "text": "风疹可经飞沫传播。",
                "source": "kb.md",
                "chunk_id": "kb:1",
                "score": 0.12,
                "rerank_score": 0.88,
                "metadata": {"department": "传染科", "title": "风疹", "row": 1, "source_file": "kb.md"},
            }
        ]

    monkeypatch.setattr(graph, "_call_llm_text", fake_llm)
    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)

    out = run_chat_v2_turn(
        session_id="low-evidence-session",
        user_message="风疹病毒是怎么感染的？",
        top_k=5,
        top_n=30,
        use_rerank=False,
    )

    assert out["mode"] == "answer"
    assert calls["llm"] == 0, "low evidence should bypass answer-generation LLM"
    assert "证据不足" in out["answer"]

    evidence_quality = (out["trace"] or {}).get("rag_stats", {}).get("evidence_quality")
    assert isinstance(evidence_quality, dict)
    assert evidence_quality.get("level") == "low"
    assert evidence_quality.get("reason") == "too_few_hits"


def test_triage_retrieve_trace_includes_rag_cache_meta(monkeypatch):
    import app.triage_service as triage_service

    engine = triage_service.TriageEngine()
    monkeypatch.setattr(engine, "init", lambda: None)
    monkeypatch.setattr(
        triage_service,
        "rag_retriever",
        SimpleNamespace(
            retrieve=lambda rag_query, top_k=5: [
                {
                    "eid": "E1",
                    "text": "发烧时要监测体温。",
                    "source": "kb.md",
                    "chunk_id": "kb:1",
                    "score": 0.1,
                    "rerank_score": 0.9,
                    "metadata": {"department": "内科", "title": "发烧", "row": 1, "source_file": "kb.md"},
                }
            ],
            get_last_retrieval_meta=lambda: {
                "cache_hit": True,
                "cache_mode": "exact",
                "hybrid_enabled": True,
                "search_query": "发烧怎么办",
            },
        ),
        raising=False,
    )

    trace = []
    evidence, rag_status, rag_error = engine.retrieve_evidence("发烧怎么办", top_k=1, trace=trace)

    assert rag_status == "ok"
    assert rag_error is None
    assert len(evidence) == 1
    rag_step = next(step for step in trace if step.get("step") == "rag.retrieve")
    assert rag_step["cache_hit"] is True
    assert rag_step["cache_mode"] == "exact"
    assert rag_step["hybrid_enabled"] is True


def test_agent_trace_includes_rag_cache_meta(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import run_chat_v2_turn
    import app.agent.graph as graph
    import app.rag.retriever as retriever
    from app.agent.storage_sqlite import SqliteSessionStore

    graph._GRAPH = None
    monkeypatch.setattr(graph, "_STORE", SqliteSessionStore(Path(tmp_path) / "agent-meta.sqlite3"))
    monkeypatch.setattr(
        retriever,
        "retrieve",
        lambda *args, **kwargs: [
            {
                "eid": "E1",
                "text": "咳嗽时注意补液休息。",
                "source": "kb.md",
                "chunk_id": "kb:1",
                "score": 0.1,
                "rerank_score": 0.9,
                "metadata": {"department": "内科", "title": "咳嗽", "row": 1, "source_file": "kb.md"},
            },
            {
                "eid": "E2",
                "text": "如果出现呼吸困难要尽快就医。",
                "source": "kb2.md",
                "chunk_id": "kb2:1",
                "score": 0.2,
                "rerank_score": 0.88,
                "metadata": {"department": "内科", "title": "咳嗽", "row": 2, "source_file": "kb2.md"},
            },
        ],
    )
    monkeypatch.setattr(
        retriever,
        "get_last_retrieval_meta",
        lambda: {
            "cache_hit": True,
            "cache_mode": "semantic",
            "hybrid_enabled": True,
            "search_query": "咳嗽 呼吸困难",
        },
    )
    monkeypatch.setattr(
        graph,
        "_call_llm_text",
        lambda *args, **kwargs: "建议先补液休息，若呼吸困难加重请尽快就医。[E1][E2]\n\n引用：[E1][E2]\n免责声明：本回答仅供信息参考，不能替代医生面诊。",
    )

    out = run_chat_v2_turn(
        session_id="agent-meta-session",
        user_message="风疹病毒是怎么感染的？",
        top_k=2,
        top_n=30,
        use_rerank=False,
    )

    rag_stats = (out["trace"] or {}).get("rag_stats") or {}
    assert rag_stats["cache_hit"] is True
    assert rag_stats["cache_mode"] == "semantic"
    assert rag_stats["hybrid_enabled"] is True
    assert rag_stats["search_query"] == "咳嗽 呼吸困难"
