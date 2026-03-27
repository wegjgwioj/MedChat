from pathlib import Path


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
