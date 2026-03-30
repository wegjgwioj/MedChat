# -*- coding: utf-8 -*-

from __future__ import annotations


def test_memory_update_builds_record_summary(monkeypatch, tmp_path):
    from app.agent import graph
    from app.agent.state import AgentSessionState
    from app.agent.storage_sqlite import SqliteSessionStore

    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)
    monkeypatch.setattr(graph, "_STORE", SqliteSessionStore(tmp_path / "agent_sessions.sqlite3"), raising=False)

    state = {
        "session": AgentSessionState(session_id="s-record-1"),
        "user_message": "我24岁，青霉素过敏，有哮喘史，目前在吃维生素。",
        "trace": {},
    }

    out = graph._node_memory_update(state)
    sess = out["session"]

    assert "年龄24岁" in sess.record_summary
    assert "既往史：哮喘史" in sess.record_summary or "既往史：哮喘" in sess.record_summary
    assert "过敏：青霉素过敏" in sess.record_summary
    assert "用药：维生素" in sess.record_summary


def test_answer_compose_removes_conflicting_medication_but_keeps_other_advice(monkeypatch):
    from app.agent import graph
    from app.agent.state import AgentSessionState, LongitudinalRecordFact

    monkeypatch.setattr(
        graph,
        "_call_llm_text",
        lambda system, user: (
            "建议先口服阿莫西林；如发热可考虑对乙酰氨基酚。[E1]\n\n"
            "引用：[E1]\n免责声明：本回答仅供信息参考，不能替代医生面诊。"
        ),
    )

    sess = AgentSessionState(
        session_id="s-guard-1",
        summary="症状：咽痛",
        longitudinal_records=[
            LongitudinalRecordFact(
                category="allergy",
                label="过敏",
                value="青霉素过敏",
                text="过敏：青霉素过敏",
                importance_score=0.98,
            )
        ],
        record_summary="过敏：青霉素过敏",
    )
    state = {
        "session": sess,
        "mode": "answer",
        "user_message": "我喉咙痛怎么办",
        "evidence": [
            {"eid": "E1", "text": "证据1", "source": "kb1", "chunk_id": "kb1:1", "score": 0.01, "rerank_score": 0.93, "metadata": {}},
            {"eid": "E2", "text": "证据2", "source": "kb2", "chunk_id": "kb2:1", "score": 0.02, "rerank_score": 0.91, "metadata": {}},
        ],
        "trace": {"rag_stats": {"evidence_quality": {"level": "ok", "reason": "enough", "count": 2}}},
        "citations": [],
    }

    out = graph._node_answer_compose(state)

    assert "青霉素过敏" in out["answer"]
    assert "建议先口服阿莫西林" not in out["answer"]
    assert "对乙酰氨基酚" in out["answer"]
    assert state["trace"]["medication_safety"]["blocked_count"] == 1


def test_answer_compose_does_not_flag_negative_warning_context(monkeypatch):
    from app.agent import graph
    from app.agent.state import AgentSessionState, LongitudinalRecordFact

    monkeypatch.setattr(
        graph,
        "_call_llm_text",
        lambda system, user: "你对青霉素过敏，因此应避免阿莫西林；如发热可用对乙酰氨基酚。[E1]\n\n引用：[E1]\n免责声明：本回答仅供信息参考，不能替代医生面诊。",
    )

    sess = AgentSessionState(
        session_id="s-record-3",
        summary="症状：咽痛",
        longitudinal_records=[
            LongitudinalRecordFact(
                category="allergy",
                label="过敏",
                value="青霉素过敏",
                text="过敏：青霉素过敏",
                importance_score=0.98,
            )
        ],
        record_summary="过敏：青霉素过敏",
    )
    state = {
        "session": sess,
        "mode": "answer",
        "user_message": "我喉咙痛怎么办",
        "evidence": [
            {"eid": "E1", "text": "证据1", "source": "kb1", "chunk_id": "kb1:1", "score": 0.01, "rerank_score": 0.93, "metadata": {}},
            {"eid": "E2", "text": "证据2", "source": "kb2", "chunk_id": "kb2:1", "score": 0.02, "rerank_score": 0.91, "metadata": {}},
        ],
        "trace": {"rag_stats": {"evidence_quality": {"level": "ok", "reason": "enough", "count": 2}}},
        "citations": [],
    }

    out = graph._node_answer_compose(state)

    assert out["answer"].count("应避免阿莫西林") == 1
    assert state["trace"]["medication_safety"]["blocked_count"] == 0
