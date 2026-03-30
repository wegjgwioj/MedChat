# -*- coding: utf-8 -*-

from __future__ import annotations


def test_triage_planner_prioritizes_pending_record_confirmation(monkeypatch, tmp_path):
    from app.agent import graph
    from app.agent.state import AgentSessionState, PendingRecordFact

    sess = AgentSessionState(session_id="confirm-1")
    sess.pending_record_facts = [
        PendingRecordFact(
            fact_id="fact-1",
            category="allergy",
            label="过敏",
            value="青霉素过敏",
            source_kind="ocr",
            source_excerpt="过敏史：青霉素",
            status="pending",
        )
    ]

    state = {"session": sess, "user_message": "我喉咙痛", "trace": {}}
    out = graph._node_triage_planner(state)

    assert out["mode"] == "ask"
    assert any("青霉素过敏" in q for q in out["next_questions"])


def test_memory_update_confirms_pending_record_fact_into_longitudinal_records(monkeypatch):
    from app.agent import graph
    from app.agent.state import AgentSessionState, PendingRecordFact

    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)

    sess = AgentSessionState(session_id="confirm-2")
    sess.pending_record_facts = [
        PendingRecordFact(
            fact_id="fact-1",
            category="allergy",
            label="过敏",
            value="青霉素过敏",
            source_kind="ocr",
            source_excerpt="过敏史：青霉素",
            status="pending",
        )
    ]

    state = {"session": sess, "user_message": "是，我对青霉素过敏", "trace": {}}
    out = graph._node_memory_update(state)
    updated = out["session"]

    assert updated.pending_record_facts[0].status == "confirmed"
    assert any(record.value == "青霉素过敏" for record in updated.longitudinal_records)
    assert "青霉素过敏" in updated.record_summary
    assert state["trace"]["record_confirmation"]["confirmed_count"] == 1


def test_triage_planner_does_not_reask_rejected_pending_record_fact(monkeypatch):
    from app.agent import graph
    from app.agent.state import AgentSessionState, PendingRecordFact

    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)

    sess = AgentSessionState(session_id="confirm-3")
    sess.pending_record_facts = [
        PendingRecordFact(
            fact_id="fact-1",
            category="allergy",
            label="过敏",
            value="青霉素过敏",
            source_kind="ocr",
            source_excerpt="过敏史：青霉素",
            status="pending",
        )
    ]

    updated = graph._node_memory_update(
        {"session": sess, "user_message": "不是，我没有青霉素过敏", "trace": {}}
    )["session"]

    assert updated.pending_record_facts[0].status == "rejected"
    assert not any(record.value == "青霉素过敏" for record in updated.longitudinal_records)
    assert "青霉素过敏" not in updated.record_summary

    out = graph._node_triage_planner({"session": updated, "user_message": "我喉咙痛", "trace": {}})
    assert out["mode"] != "ask" or not any("青霉素过敏" in q for q in out.get("next_questions", []))
