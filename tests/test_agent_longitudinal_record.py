# -*- coding: utf-8 -*-

from __future__ import annotations


def test_upsert_longitudinal_records_applies_importance_threshold(monkeypatch):
    import app.agent.record_index as record_index
    from app.agent.state import Slots

    monkeypatch.setenv("AGENT_RECORD_MIN_IMPORTANCE", "0.9")

    records, stats = record_index.upsert_longitudinal_records(
        [],
        Slots(age=24, sex="男", meds="维生素", allergy="青霉素过敏"),
    )

    assert len(records) == 1
    assert records[0].category == "allergy"
    assert records[0].importance_score >= 0.9
    assert stats["added"] == 1
    assert stats["skipped"] >= 1


def test_upsert_longitudinal_records_merges_semantic_duplicates(monkeypatch):
    import app.agent.record_index as record_index
    from app.agent.state import Slots

    monkeypatch.setattr(
        record_index,
        "_compute_record_similarity",
        lambda left, right: 0.96 if "过敏" in left and "过敏" in right else 0.05,
    )

    records, _ = record_index.upsert_longitudinal_records([], Slots(allergy="青霉素过敏"))
    records, stats = record_index.upsert_longitudinal_records(records, Slots(allergy="阿莫西林过敏"))

    allergy_records = [record for record in records if record.category == "allergy"]

    assert len(allergy_records) == 1
    assert stats["merged"] == 1
    assert stats["added"] == 0


def test_memory_update_builds_longitudinal_records_and_summary(monkeypatch, tmp_path):
    import app.agent.record_index as record_index
    from app.agent import graph
    from app.agent.state import AgentSessionState
    from app.agent.storage_sqlite import SqliteSessionStore

    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)
    monkeypatch.setattr(graph, "_STORE", SqliteSessionStore(tmp_path / "agent_sessions.sqlite3"), raising=False)
    monkeypatch.setattr(
        record_index,
        "_compute_record_similarity",
        lambda left, right: 0.96 if "过敏" in left and "过敏" in right else 0.05,
    )

    first = graph._node_memory_update(
        {
            "session": AgentSessionState(session_id="record-session-1"),
            "user_message": "我24岁，男，青霉素过敏，有哮喘史。",
            "trace": {},
        }
    )["session"]

    second_state = {
        "session": first,
        "user_message": "我对阿莫西林也过敏，目前在吃维生素。",
        "trace": {},
    }
    second = graph._node_memory_update(second_state)["session"]

    allergy_records = [record for record in second.longitudinal_records if record.category == "allergy"]
    med_records = [record for record in second.longitudinal_records if record.category == "medication"]

    assert len(allergy_records) == 1
    assert len(med_records) == 1
    assert "过敏：" in second.record_summary
    assert "用药：维生素" in second.record_summary
    assert second_state["trace"]["record_admission"]["merged"] == 1
