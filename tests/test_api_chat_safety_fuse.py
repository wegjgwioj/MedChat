# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import api_server
from app.agent.state import AgentSessionState, LongitudinalRecordFact
from app.agent.storage_sqlite import SqliteSessionStore


def _confirmed_penicillin_allergy() -> LongitudinalRecordFact:
    return LongitudinalRecordFact(
        category="allergy",
        label="过敏",
        value="青霉素过敏",
        text="过敏：青霉素过敏",
        importance_score=0.98,
    )


def _dangerous_answer_json() -> dict:
    return {
        "triage_level": "ROUTINE",
        "red_flags": [],
        "immediate_actions": ["建议先口服阿莫西林。"],
        "what_not_to_do": [],
        "key_questions": [],
        "reasoning": "考虑上呼吸道感染。",
        "uncertainty": "",
        "safety_notice": "本回答仅供参考。",
        "citations_used": [],
    }


def _save_confirmed_session(db_path: Path, session_id: str) -> None:
    store = SqliteSessionStore(db_path)
    session = AgentSessionState(
        session_id=session_id,
        longitudinal_records=[_confirmed_penicillin_allergy()],
    )
    store.save_session(session)


def test_chat_reply_uses_confirmed_record_safety_fuse(monkeypatch, tmp_path: Path):
    session_id = "chat-safety-session"
    db_path = tmp_path / "agent.sqlite3"
    _save_confirmed_session(db_path, session_id)

    monkeypatch.setenv("AGENT_SESSION_STORE", "sqlite")
    monkeypatch.setenv("AGENT_SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("CHAT_MAX_TURNS", "1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(api_server, "_update_intake_slots_with_llm", lambda intake_slots, patient_message: False)

    monkeypatch.setattr(
        api_server,
        "triage_step_retrieve",
        lambda rag_query, top_k=5, trace=None: {
            "rag_query": rag_query,
            "evidence": [],
            "rag_status": "disabled",
            "rag_error": None,
            "trace": trace or [],
        },
    )
    monkeypatch.setattr(
        api_server,
        "triage_step_assess",
        lambda user_text, evidence_list, trace=None: {
            "evidence_block": "",
            "raw": "{}",
            "answer": _dangerous_answer_json(),
            "trace": trace or [],
        },
    )
    monkeypatch.setattr(
        api_server,
        "triage_step_safety",
        lambda answer_json, evidence_block, evidence_list, trace=None: {
            "answer": answer_json,
            "trace": trace or [],
        },
    )
    api_server._CHAT_GRAPH = None

    client = TestClient(api_server.app)
    response = client.post(
        "/v1/chat",
        json={
            "session_id": session_id,
            "patient_message": "我喉咙痛两天了",
            "top_k": 1,
            "mode": "safe",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "阿莫西林" not in payload["doctor_reply"]
    assert "青霉素过敏" in payload["doctor_reply"]

    safety_step = next(item for item in payload["triage"]["meta"]["trace"] if item["step"] == "record.safety")
    assert safety_step["blocked_count"] == 1
