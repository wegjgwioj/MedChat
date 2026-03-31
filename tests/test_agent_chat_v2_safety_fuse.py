# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import api_server
from app.agent import graph, router
from app.agent.state import AgentSessionState, LongitudinalRecordFact, Slots
from app.agent.storage_sqlite import SqliteSessionStore


def _confirmed_penicillin_allergy() -> LongitudinalRecordFact:
    return LongitudinalRecordFact(
        category="allergy",
        label="过敏",
        value="青霉素过敏",
        text="过敏：青霉素过敏",
        importance_score=0.98,
    )


def _save_answer_ready_session(db_path: Path, session_id: str) -> None:
    store = SqliteSessionStore(db_path)
    session = AgentSessionState(
        session_id=session_id,
        slots=Slots(
            age=28,
            sex="女",
            symptoms=["咽痛"],
            duration="2天",
            severity="中",
            fever="no",
        ),
        longitudinal_records=[_confirmed_penicillin_allergy()],
    )
    store.save_session(session)


def test_agent_chat_v2_blocks_conflicting_medication_and_appends_unified_trace(monkeypatch, tmp_path: Path):
    session_id = "agent-safety-session"
    db_path = tmp_path / "agent.sqlite3"
    _save_answer_ready_session(db_path, session_id)

    monkeypatch.setenv("AGENT_SESSION_STORE", "sqlite")
    monkeypatch.setenv("AGENT_SQLITE_DB_PATH", str(db_path))
    monkeypatch.setattr(graph, "_extract_slots_with_llm", lambda message: Slots())
    monkeypatch.setattr(graph, "_build_low_evidence_answer", lambda evidence: "建议先口服阿莫西林。")
    monkeypatch.setattr("app.rag.retriever.retrieve", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.rag.retriever.get_last_retrieval_meta", lambda: {})
    monkeypatch.setattr(
        "app.rag.rag_core.get_stats",
        lambda: SimpleNamespace(
            backend="faiss-hnsw",
            device="cpu",
            collection="test",
            count=0,
        ),
    )
    graph._STORE = None
    graph._GRAPH = None

    out = graph.run_chat_v2_turn(
        session_id=session_id,
        user_message="现在还是不舒服",
        top_k=1,
        top_n=1,
        use_rerank=False,
    )

    assert out["mode"] == "answer"
    assert "阿莫西林" not in out["answer"]
    assert "青霉素过敏" in out["answer"]
    assert out["trace"]["safety_fuse"]["blocked_count"] == 1


def test_agent_stream_final_event_reuses_same_final_result(monkeypatch):
    final_payload = {
        "session_id": "stream-session",
        "mode": "answer",
        "ask_text": "",
        "questions": [],
        "next_questions": [],
        "answer": "你已确认青霉素过敏，本轮已移除冲突用药建议，请线下咨询替代方案。",
        "citations": [],
        "slots": {},
        "summary": "",
        "trace": {"safety_fuse": {"blocked_count": 1}},
    }
    monkeypatch.setattr(router, "run_chat_v2_turn", lambda **kwargs: final_payload)

    client = TestClient(api_server.app)
    normal = client.post(
        "/v1/agent/chat_v2",
        json={"session_id": "stream-session", "user_message": "需要建议", "top_k": 1, "top_n": 1, "use_rerank": False},
    )
    assert normal.status_code == 200

    with client.stream(
        "POST",
        "/v1/agent/chat_v2/stream",
        json={"session_id": "stream-session", "user_message": "需要建议", "top_k": 1, "top_n": 1, "use_rerank": False},
    ) as response:
        assert response.status_code == 200
        final_data = None
        current_event = None
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[len("event: ") :]
                continue
            if line.startswith("data: ") and current_event == "final":
                final_data = json.loads(line[len("data: ") :])
                break

    assert final_data == normal.json()
