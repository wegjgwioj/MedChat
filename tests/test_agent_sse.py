# -*- coding: utf-8 -*-

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def _parse_sse_events(raw: str):
    events = []
    chunks = [chunk.strip() for chunk in str(raw or "").split("\n\n") if chunk.strip()]
    for chunk in chunks:
        event_name = "message"
        data_lines = []
        for line in chunk.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        payload = None
        if data_lines:
            payload = json.loads("\n".join(data_lines))
        events.append((event_name, payload))
    return events


def test_agent_chat_v2_stream_returns_sse_events(monkeypatch):
    from app.api_server import app
    import app.agent.router as agent_router

    monkeypatch.setattr(
        agent_router,
        "run_chat_v2_turn",
        lambda **kwargs: {
            "session_id": "sse-demo",
            "mode": "ask",
            "ask_text": "为了更准确判断，我还需要补充几个信息。",
            "questions": [{"slot": "age", "question": "请问你多大年龄？", "type": "text"}],
            "next_questions": ["请问你多大年龄？"],
            "answer": "",
            "citations": [],
            "slots": {"age": None},
            "summary": "",
            "trace": {"node_order": ["SafetyGate", "PersistState"], "timings_ms": {"SafetyGate": 0, "PersistState": 1}},
        },
    )

    client = TestClient(app)
    with client.stream("POST", "/v1/agent/chat_v2/stream", json={"user_message": "我头疼"}) as resp:
        raw = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(raw)
    assert [name for name, _ in events][:2] == ["ack", "stage"]
    assert events[-1][0] == "final"
    assert events[-1][1]["session_id"] == "sse-demo"
    assert events[-1][1]["mode"] == "ask"
    assert events[-1][1]["questions"][0]["slot"] == "age"


def test_agent_chat_v2_stream_emits_error_event_when_turn_fails(monkeypatch):
    from app.api_server import app
    import app.agent.router as agent_router

    def _boom(**kwargs):
        raise RuntimeError("stream failed")

    monkeypatch.setattr(agent_router, "run_chat_v2_turn", _boom)

    client = TestClient(app)
    with client.stream("POST", "/v1/agent/chat_v2/stream", json={"user_message": "我头疼"}) as resp:
        raw = "".join(resp.iter_text())

    assert resp.status_code == 200
    events = _parse_sse_events(raw)
    assert events[0][0] == "ack"
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "STREAM_RUNTIME_ERROR"
    assert "stream failed" in events[-1][1]["message"]


def test_agent_chat_v2_stream_reuses_response_contract(monkeypatch):
    from app.api_server import app
    import app.agent.router as agent_router

    monkeypatch.setattr(
        agent_router,
        "run_chat_v2_turn",
        lambda **kwargs: {
            "session_id": "sse-answer",
            "mode": "answer",
            "ask_text": "",
            "questions": [],
            "next_questions": [],
            "answer": "建议补液休息。[E1]\n\n引用：[E1]\n免责声明：本回答仅供信息参考，不能替代医生面诊。",
            "citations": [{"eid": "E1", "score": 0.1}],
            "slots": {"symptoms": ["头痛"]},
            "summary": "症状：头痛",
            "trace": {"node_order": ["SafetyGate", "MemoryUpdate", "RAGRetrieve"], "timings_ms": {"SafetyGate": 0}},
        },
    )

    client = TestClient(app)
    with client.stream("POST", "/v1/agent/chat_v2/stream", json={"user_message": "我头疼"}) as resp:
        raw = "".join(resp.iter_text())

    events = _parse_sse_events(raw)
    final_payload = events[-1][1]
    assert final_payload["mode"] == "answer"
    assert final_payload["answer"]
    assert final_payload["citations"][0]["eid"] == "E1"
    assert isinstance(final_payload["trace"]["node_order"], list)
