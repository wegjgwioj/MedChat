from fastapi.testclient import TestClient


def test_agent_chat_v2_emits_trace_node_order_and_timings(monkeypatch):
    import app.agent.graph as graph

    graph._GRAPH = None
    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "我头疼"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert isinstance(body.get("session_id"), str) and body["session_id"]
    assert body.get("mode") in {"ask", "answer", "escalate"}

    trace = body.get("trace") or {}
    assert isinstance(trace, dict)

    node_order = trace.get("node_order") or []
    assert isinstance(node_order, list) and node_order

    timings = trace.get("timings_ms")
    assert isinstance(timings, dict)

    # Ask-path should at least run these nodes
    must = {"SafetyGate", "MemoryUpdate", "TriagePlanner", "PersistState"}
    assert must.issubset(set(node_order))

    for k in must:
        assert k in timings
        assert isinstance(timings[k], int)
        assert timings[k] >= 0

    # Structured follow-ups contract when mode=ask
    if body.get("mode") == "ask":
        assert isinstance(body.get("ask_text"), str)
        assert body["ask_text"].strip()
        qs = body.get("questions")
        assert isinstance(qs, list) and qs
        assert all(isinstance(x, dict) and x.get("slot") and x.get("question") for x in qs)
