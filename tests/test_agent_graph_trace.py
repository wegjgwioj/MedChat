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


def test_answer_trace_includes_medication_safety(monkeypatch):
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

    state = {
        "session": AgentSessionState(
            session_id="trace-safety",
            summary="症状：咽痛；时长：2天；年龄24岁；性别女；严重程度3/10；发热：no",
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
        ),
        "mode": "answer",
        "user_message": "我喉咙痛两天了",
        "evidence": [
            {"eid": "E1", "text": "证据1", "source": "kb1", "chunk_id": "kb1:1", "score": 0.01, "rerank_score": 0.93, "metadata": {}},
        ],
        "trace": {"rag_stats": {"evidence_quality": {"level": "ok", "reason": "enough", "count": 1}}},
        "citations": [],
    }

    graph._node_answer_compose(state)

    trace = state["trace"]
    assert "medication_safety" in trace
    assert trace["medication_safety"]["blocked_count"] == 1
