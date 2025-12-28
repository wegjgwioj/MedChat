from fastapi.testclient import TestClient


def test_agent_kb_question_bypasses_followups(monkeypatch):
    # Offline deterministic
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "风疹病毒是怎么感染的？"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body.get("mode") == "answer"
    assert isinstance(body.get("answer"), str) and body["answer"].strip()

    # In kb_qa strategy, should not emit followups
    assert (body.get("questions") or []) == []
    assert (body.get("next_questions") or []) == []

    trace = body.get("trace") or {}
    assert isinstance(trace, dict)
    assert trace.get("planner_strategy") == "kb_qa"


# ===== 批次3新增测试 =====

def test_kb_question_confidence_high(monkeypatch):
    """测试高置信度科普问题"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_kb_question

    is_kb, confidence = _looks_like_kb_question("感冒是什么病？")
    assert is_kb == True
    assert confidence >= 0.9


def test_kb_question_confidence_medium(monkeypatch):
    """测试中置信度科普问题"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_kb_question

    # 包含个人标记但仍是科普问题
    is_kb, confidence = _looks_like_kb_question("我想问一下感冒怎么传播的？")
    assert is_kb == True
    assert 0.5 <= confidence < 0.9


def test_personal_symptom_not_kb_question(monkeypatch):
    """测试个人症状不是科普问题"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_kb_question

    is_kb, confidence = _looks_like_kb_question("我头疼三天了怎么办？")
    assert is_kb == False
    assert confidence == 0.0


def test_trace_contains_kb_qa_stats(monkeypatch):
    """测试 trace 包含 kb_qa_stats"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from fastapi.testclient import TestClient
    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "风疹病毒是怎么感染的？"},
    )

    assert resp.status_code == 200
    body = resp.json()

    trace = body.get("trace") or {}
    assert "kb_qa_stats" in trace
    assert trace["kb_qa_stats"].get("detected") == True
    assert trace["kb_qa_stats"].get("confidence") >= 0.7


def test_strong_kb_pattern_high_confidence(monkeypatch):
    """测试强科普句式获得高置信度"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_kb_question

    # 测试多种强信号句式
    test_cases = [
        ("糖尿病是什么病？", True, 0.9),
        ("流感怎么传播？", True, 0.9),
        ("肺炎如何预防？", True, 0.9),
        ("潜伏期多长？", True, 0.9),
        ("会不会传染？", True, 0.9),
        ("能不能治愈？", True, 0.9),
    ]

    for msg, expected_is_kb, min_confidence in test_cases:
        is_kb, confidence = _looks_like_kb_question(msg)
        assert is_kb == expected_is_kb, f"Failed for: {msg}"
        assert confidence >= min_confidence, f"Confidence too low for: {msg}, got {confidence}"


def test_empty_message_returns_false(monkeypatch):
    """测试空消息返回 False"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_kb_question

    is_kb, confidence = _looks_like_kb_question("")
    assert is_kb == False
    assert confidence == 0.0

    is_kb, confidence = _looks_like_kb_question(None)
    assert is_kb == False
    assert confidence == 0.0
