# -*- coding: utf-8 -*-
"""test_agent_escalate.py

测试 escalate 模式（红旗症状处理）：
- 红旗症状触发 escalate
- escalate 模式不继续追问
- safety_level 正确设置

批次4新增。
"""

import pytest
from fastapi.testclient import TestClient


def test_red_flag_triggers_escalate(monkeypatch):
    """测试红旗症状触发 escalate"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    # 胸痛是红旗症状
    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "我突然胸痛，喘不上气"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body.get("mode") == "escalate"
    assert "急诊" in body.get("answer", "") or "就医" in body.get("answer", "")


def test_escalate_no_further_questions(monkeypatch):
    """测试 escalate 模式不继续追问"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "意识不清，抽搐"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body.get("mode") == "escalate"
    assert body.get("questions") == []
    assert body.get("next_questions") == []


def test_red_flag_detection_function(monkeypatch):
    """测试红旗检测函数"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_red_flag

    # 验证红旗检测函数
    hits = _looks_like_red_flag("胸痛，呼吸困难")
    assert len(hits) >= 1
    assert "胸痛" in hits or "呼吸困难" in hits


def test_multiple_red_flags_detected(monkeypatch):
    """测试多个红旗症状都被检测到"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_red_flag

    hits = _looks_like_red_flag("胸痛加上呼吸困难，口唇发紫")
    # 应该检测到多个红旗症状
    assert len(hits) >= 2


def test_escalate_answer_mentions_emergency(monkeypatch):
    """测试 escalate 回答提到紧急就医"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "胸痛加上呼吸困难"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body.get("mode") == "escalate"
    answer = body.get("answer", "")
    # 应该包含紧急相关词汇
    assert any(w in answer for w in ["急诊", "就医", "紧急", "医院", "120", "急救"])


def test_escalate_trace_has_safety_gate(monkeypatch):
    """测试 escalate 模式的 trace 包含 SafetyGate"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "昏迷不醒"},
    )

    assert resp.status_code == 200
    body = resp.json()

    trace = body.get("trace", {})
    node_order = trace.get("node_order", [])

    # escalate 应该经过 SafetyGate
    assert "SafetyGate" in node_order


def test_non_red_flag_not_escalate(monkeypatch):
    """测试非红旗症状不触发 escalate"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_red_flag

    # 普通症状不应该触发红旗
    hits = _looks_like_red_flag("我有点头疼")
    assert len(hits) == 0


def test_empty_message_no_red_flag(monkeypatch):
    """测试空消息不触发红旗"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_red_flag

    hits = _looks_like_red_flag("")
    assert len(hits) == 0

    hits = _looks_like_red_flag(None)
    assert len(hits) == 0
