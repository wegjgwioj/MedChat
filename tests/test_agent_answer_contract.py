# -*- coding: utf-8 -*-
"""test_agent_answer_contract.py

测试 Answer 模式的契约：
- 必须包含引用行
- 必须包含免责声明
- 无效引用被移除

批次1新增测试文件
"""

import pytest


def test_answer_must_have_citation_line(monkeypatch):
    """测试回答必须包含引用行"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = [{"eid": "E1", "text": "测试证据"}]
    answer = "这是一个测试回答。"

    result = _ensure_answer_contract(answer, evidence)

    assert "引用：" in result
    assert "[E1]" in result


def test_answer_must_have_disclaimer(monkeypatch):
    """测试回答必须包含免责声明"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = []
    answer = "这是一个测试回答。"

    result = _ensure_answer_contract(answer, evidence)

    assert "免责声明" in result
    assert "不能替代医生面诊" in result


def test_empty_evidence_shows_empty_citation(monkeypatch):
    """测试无证据时显示空引用"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = []
    answer = "这是一个测试回答。"

    result = _ensure_answer_contract(answer, evidence)

    assert "引用：[]" in result


def test_invalid_citation_removed(monkeypatch):
    """测试无效引用被移除"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = [{"eid": "E1", "text": "测试证据1"}]
    # 回答中引用了不存在的 E2, E3
    answer = "根据[E1][E2][E3]的资料，建议休息。"

    result = _ensure_answer_contract(answer, evidence)

    # E1 应保留在正文和引用行，E2/E3 应被移除
    assert "[E1]" in result
    assert "[E2]" not in result
    assert "[E3]" not in result


def test_citation_line_at_end(monkeypatch):
    """测试引用行在末尾（免责声明之前）"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = [{"eid": "E1", "text": "测试证据1"}, {"eid": "E2", "text": "测试证据2"}]
    answer = "这是回答。"

    result = _ensure_answer_contract(answer, evidence)

    # 引用行应在免责声明之前
    cite_pos = result.find("引用：")
    disclaimer_pos = result.find("免责声明")

    assert cite_pos < disclaimer_pos
    assert cite_pos > result.find("这是回答")


def test_multiple_evidences_sorted_by_number(monkeypatch):
    """测试多个证据按数字排序"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    # 故意乱序
    evidence = [
        {"eid": "E3", "text": "测试证据3"},
        {"eid": "E1", "text": "测试证据1"},
        {"eid": "E2", "text": "测试证据2"},
    ]
    answer = "这是回答。"

    result = _ensure_answer_contract(answer, evidence)

    # 引用应该按顺序
    assert "引用：[E1][E2][E3]" in result


def test_duplicate_citation_in_text_not_duplicated(monkeypatch):
    """测试正文中重复引用不影响引用行"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = [{"eid": "E1", "text": "测试证据1"}]
    answer = "根据[E1]的资料，[E1]建议休息。"

    result = _ensure_answer_contract(answer, evidence)

    # 引用行应该只有一个 E1
    assert result.count("引用：[E1]") == 1


def test_trace_contains_slots_changed(monkeypatch):
    """测试 trace 包含 slots_changed 字段"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from fastapi.testclient import TestClient
    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "我今年25岁，头疼"},
    )

    assert resp.status_code == 200
    body = resp.json()

    trace = body.get("trace") or {}
    assert "slots_changed" in trace
    assert isinstance(trace["slots_changed"], list)
    # 应该检测到 age 和 symptoms 的变化
    assert "age" in trace["slots_changed"] or "symptoms" in trace["slots_changed"]
