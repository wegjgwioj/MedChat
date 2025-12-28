# -*- coding: utf-8 -*-
"""test_agent_anti_repeat.py

测试 Anti-repeat 机制：
- 同一槽位追问次数限制
- 用户拒绝回答检测
- 连续两轮不重复同一问题

批次2新增测试文件
"""

import pytest


def test_user_decline_detection(monkeypatch):
    """测试用户拒绝回答检测"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _user_declined_slot

    # 应该检测为拒绝
    assert _user_declined_slot("不想说") == True
    assert _user_declined_slot("跳过这个问题") == True
    assert _user_declined_slot("不方便透露") == True
    assert _user_declined_slot("不知道") == True
    assert _user_declined_slot("不确定") == True
    assert _user_declined_slot("算了不说了") == True

    # 不应该检测为拒绝
    assert _user_declined_slot("我今年25岁") == False
    assert _user_declined_slot("头疼三天了") == False
    assert _user_declined_slot("") == False
    assert _user_declined_slot("好的我来回答") == False


def test_slot_ask_counts_initialized(monkeypatch):
    """测试 slot_ask_counts 字段初始化"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import AgentSessionState

    sess = AgentSessionState(session_id="test-123")

    assert hasattr(sess, "slot_ask_counts")
    assert isinstance(sess.slot_ask_counts, dict)
    assert len(sess.slot_ask_counts) == 0


def test_slot_ask_counts_updated_after_question(monkeypatch):
    """测试追问后 slot_ask_counts 更新"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import AgentSessionState, Slots
    from app.agent.graph import _build_structured_questions

    sess = AgentSessionState(session_id="test-123")
    sess.slots = Slots()  # 空槽位，会触发追问

    ask_text, questions, next_questions = _build_structured_questions(sess, "我不舒服")

    # 应该有追问
    assert len(questions) > 0

    # slot_ask_counts 应该更新
    for q in questions:
        slot = q.get("slot")
        if slot:
            assert sess.slot_ask_counts.get(slot, 0) >= 1


def test_slot_not_asked_after_max_attempts(monkeypatch):
    """测试达到最大追问次数后不再追问该槽位"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import AgentSessionState, Slots
    from app.agent.graph import _missing_slots, MAX_ASK_PER_SLOT

    sess = AgentSessionState(session_id="test-123")
    sess.slots = Slots()  # 空槽位

    # 模拟 age 已追问达到上限
    sess.slot_ask_counts["age"] = MAX_ASK_PER_SLOT

    missing = _missing_slots(sess.slots, "我不舒服", sess)

    # age 不应该在 missing 中
    assert "age" not in missing


def test_max_ask_per_slot_constant_exists(monkeypatch):
    """测试 MAX_ASK_PER_SLOT 常量存在且合理"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import MAX_ASK_PER_SLOT

    assert isinstance(MAX_ASK_PER_SLOT, int)
    assert MAX_ASK_PER_SLOT >= 1
    assert MAX_ASK_PER_SLOT <= 5  # 合理范围


def test_declined_slot_skipped(monkeypatch):
    """测试用户拒绝后跳过该槽位"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import AgentSessionState, Slots
    from app.agent.graph import _missing_slots

    sess = AgentSessionState(session_id="test-123")
    sess.slots = Slots()  # 空槽位
    sess.asked_slots = ["age", "symptoms"]  # 模拟上一轮追问了这些

    # 用户回复"不想说" - 应该跳过上一轮追问的槽位
    missing = _missing_slots(sess.slots, "不想说", sess)

    # 最近追问的槽位应该被跳过
    assert "age" not in missing
    assert "symptoms" not in missing


def test_missing_slots_without_sess_still_works(monkeypatch):
    """测试不传 sess 时仍然正常工作（向后兼容）"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import Slots
    from app.agent.graph import _missing_slots

    slots = Slots()  # 空槽位

    # 不传 sess，应该不报错
    missing = _missing_slots(slots, "我不舒服")

    assert isinstance(missing, list)
    assert len(missing) <= 3
