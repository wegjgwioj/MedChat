# -*- coding: utf-8 -*-

from __future__ import annotations


def test_rule_extract_slots_parses_chinese_liang_duration():
    from app.agent.graph import _rule_extract_slots

    slots = _rule_extract_slots("我已经头痛两天了")

    assert slots.duration == "两天"


def test_rule_extract_slots_parses_severity_without_spaces():
    from app.agent.graph import _rule_extract_slots

    slots = _rule_extract_slots("头痛，严重程度6/10")

    assert slots.severity == "6/10"
