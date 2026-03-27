# -*- coding: utf-8 -*-

from __future__ import annotations


def test_build_reply_from_questions_covers_known_slots():
    from scripts.selftest_agent_v2 import _build_reply_from_questions

    reply = _build_reply_from_questions(
        [
            {"slot": "age"},
            {"slot": "sex"},
            {"slot": "severity"},
            {"slot": "meds"},
            {"slot": "allergy"},
        ]
    )

    assert "24岁" in reply
    assert "男" in reply
    assert "6/10" in reply
    assert "没有用药" in reply
    assert "无药物过敏史" in reply


def test_build_reply_from_questions_has_fallback_for_unknown_slots():
    from scripts.selftest_agent_v2 import _build_reply_from_questions

    reply = _build_reply_from_questions([{"slot": "unknown_slot"}])

    assert isinstance(reply, str)
    assert reply.strip()


def test_is_valid_escalate_trace_accepts_safety_short_circuit():
    from scripts.selftest_agent_v2 import _is_valid_escalate_trace

    trace = {
        "node_order": ["SafetyGate", "PersistState"],
        "timings_ms": {"SafetyGate": 0, "PersistState": 3},
    }

    assert _is_valid_escalate_trace(trace) is True


def test_is_valid_escalate_trace_rejects_missing_persist():
    from scripts.selftest_agent_v2 import _is_valid_escalate_trace

    trace = {
        "node_order": ["SafetyGate"],
        "timings_ms": {"SafetyGate": 0},
    }

    assert _is_valid_escalate_trace(trace) is False
