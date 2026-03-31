# -*- coding: utf-8 -*-

from __future__ import annotations

from app.agent.state import LongitudinalRecordFact
from app.safety.safety_fuse import apply_confirmed_safety_fuse_to_text


def _confirmed_penicillin_allergy() -> LongitudinalRecordFact:
    return LongitudinalRecordFact(
        category="allergy",
        label="过敏",
        value="青霉素过敏",
        text="过敏：青霉素过敏",
        importance_score=0.98,
    )


def test_text_fuse_removes_conflicting_medication_and_adds_warning():
    answer = "建议先口服阿莫西林，多喝水休息。"

    out = apply_confirmed_safety_fuse_to_text(
        answer_text=answer,
        longitudinal_records=[_confirmed_penicillin_allergy()],
    )

    assert "阿莫西林" not in out["answer"]
    assert "青霉素过敏" in out["answer"]
    assert out["trace"]["constraint_count"] == 1
    assert out["trace"]["blocked_count"] == 1
    assert out["trace"]["rewrite_used"] is False


def test_text_fuse_keeps_safe_answer_when_no_constraints_match():
    answer = "建议多喝水，注意休息，必要时线下就诊。"

    out = apply_confirmed_safety_fuse_to_text(
        answer_text=answer,
        longitudinal_records=[_confirmed_penicillin_allergy()],
    )

    assert out["answer"] == answer
    assert out["trace"]["blocked_count"] == 0
    assert out["trace"]["warning_count"] == 0


def test_text_fuse_uses_rewrite_when_all_conflicting_content_is_removed():
    answer = "建议先口服阿莫西林。"

    out = apply_confirmed_safety_fuse_to_text(
        answer_text=answer,
        longitudinal_records=[_confirmed_penicillin_allergy()],
        rewrite_fn=lambda text, trace: "你已确认对青霉素过敏，应避免使用阿莫西林，请线下咨询可替代药物。",
    )

    assert out["answer"] == "你已确认对青霉素过敏，应避免使用阿莫西林，请线下咨询可替代药物。"
    assert out["trace"]["blocked_count"] == 1
    assert out["trace"]["rewrite_used"] is True
