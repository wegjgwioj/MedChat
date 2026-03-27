# -*- coding: utf-8 -*-

from __future__ import annotations


def test_build_record_summary_keeps_longitudinal_fields():
    from app.agent.state import Slots
    from app.safety.record_guard import build_record_summary_from_slots

    slots = Slots(
        age=28,
        sex="女",
        symptoms=["头痛", "恶心"],
        history="哮喘",
        meds="维生素",
        allergy="青霉素过敏",
        pregnancy="no",
    )

    summary = build_record_summary_from_slots(slots)

    assert "年龄28岁" in summary
    assert "性别女" in summary
    assert "既往史：哮喘" in summary
    assert "用药：维生素" in summary
    assert "过敏：青霉素过敏" in summary
    assert "症状" not in summary


def test_detect_record_conflicts_hits_penicillin_allergy():
    from app.safety.record_guard import detect_record_conflicts

    conflicts = detect_record_conflicts(
        "建议先口服阿莫西林并观察体温变化。",
        "年龄28岁；过敏：青霉素过敏；既往史：哮喘",
    )

    assert len(conflicts) == 1
    assert conflicts[0]["category"] == "drug_allergy"
    assert conflicts[0]["record_term"] == "青霉素过敏"
    assert conflicts[0]["matched_term"] == "阿莫西林"


def test_apply_record_conflicts_to_answer_text_appends_warning_once():
    from app.safety.record_guard import apply_record_conflicts_to_answer_text

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]

    answer = "建议先口服阿莫西林，多喝水。"
    updated = apply_record_conflicts_to_answer_text(answer, conflicts)
    updated_twice = apply_record_conflicts_to_answer_text(updated, conflicts)

    assert "既往记录提示青霉素过敏" in updated
    assert updated.count("既往记录提示青霉素过敏") == 1
    assert updated_twice.count("既往记录提示青霉素过敏") == 1


def test_apply_record_conflicts_to_triage_json_rewrites_actions_and_uncertainty():
    from app.safety.record_guard import apply_record_conflicts_to_triage_json

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]
    answer_json = {
        "triage_level": "ROUTINE",
        "red_flags": [],
        "immediate_actions": ["可先口服阿莫西林，每天三次。"],
        "what_not_to_do": [],
        "key_questions": [],
        "reasoning": "考虑上呼吸道感染。",
        "uncertainty": "",
        "safety_notice": "本回答仅供参考。",
        "citations_used": [],
    }

    updated = apply_record_conflicts_to_triage_json(answer_json, conflicts)

    assert updated["immediate_actions"] == []
    assert "record_conflict" in updated["uncertainty"]
    assert "青霉素过敏" in updated["safety_notice"]
    assert updated["record_conflicts"] == conflicts
