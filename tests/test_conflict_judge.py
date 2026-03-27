# -*- coding: utf-8 -*-

from __future__ import annotations


def test_judge_text_conflicts_confirms_positive_medication_recommendation():
    from app.safety.conflict_judge import judge_text_conflicts

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]

    confirmed, dismissed = judge_text_conflicts("建议先口服阿莫西林，每天三次。", conflicts)

    assert len(confirmed) == 1
    assert dismissed == []


def test_judge_text_conflicts_dismisses_negative_warning_context():
    from app.safety.conflict_judge import judge_text_conflicts

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]

    confirmed, dismissed = judge_text_conflicts("你对青霉素过敏，因此应避免阿莫西林。", conflicts)

    assert confirmed == []
    assert len(dismissed) == 1


def test_judge_json_conflicts_only_confirms_immediate_actions():
    from app.safety.conflict_judge import judge_json_conflicts

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]

    answer_json = {
        "immediate_actions": [],
        "what_not_to_do": ["不要自行服用阿莫西林。"],
        "safety_notice": "因青霉素过敏，应避免阿莫西林。",
    }

    confirmed, dismissed = judge_json_conflicts(answer_json, conflicts)

    assert confirmed == []
    assert len(dismissed) == 1
