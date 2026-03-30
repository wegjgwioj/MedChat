# -*- coding: utf-8 -*-

from __future__ import annotations


def test_medication_safety_guard_blocks_conflicting_medication():
    from app.safety.medication_safety_guard import guard_medication_candidates

    candidates = [
        {"name": "阿莫西林", "text": "可先口服阿莫西林，每天三次。"},
        {"name": "对乙酰氨基酚", "text": "如发热可考虑对乙酰氨基酚。"},
    ]
    constraints = [
        {
            "constraint_id": "c1",
            "constraint_type": "drug_allergy",
            "unsafe_terms": ["阿莫西林"],
            "warning_message": "你已确认对青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]

    result = guard_medication_candidates(candidates, constraints)

    assert [item["name"] for item in result["allowed_medications"]] == ["对乙酰氨基酚"]
    assert [item["name"] for item in result["blocked_medications"]] == ["阿莫西林"]
    assert result["warnings"] == ["你已确认对青霉素过敏，应避免阿莫西林等青霉素类药物。"]


def test_extract_medication_candidates_does_not_treat_negative_warning_as_positive():
    from app.safety.medication_safety_guard import extract_medication_candidates_from_answer

    answer = "因青霉素过敏，应避免阿莫西林；如发热可用对乙酰氨基酚。"
    candidates = extract_medication_candidates_from_answer(answer)

    assert [item["name"] for item in candidates] == ["对乙酰氨基酚"]
