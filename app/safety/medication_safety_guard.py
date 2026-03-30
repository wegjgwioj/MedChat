# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List


_KNOWN_MEDICATIONS = [
    "阿莫西林克拉维酸",
    "对乙酰氨基酚",
    "氨苄西林",
    "哌拉西林",
    "阿莫西林",
    "青霉素",
]
_NEGATIVE_CUES = ("避免", "不要", "禁用", "勿用", "不建议", "不可", "不能用")
_POSITIVE_CUES = ("建议", "可", "可以", "可用", "可考虑", "先口服", "服用", "使用")


def _split_sentences(answer: str) -> List[str]:
    parts = re.split(r"[\r\n。；;]+", str(answer or "").strip())
    return [part.strip(" ，,") for part in parts if part.strip(" ，,")]


def _contains_negative_cue(text: str, medication_name: str) -> bool:
    med_index = text.find(medication_name)
    if med_index < 0:
        return False

    window = text[max(0, med_index - 8) : med_index + len(medication_name) + 8]
    return any(cue in window for cue in _NEGATIVE_CUES)


def _contains_positive_cue(text: str) -> bool:
    return any(cue in text for cue in _POSITIVE_CUES)


def extract_medication_candidates_from_answer(answer: str) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    seen_names = set()

    for sentence in _split_sentences(answer):
        if not _contains_positive_cue(sentence):
            continue

        for medication_name in _KNOWN_MEDICATIONS:
            if medication_name not in sentence:
                continue
            if _contains_negative_cue(sentence, medication_name):
                continue
            if medication_name in seen_names:
                continue

            seen_names.add(medication_name)
            candidates.append({"name": medication_name, "text": sentence})

    return candidates


def guard_medication_candidates(candidates, constraints) -> Dict[str, Any]:
    allowed: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    warnings: List[str] = []
    conflicts: List[Dict[str, Any]] = []

    for candidate in candidates or []:
        candidate_name = str(candidate.get("name") or "").strip()
        candidate_text = str(candidate.get("text") or "").strip()
        matched_constraint = None

        for constraint in constraints or []:
            unsafe_terms = [str(term or "").strip() for term in constraint.get("unsafe_terms", []) if str(term or "").strip()]
            if any(term and (term in candidate_name or term in candidate_text) for term in unsafe_terms):
                matched_constraint = constraint
                break

        if matched_constraint is None:
            allowed.append(deepcopy(candidate))
            continue

        blocked.append(deepcopy(candidate))

        warning_message = str(matched_constraint.get("warning_message") or "").strip()
        if warning_message and warning_message not in warnings:
            warnings.append(warning_message)

        conflicts.append(
            {
                "constraint_id": str(matched_constraint.get("constraint_id") or "").strip(),
                "constraint_type": str(matched_constraint.get("constraint_type") or "").strip(),
                "medication_name": candidate_name,
                "source_fact_value": str(matched_constraint.get("source_fact_value") or "").strip(),
                "warning_message": warning_message,
            }
        )

    return {
        "allowed_medications": allowed,
        "blocked_medications": blocked,
        "warnings": warnings,
        "conflicts": conflicts,
    }
