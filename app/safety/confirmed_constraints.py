# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List


_DRUG_ALLERGY_RULES = [
    {
        "rule_id": "drug-allergy-penicillin",
        "record_terms": ["青霉素过敏", "对青霉素过敏"],
        "unsafe_terms": ["青霉素", "阿莫西林", "氨苄西林", "阿莫西林克拉维酸", "哌拉西林"],
        "warning_message": "你已确认对青霉素过敏，应避免阿莫西林等青霉素类药物。",
    },
]


def _record_text(record: Any) -> str:
    parts = [
        str(getattr(record, "label", "") or "").strip(),
        str(getattr(record, "value", "") or "").strip(),
        str(getattr(record, "text", "") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def build_confirmed_constraints(records: Iterable[Any]) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    seen_rule_ids = set()

    for record in records or []:
        if str(getattr(record, "category", "") or "").strip() != "allergy":
            continue

        record_text = _record_text(record)
        if not record_text:
            continue

        for rule in _DRUG_ALLERGY_RULES:
            if rule["rule_id"] in seen_rule_ids:
                continue
            if not any(term in record_text for term in rule["record_terms"]):
                continue

            seen_rule_ids.add(rule["rule_id"])
            constraints.append(
                {
                    "constraint_id": str(rule["rule_id"]),
                    "constraint_type": "drug_allergy",
                    "source_fact_value": str(getattr(record, "value", "") or "").strip(),
                    "unsafe_terms": deepcopy(rule["unsafe_terms"]),
                    "warning_message": str(rule["warning_message"]),
                }
            )

    return constraints
