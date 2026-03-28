# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


_ALLERGY_RULES = [
    {
        "label": "青霉素过敏",
        "record_terms": ["青霉素", "盘尼西林", "阿莫西林", "氨苄西林", "阿莫西林克拉维酸"],
        "unsafe_terms": ["青霉素", "阿莫西林", "氨苄西林", "阿莫西林克拉维酸", "哌拉西林"],
        "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
    },
    {
        "label": "布洛芬过敏",
        "record_terms": ["布洛芬", "NSAID", "非甾体抗炎药"],
        "unsafe_terms": ["布洛芬", "双氯芬酸", "萘普生"],
        "message": "既往记录提示对布洛芬或同类止痛药过敏，应避免再次使用相关药物。",
    },
]


def build_record_summary_from_slots(slots: Any) -> str:
    parts: List[str] = []

    age = getattr(slots, "age", None)
    if age is not None:
        parts.append(f"年龄{int(age)}岁")

    sex = str(getattr(slots, "sex", "") or "").strip()
    if sex:
        parts.append(f"性别{sex}")

    pregnancy = str(getattr(slots, "pregnancy", "") or "").strip()
    if pregnancy and pregnancy != "unknown":
        parts.append(f"妊娠：{pregnancy}")

    history = str(getattr(slots, "history", "") or "").strip()
    if history:
        parts.append(f"既往史：{history}")

    meds = str(getattr(slots, "meds", "") or "").strip()
    if meds:
        parts.append(f"用药：{meds}")

    allergy = str(getattr(slots, "allergy", "") or "").strip()
    if allergy:
        parts.append(f"过敏：{allergy}")

    return "；".join(parts).strip()


def extract_record_constraints(record_text: str) -> List[Dict[str, Any]]:
    text = str(record_text or "").strip()
    if not text:
        return []

    constraints: List[Dict[str, Any]] = []
    for rule in _ALLERGY_RULES:
        matched = False
        for term in rule["record_terms"]:
            if (f"{term}过敏" in text) or ("过敏" in text and term in text):
                matched = True
                break
        if matched:
            constraints.append(
                {
                    "category": "drug_allergy",
                    "record_term": rule["label"],
                    "unsafe_terms": list(rule["unsafe_terms"]),
                    "message": rule["message"],
                }
            )
    return constraints


def detect_record_conflicts(text: str, record_text: str) -> List[Dict[str, Any]]:
    content = str(text or "").strip()
    if not content:
        return []

    conflicts: List[Dict[str, Any]] = []
    seen = set()
    for constraint in extract_record_constraints(record_text):
        for term in constraint["unsafe_terms"]:
            if term not in content:
                continue
            key = (constraint["record_term"], term)
            if key in seen:
                continue
            seen.add(key)
            conflicts.append(
                {
                    "category": str(constraint["category"]),
                    "record_term": str(constraint["record_term"]),
                    "matched_term": str(term),
                    "message": str(constraint["message"]),
                }
            )
    return conflicts


def apply_record_conflicts_to_answer_text(answer: str, conflicts: List[Dict[str, Any]]) -> str:
    text = str(answer or "").strip()
    if not conflicts:
        return text

    out = text
    for conflict in conflicts:
        message = str(conflict.get("message") or "").strip()
        if not message or message in out:
            continue
        prefix = "" if not out else "\n\n"
        out = f"{out}{prefix}注意：{message}"
    return out.strip()


def _append_tag(text: str, tag: str) -> str:
    parts = [p.strip() for p in str(text or "").split("|") if p.strip()]
    if tag not in parts:
        parts.append(tag)
    return " | ".join(parts)


def apply_record_conflicts_to_triage_json(answer_json: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = deepcopy(answer_json if isinstance(answer_json, dict) else {})
    if not conflicts:
        out["record_conflicts"] = []
        return out

    matched_terms = {str(item.get("matched_term") or "").strip() for item in conflicts if str(item.get("matched_term") or "").strip()}

    actions = out.get("immediate_actions")
    if isinstance(actions, list):
        out["immediate_actions"] = [
            str(item)
            for item in actions
            if not any(term in str(item) for term in matched_terms)
        ]

    what_not_to_do = out.get("what_not_to_do")
    if isinstance(what_not_to_do, list):
        notes = [str(item) for item in what_not_to_do]
    else:
        notes = []
    for conflict in conflicts:
        message = str(conflict.get("message") or "").strip()
        if message and message not in notes:
            notes.append(message)
    out["what_not_to_do"] = notes

    notice = str(out.get("safety_notice") or "").strip()
    for conflict in conflicts:
        message = str(conflict.get("message") or "").strip()
        if message and message not in notice:
            joiner = " " if notice else ""
            notice = f"{notice}{joiner}{message}".strip()
    out["safety_notice"] = notice
    out["uncertainty"] = _append_tag(str(out.get("uncertainty") or ""), "record_conflict")
    out["record_conflicts"] = deepcopy(conflicts)
    return out
