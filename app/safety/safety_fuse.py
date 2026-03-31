# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional

from app.safety.confirmed_constraints import build_confirmed_constraints
from app.safety.medication_safety_guard import (
    extract_medication_candidates_from_answer,
    guard_medication_candidates,
)


RewriteFn = Callable[[str, Dict[str, Any]], str]


def _split_sentences(answer: str) -> List[str]:
    parts = re.split(r"[\r\n。；;]+", str(answer or "").strip())
    return [part.strip(" ，,") for part in parts if part.strip(" ，,")]


def _compose_warning_lines(conflicts: Iterable[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    seen = set()
    for conflict in conflicts or []:
        source_fact_value = str(conflict.get("source_fact_value") or "").strip()
        if source_fact_value:
            line = f"你已确认{source_fact_value}，本轮已移除冲突用药建议，请线下咨询替代方案。"
        else:
            line = "系统已移除与已确认禁忌冲突的用药建议，请线下咨询替代方案。"
        if line not in seen:
            seen.add(line)
            warnings.append(line)
    return warnings


def apply_confirmed_safety_fuse_to_text(
    *,
    answer_text: str,
    longitudinal_records: Iterable[Any],
    rewrite_fn: Optional[RewriteFn] = None,
) -> Dict[str, Any]:
    original_answer = str(answer_text or "").strip()
    constraints = build_confirmed_constraints(longitudinal_records)
    candidates = extract_medication_candidates_from_answer(original_answer)
    guard_result = guard_medication_candidates(candidates, constraints)

    blocked_items = list(guard_result.get("blocked_medications", []) or [])
    blocked_texts = {
        str(item.get("text") or "").strip()
        for item in blocked_items
        if str(item.get("text") or "").strip()
    }
    kept_sentences = [
        sentence
        for sentence in _split_sentences(original_answer)
        if sentence not in blocked_texts
    ]

    warning_lines = _compose_warning_lines(guard_result.get("conflicts", []) or [])
    rewrite_used = False

    if not blocked_items:
        final_answer = original_answer
    else:
        merged_parts = warning_lines + kept_sentences
        filtered_answer = "；".join([part for part in merged_parts if part]).strip("； ")
        if not kept_sentences and callable(rewrite_fn):
            rewrite_used = True
            final_answer = str(rewrite_fn(original_answer, {"conflicts": guard_result.get("conflicts", []) or []}) or "").strip()
        elif filtered_answer:
            final_answer = filtered_answer
        else:
            final_answer = "；".join(warning_lines).strip("； ")

    trace = {
        "constraint_count": len(constraints),
        "candidate_count": len(candidates),
        "blocked_count": len(blocked_items),
        "warning_count": len(warning_lines),
        "model_judge_used": False,
        "rewrite_used": rewrite_used,
        "blocked_items": blocked_items,
        "conflicts": list(guard_result.get("conflicts", []) or []),
    }
    return {"answer": final_answer, "trace": trace}


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _merge_trace_items(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "constraint_count": 0,
        "candidate_count": 0,
        "blocked_count": 0,
        "warning_count": 0,
        "model_judge_used": False,
        "rewrite_used": False,
        "blocked_items": [],
        "conflicts": [],
    }
    seen_blocked = set()
    seen_conflicts = set()
    warning_messages = set()

    for item in items or []:
        trace = item if isinstance(item, dict) else {}
        merged["constraint_count"] = max(int(merged["constraint_count"]), int(trace.get("constraint_count") or 0))
        merged["candidate_count"] = int(merged["candidate_count"]) + int(trace.get("candidate_count") or 0)
        merged["blocked_count"] = int(merged["blocked_count"]) + int(trace.get("blocked_count") or 0)
        merged["model_judge_used"] = bool(merged["model_judge_used"] or trace.get("model_judge_used"))
        merged["rewrite_used"] = bool(merged["rewrite_used"] or trace.get("rewrite_used"))

        for blocked_item in trace.get("blocked_items", []) or []:
            key = (
                str(blocked_item.get("name") or "").strip(),
                str(blocked_item.get("text") or "").strip(),
            )
            if key in seen_blocked:
                continue
            seen_blocked.add(key)
            merged["blocked_items"].append(deepcopy(blocked_item))

        for conflict in trace.get("conflicts", []) or []:
            key = (
                str(conflict.get("constraint_id") or "").strip(),
                str(conflict.get("medication_name") or "").strip(),
                str(conflict.get("source_fact_value") or "").strip(),
            )
            if key in seen_conflicts:
                continue
            seen_conflicts.add(key)
            merged["conflicts"].append(deepcopy(conflict))
            warning_message = str(conflict.get("warning_message") or "").strip()
            if warning_message:
                warning_messages.add(warning_message)

    merged["warning_count"] = len(warning_messages)
    return merged


def apply_confirmed_safety_fuse_to_triage_answer(
    *,
    answer_json: Dict[str, Any],
    longitudinal_records: Iterable[Any],
    rewrite_fn: Optional[RewriteFn] = None,
) -> Dict[str, Any]:
    out = dict(answer_json if isinstance(answer_json, dict) else {})
    longitudinal_records = list(longitudinal_records or [])
    trace_items: List[Dict[str, Any]] = []

    for field_name in ("immediate_actions", "what_not_to_do"):
        fused_items: List[str] = []
        for text in _normalize_string_list(out.get(field_name)):
            fused = apply_confirmed_safety_fuse_to_text(
                answer_text=text,
                longitudinal_records=longitudinal_records,
                rewrite_fn=rewrite_fn,
            )
            trace_items.append(dict(fused.get("trace") or {}))
            fused_text = str(fused.get("answer") or "").strip()
            if fused_text:
                fused_items.append(fused_text)
        out[field_name] = fused_items

    reasoning = str(out.get("reasoning") or "").strip()
    reasoning_fused = apply_confirmed_safety_fuse_to_text(
        answer_text=reasoning,
        longitudinal_records=longitudinal_records,
        rewrite_fn=rewrite_fn,
    )
    trace_items.append(dict(reasoning_fused.get("trace") or {}))
    out["reasoning"] = str(reasoning_fused.get("answer") or "").strip()

    merged_trace = _merge_trace_items(trace_items)
    out["record_conflicts"] = list(merged_trace.get("conflicts", []) or [])
    return {"answer_json": out, "trace": merged_trace}
