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
