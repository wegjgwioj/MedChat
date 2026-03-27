# -*- coding: utf-8 -*-
"""记录冲突的二次判别器。

目的：
- 保留 lexical detector 的高召回
- 在 apply 之前过滤“否定/警示”上下文，减少误杀
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple


_NEGATIVE_CUES = ["避免", "不要", "不能", "禁用", "慎用", "勿用", "不建议", "过敏", "禁忌"]
_POSITIVE_CUES = ["建议", "可先", "可以", "口服", "服用", "使用", "先用", "吃"]


def _split_text_units(text: str) -> List[str]:
    parts = re.split(r"[。！？!?；;\n\r]+", str(text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def _has_any_cue(text: str, cues: List[str]) -> bool:
    content = str(text or "")
    return any(cue in content for cue in cues)


def judge_text_conflicts(answer_text: str, conflicts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    units = _split_text_units(answer_text)
    confirmed: List[Dict[str, Any]] = []
    dismissed: List[Dict[str, Any]] = []

    for conflict in conflicts or []:
        term = str(conflict.get("matched_term") or "").strip()
        if not term:
            dismissed.append(deepcopy(conflict))
            continue

        matched_units = [unit for unit in units if term in unit]
        if not matched_units:
            dismissed.append(deepcopy(conflict))
            continue

        has_negative = any(_has_any_cue(unit, _NEGATIVE_CUES) for unit in matched_units)
        has_positive = any(_has_any_cue(unit, _POSITIVE_CUES) for unit in matched_units)

        if has_negative and not has_positive:
            dismissed.append(deepcopy(conflict))
            continue

        confirmed.append(deepcopy(conflict))

    return confirmed, dismissed


def judge_json_conflicts(answer_json: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    actions = [str(item) for item in (answer_json or {}).get("immediate_actions") or [] if str(item).strip()]

    confirmed: List[Dict[str, Any]] = []
    dismissed: List[Dict[str, Any]] = []

    for conflict in conflicts or []:
        term = str(conflict.get("matched_term") or "").strip()
        if not term:
            dismissed.append(deepcopy(conflict))
            continue

        matched_actions = [action for action in actions if term in action]
        if not matched_actions:
            dismissed.append(deepcopy(conflict))
            continue

        if any(_has_any_cue(action, _NEGATIVE_CUES) for action in matched_actions):
            dismissed.append(deepcopy(conflict))
            continue

        confirmed.append(deepcopy(conflict))

    return confirmed, dismissed
