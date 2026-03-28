# -*- coding: utf-8 -*-
"""记录冲突的模型化判别器（NLI/SLM）。"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_JUDGE_MODEL: Any = None


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _judge_threshold() -> float:
    return _env_float("AGENT_CONFLICT_JUDGE_THRESHOLD", 0.5)


def _split_text_units(text: str) -> List[str]:
    parts = re.split(r"[。！？!?；;\n\r]+", str(text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def _default_model_name() -> str:
    return str(os.getenv("AGENT_CONFLICT_JUDGE_MODEL") or "cross-encoder/nli-deberta-v3-base").strip()


def _get_judge_model() -> Any:
    global _JUDGE_MODEL
    if _JUDGE_MODEL is not None:
        return _JUDGE_MODEL
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少 sentence-transformers，无法加载冲突判别模型。") from e
    try:
        _JUDGE_MODEL = CrossEncoder(_default_model_name())
    except Exception as e:
        raise RuntimeError(f"冲突判别模型加载失败：{type(e).__name__}: {e}") from e
    return _JUDGE_MODEL


def _build_premise(conflict: Dict[str, Any]) -> str:
    category = str(conflict.get("category") or "").strip()
    record_term = str(conflict.get("record_term") or "").strip()
    message = str(conflict.get("message") or "").strip()
    parts = [p for p in [category, record_term, message] if p]
    return "；".join(parts) if parts else message


def _as_float_score(raw: Any) -> float:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if not raw:
            raise RuntimeError("冲突判别模型返回了空分数向量。")
        # 约定取最后一个维度作为“冲突成立”分数，便于兼容二分类/三分类输出。
        tail = raw[-1]
        if isinstance(tail, (int, float)) and not isinstance(tail, bool):
            return float(tail)
    raise RuntimeError("冲突判别模型返回了不可解析的分数格式。")


def _predict_conflict_scores(premise: str, hypotheses: Iterable[str]) -> List[float]:
    cleaned = [str(h).strip() for h in hypotheses if str(h).strip()]
    if not cleaned:
        return []
    model = _get_judge_model()
    pairs = [(premise, hypo) for hypo in cleaned]
    try:
        outputs = model.predict(pairs)
    except Exception as e:
        raise RuntimeError(f"冲突判别推理失败：{type(e).__name__}: {e}") from e
    try:
        return [_as_float_score(item) for item in list(outputs)]
    except Exception as e:
        raise RuntimeError(f"冲突判别分数解析失败：{type(e).__name__}: {e}") from e


def _is_conflict(premise: str, hypotheses: List[str]) -> bool:
    scores = _predict_conflict_scores(premise, hypotheses)
    if not scores:
        return False
    return max(scores) >= _judge_threshold()


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

        premise = _build_premise(conflict)
        if not _is_conflict(premise, matched_units):
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

        premise = _build_premise(conflict)
        if not _is_conflict(premise, matched_actions):
            dismissed.append(deepcopy(conflict))
            continue

        confirmed.append(deepcopy(conflict))

    return confirmed, dismissed
