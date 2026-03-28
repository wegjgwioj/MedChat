from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _env_int(name: str, default: int) -> int:
    try:
        raw = str(os.getenv(name, str(default)) or str(default)).strip()
        value = int(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        raw = str(os.getenv(name, str(default)) or str(default)).strip()
        value = float(raw)
        return value if value >= 0 else default
    except Exception:
        return default


def summarize_evidence_quality(evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize whether current evidence is enough for a confident answer."""

    items = list(evidence_list or [])
    count = len(items)

    min_hits = _env_int("RAG_MIN_EVIDENCE", 2)
    min_rerank_score = _env_float("RAG_RERANK_MIN_SCORE", 0.0)

    rerank_scores: List[float] = []
    for item in items:
        value = item.get("rerank_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rerank_scores.append(float(value))

    best_rerank_score: Optional[float] = max(rerank_scores) if rerank_scores else None

    level = "ok"
    reason = "ok"
    if count <= 0:
        level = "none"
        reason = "no_hits"
    elif count < min_hits:
        level = "low"
        reason = "too_few_hits"
    elif min_rerank_score > 0 and best_rerank_score is not None and best_rerank_score < min_rerank_score:
        level = "low"
        reason = "low_rerank_score"

    return {
        "level": level,
        "reason": reason,
        "count": count,
        "min_hits": min_hits,
        "min_rerank_score": min_rerank_score,
        "best_rerank_score": best_rerank_score,
    }


def is_low_evidence(summary: Dict[str, Any]) -> bool:
    return str((summary or {}).get("level") or "").strip().lower() in {"low", "none"}
