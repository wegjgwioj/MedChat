# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import threading
from typing import Dict, List, Sequence, Tuple

from app.agent.state import LongitudinalRecordFact, Slots, utc_now_iso
from app.rag.utils.rag_shared import make_embeddings

_EMBED_LOCK = threading.Lock()
_EMBEDDINGS = None


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _min_importance_threshold() -> float:
    return min(1.0, max(0.0, _env_float("AGENT_RECORD_MIN_IMPORTANCE", 0.55)))


def _similarity_threshold() -> float:
    return min(1.0, max(0.0, _env_float("AGENT_RECORD_SIM_THRESHOLD", 0.92)))


def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    with _EMBED_LOCK:
        if _EMBEDDINGS is None:
            _EMBEDDINGS, _ = make_embeddings()
        return _EMBEDDINGS


def _embed_text(text: str) -> List[float]:
    embeddings = _get_embeddings()
    if hasattr(embeddings, "embed_query"):
        return [float(x) for x in embeddings.embed_query(text)]
    if hasattr(embeddings, "embed_documents"):
        docs = embeddings.embed_documents([text])
        if docs and isinstance(docs[0], Sequence):
            return [float(x) for x in docs[0]]
    raise RuntimeError("记录索引 embedding 接口不可用。")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise RuntimeError("记录索引向量维度不一致。")

    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(a) * float(a) for a in left))
    norm_right = math.sqrt(sum(float(b) * float(b) for b in right))
    if norm_left <= 0 or norm_right <= 0:
        return 0.0
    return dot / (norm_left * norm_right)


def _compute_record_similarity(left: str, right: str) -> float:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    return _cosine_similarity(_embed_text(left_text), _embed_text(right_text))


def _candidate_specs(slots: Slots) -> List[Tuple[str, str, str, float]]:
    out: List[Tuple[str, str, str, float]] = []

    if slots.age is not None:
        out.append(("demographic", "年龄", f"{int(slots.age)}岁", 0.65))

    sex = str(slots.sex or "").strip()
    if sex:
        out.append(("demographic", "性别", sex, 0.55))

    pregnancy = str(slots.pregnancy or "").strip()
    if pregnancy and pregnancy != "unknown":
        out.append(("pregnancy", "妊娠", pregnancy, 0.85))

    history = str(slots.history or "").strip()
    if history:
        out.append(("history", "既往史", history, 0.95))

    meds = str(slots.meds or "").strip()
    if meds:
        out.append(("medication", "用药", meds, 0.80))

    allergy = str(slots.allergy or "").strip()
    if allergy:
        out.append(("allergy", "过敏", allergy, 0.98))

    return out


def build_record_candidates_from_slots(slots: Slots) -> List[LongitudinalRecordFact]:
    candidates: List[LongitudinalRecordFact] = []
    for category, label, value, importance in _candidate_specs(slots):
        text = f"{label}：{value}"
        candidates.append(
            LongitudinalRecordFact(
                category=category,  # type: ignore[arg-type]
                label=label,
                value=value,
                text=text,
                importance_score=importance,
            )
        )
    return candidates


def _should_replace(existing: LongitudinalRecordFact, candidate: LongitudinalRecordFact) -> bool:
    if candidate.importance_score > existing.importance_score:
        return True
    if math.isclose(candidate.importance_score, existing.importance_score) and len(candidate.value) > len(existing.value):
        return True
    return False


def upsert_longitudinal_records(
    existing_records: Sequence[LongitudinalRecordFact],
    slots: Slots,
) -> Tuple[List[LongitudinalRecordFact], Dict[str, int]]:
    records = [record.model_copy(deep=True) for record in existing_records]
    stats = {"added": 0, "merged": 0, "skipped": 0}
    threshold = _min_importance_threshold()
    sim_threshold = _similarity_threshold()

    for candidate in build_record_candidates_from_slots(slots):
        if candidate.importance_score < threshold:
            stats["skipped"] += 1
            continue

        merged = False
        for idx, existing in enumerate(records):
            if existing.category != candidate.category:
                continue
            similarity = _compute_record_similarity(existing.text, candidate.text)
            if similarity < sim_threshold:
                continue

            merged = True
            stats["merged"] += 1
            updated = existing.model_copy(deep=True)
            if _should_replace(existing, candidate):
                updated.label = candidate.label
                updated.value = candidate.value
                updated.text = candidate.text
                updated.importance_score = candidate.importance_score
            updated.updated_at = utc_now_iso()
            records[idx] = updated
            break

        if merged:
            continue

        candidate.updated_at = utc_now_iso()
        records.append(candidate)
        stats["added"] += 1

    order = {"过敏": 0, "既往史": 1, "用药": 2, "妊娠": 3, "年龄": 4, "性别": 5}
    records.sort(key=lambda item: (order.get(item.label, 99), -float(item.importance_score), item.updated_at))
    return records, stats


def build_record_summary_from_records(records: Sequence[LongitudinalRecordFact]) -> str:
    parts: List[str] = []
    for record in records:
        label = str(record.label or "").strip()
        value = str(record.value or "").strip()
        if not label or not value:
            continue
        if label == "年龄":
            parts.append(f"{label}{value}")
            continue
        if label == "性别":
            parts.append(f"{label}{value}")
            continue
        parts.append(f"{label}：{value}")
    return "；".join(parts).strip()
