# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from app.agent.state import LongitudinalRecordFact
from app import triage_service


def _confirmed_penicillin_allergy() -> LongitudinalRecordFact:
    return LongitudinalRecordFact(
        category="allergy",
        label="过敏",
        value="青霉素过敏",
        text="过敏：青霉素过敏",
        importance_score=0.98,
    )


def _dangerous_answer_json() -> dict:
    return {
        "triage_level": "ROUTINE",
        "red_flags": [],
        "immediate_actions": ["建议先口服阿莫西林。"],
        "what_not_to_do": [],
        "key_questions": [],
        "reasoning": "考虑上呼吸道感染。",
        "uncertainty": "",
        "safety_notice": "本回答仅供参考。",
        "citations_used": [],
    }


def _stub_engine(monkeypatch) -> None:
    monkeypatch.setattr(triage_service._ENGINE, "init", lambda: None)
    monkeypatch.setattr(
        triage_service._ENGINE,
        "retrieve_evidence",
        lambda rag_query, top_k, trace=None: ([], "disabled", None),
    )
    monkeypatch.setattr(
        triage_service._ENGINE,
        "suggest_answer_raw",
        lambda user_text, evidence_block, trace=None: "{}",
    )
    monkeypatch.setattr(
        triage_service._ENGINE,
        "ensure_answer_json",
        lambda raw_text, evidence_block, evidence_list, trace=None, trace_step="answer.ensure_json": _dangerous_answer_json(),
    )


def test_triage_blocks_confirmed_conflicting_medication_and_writes_trace(monkeypatch):
    _stub_engine(monkeypatch)

    payload = triage_service.triage_once(
        user_text="喉咙痛两天",
        top_k=1,
        mode="fast",
        longitudinal_records=[_confirmed_penicillin_allergy()],
    )

    answer = payload["answer"]
    joined_actions = "\n".join(answer["immediate_actions"])
    assert "阿莫西林" not in joined_actions
    assert "青霉素过敏" in joined_actions
    assert answer["record_conflicts"][0]["medication_name"] == "阿莫西林"

    safety_step = next(item for item in payload["meta"]["trace"] if item["step"] == "record.safety")
    assert safety_step["constraint_count"] == 1
    assert safety_step["blocked_count"] == 1


def test_triage_does_not_block_on_unconfirmed_external_record(monkeypatch, tmp_path: Path):
    _stub_engine(monkeypatch)
    clinical_record = tmp_path / "record.txt"
    clinical_record.write_text("病历提示：青霉素过敏。", encoding="utf-8")

    payload = triage_service.triage_once(
        user_text="喉咙痛两天",
        top_k=1,
        mode="fast",
        clinical_record_path=str(clinical_record),
    )

    answer = payload["answer"]
    assert "阿莫西林" in "\n".join(answer["immediate_actions"])
    assert answer["record_conflicts"] == []

    safety_step = next(item for item in payload["meta"]["trace"] if item["step"] == "record.safety")
    assert safety_step["status"] == "disabled_unconfirmed_source"
