# -*- coding: utf-8 -*-

from __future__ import annotations


def _install_fake_engine(monkeypatch):
    from app import triage_service

    engine = triage_service.TriageEngine()
    monkeypatch.setattr(engine, "init", lambda: None)
    monkeypatch.setattr(
        engine,
        "retrieve_evidence",
        lambda rag_query, top_k, trace=None: ([], "ok", None),
    )
    monkeypatch.setattr(
        engine,
        "suggest_answer_raw",
        lambda user_text, evidence_block, trace=None: "{}",
    )
    monkeypatch.setattr(
        engine,
        "ensure_answer_json",
        lambda raw_text, evidence_block, evidence_list, trace=None, trace_step="answer.ensure_json": {
            "triage_level": "ROUTINE",
            "red_flags": [],
            "immediate_actions": ["可先口服阿莫西林，每天三次。"],
            "what_not_to_do": [],
            "key_questions": [],
            "reasoning": "考虑上呼吸道感染。",
            "uncertainty": "",
            "safety_notice": "本回答仅供参考。",
            "citations_used": [],
        },
    )
    monkeypatch.setattr(
        engine,
        "run_safety_chain",
        lambda answer_json, evidence_block, evidence_list, trace=None: answer_json,
    )
    monkeypatch.setattr(triage_service, "_ENGINE", engine, raising=False)
    return triage_service


def test_triage_once_does_not_apply_record_veto_from_clinical_record_path(monkeypatch, tmp_path):
    triage_service = _install_fake_engine(monkeypatch)

    record_path = tmp_path / "record.txt"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("过敏：青霉素过敏", encoding="utf-8")

    payload = triage_service.triage_once(
        user_text="喉咙痛怎么办",
        clinical_record_path=str(record_path),
    )

    answer = payload["answer"]
    assert answer["immediate_actions"] == ["可先口服阿莫西林，每天三次。"]
    assert answer.get("record_conflicts", []) == []


def test_triage_once_trace_marks_external_record_veto_as_disabled(monkeypatch, tmp_path):
    triage_service = _install_fake_engine(monkeypatch)

    record_path = tmp_path / "record.txt"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("过敏：青霉素过敏", encoding="utf-8")

    payload = triage_service.triage_once(
        user_text="喉咙痛怎么办",
        clinical_record_path=str(record_path),
    )

    trace = payload["meta"]["trace"]
    assert any(step.get("step") == "record.safety" for step in trace)
    assert any(step.get("status") == "disabled_unconfirmed_source" for step in trace if step.get("step") == "record.safety")


def test_triage_once_does_not_treat_negative_warning_fields_as_positive_conflict(monkeypatch, tmp_path):
    from app import triage_service
    import app.safety.conflict_judge as conflict_judge

    engine = triage_service.TriageEngine()
    monkeypatch.setattr(engine, "init", lambda: None)
    monkeypatch.setattr(engine, "retrieve_evidence", lambda rag_query, top_k, trace=None: ([], "ok", None))
    monkeypatch.setattr(engine, "suggest_answer_raw", lambda user_text, evidence_block, trace=None: "{}")
    monkeypatch.setattr(
        engine,
        "ensure_answer_json",
        lambda raw_text, evidence_block, evidence_list, trace=None, trace_step="answer.ensure_json": {
            "triage_level": "ROUTINE",
            "red_flags": [],
            "immediate_actions": [],
            "what_not_to_do": ["不要自行服用阿莫西林。"],
            "key_questions": [],
            "reasoning": "考虑上呼吸道感染。",
            "uncertainty": "",
            "safety_notice": "因青霉素过敏，应避免阿莫西林。",
            "citations_used": [],
        },
    )
    monkeypatch.setattr(engine, "run_safety_chain", lambda answer_json, evidence_block, evidence_list, trace=None: answer_json)
    monkeypatch.setattr(triage_service, "_ENGINE", engine, raising=False)
    monkeypatch.setattr(conflict_judge, "_predict_conflict_scores", lambda premise, hypotheses: [0.05 for _ in hypotheses])

    record_path = tmp_path / "record.txt"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("过敏：青霉素过敏", encoding="utf-8")

    payload = triage_service.triage_once(
        user_text="喉咙痛怎么办",
        clinical_record_path=str(record_path),
    )

    answer = payload["answer"]
    assert "record_conflict" not in answer["uncertainty"]
    assert answer["record_conflicts"] == []
