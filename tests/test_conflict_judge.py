# -*- coding: utf-8 -*-

from __future__ import annotations


class _FakeJudgeModel:
    def __init__(self, score_map):
        self._score_map = score_map
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        out = []
        for _, hypothesis in pairs:
            out.append(self._score_map.get(hypothesis, 0.0))
        return out


def test_judge_text_conflicts_confirms_positive_medication_recommendation():
    from app.safety import conflict_judge

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]

    text = "建议先口服阿莫西林，每天三次。"
    fake = _FakeJudgeModel({text.rstrip("。"): 0.95, text: 0.95})

    conflict_judge._JUDGE_MODEL = fake
    confirmed, dismissed = conflict_judge.judge_text_conflicts(text, conflicts)

    assert len(confirmed) == 1
    assert dismissed == []
    assert len(fake.calls) == 1


def test_judge_text_conflicts_dismisses_when_model_score_is_below_threshold(monkeypatch):
    from app.safety import conflict_judge

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]

    text = "你对青霉素过敏，因此应避免阿莫西林。"
    fake = _FakeJudgeModel({text.rstrip("。"): 0.05, text: 0.05})

    monkeypatch.setenv("AGENT_CONFLICT_JUDGE_THRESHOLD", "0.5")
    conflict_judge._JUDGE_MODEL = fake
    confirmed, dismissed = conflict_judge.judge_text_conflicts(text, conflicts)

    assert confirmed == []
    assert len(dismissed) == 1
    assert len(fake.calls) == 1


def test_judge_json_conflicts_only_confirms_immediate_actions():
    from app.safety import conflict_judge

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]

    answer_json = {
        "immediate_actions": [],
        "what_not_to_do": ["不要自行服用阿莫西林。"],
        "safety_notice": "因青霉素过敏，应避免阿莫西林。",
    }

    fake = _FakeJudgeModel({"不要自行服用阿莫西林": 0.99})
    conflict_judge._JUDGE_MODEL = fake
    confirmed, dismissed = conflict_judge.judge_json_conflicts(answer_json, conflicts)

    assert confirmed == []
    assert len(dismissed) == 1
    assert fake.calls == []


def test_judge_json_conflicts_confirms_when_immediate_action_score_high():
    from app.safety import conflict_judge

    conflicts = [
        {
            "category": "drug_allergy",
            "record_term": "青霉素过敏",
            "matched_term": "阿莫西林",
            "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。",
        }
    ]
    answer_json = {"immediate_actions": ["建议先口服阿莫西林。"]}
    action = "建议先口服阿莫西林。"
    fake = _FakeJudgeModel({action.rstrip("。"): 0.8, action: 0.8})

    conflict_judge._JUDGE_MODEL = fake
    confirmed, dismissed = conflict_judge.judge_json_conflicts(answer_json, conflicts)

    assert len(confirmed) == 1
    assert dismissed == []
    assert len(fake.calls) == 1
