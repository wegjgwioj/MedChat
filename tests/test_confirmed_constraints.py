# -*- coding: utf-8 -*-

from __future__ import annotations


def test_build_confirmed_constraints_extracts_drug_allergy():
    from app.agent.state import LongitudinalRecordFact
    from app.safety.confirmed_constraints import build_confirmed_constraints

    records = [
        LongitudinalRecordFact(
            category="allergy",
            label="过敏",
            value="青霉素过敏",
            text="过敏：青霉素过敏",
            importance_score=0.98,
        )
    ]

    constraints = build_confirmed_constraints(records)

    assert len(constraints) == 1
    assert constraints[0]["constraint_type"] == "drug_allergy"
    assert "阿莫西林" in constraints[0]["unsafe_terms"]
