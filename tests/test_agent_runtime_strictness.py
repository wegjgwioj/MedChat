# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


def test_memory_update_fails_fast_when_slot_extraction_errors(monkeypatch):
    from app.agent import graph
    from app.agent.state import AgentSessionState

    monkeypatch.setattr(
        graph,
        "_extract_slots_with_llm",
        lambda _msg: (_ for _ in ()).throw(RuntimeError("slot extractor down")),
    )

    state = {
        "session": AgentSessionState(session_id="strict-memory"),
        "user_message": "我头痛两天了",
        "trace": {},
    }

    with pytest.raises(RuntimeError, match="slot extractor down"):
        graph._node_memory_update(state)


def test_rag_retrieve_fails_fast_when_backend_errors(monkeypatch):
    from app.agent import graph
    from app.agent.state import AgentSessionState
    import app.rag.retriever as retriever

    monkeypatch.setattr(retriever, "retrieve", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rag down")))

    state = {
        "session": AgentSessionState(session_id="strict-rag", summary="症状：咳嗽"),
        "mode": "answer",
        "user_message": "我咳嗽三天了",
        "top_k": 5,
        "top_n": 30,
        "use_rerank": False,
        "trace": {"rag_stats": {}},
    }

    with pytest.raises(RuntimeError, match="rag down"):
        graph._node_rag_retrieve(state)


def test_answer_compose_fails_fast_when_llm_errors(monkeypatch):
    from app.agent import graph
    from app.agent.state import AgentSessionState

    monkeypatch.setattr(
        graph,
        "_call_llm_text",
        lambda _system, _user: (_ for _ in ()).throw(RuntimeError("llm down")),
    )

    state = {
        "session": AgentSessionState(
            session_id="strict-answer",
            summary="症状：咽痛；时长：2天；年龄：24岁；性别：女；严重程度：3/10",
            record_summary="过敏：青霉素过敏",
        ),
        "mode": "answer",
        "user_message": "我喉咙痛怎么办",
        "evidence": [
            {
                "eid": "E1",
                "text": "充足证据 1",
                "source": "kb1.md",
                "chunk_id": "kb1:1",
                "score": 0.01,
                "rerank_score": 0.95,
                "metadata": {},
            },
            {
                "eid": "E2",
                "text": "充足证据 2",
                "source": "kb2.md",
                "chunk_id": "kb2:1",
                "score": 0.02,
                "rerank_score": 0.93,
                "metadata": {},
            },
        ],
        "trace": {"rag_stats": {"evidence_quality": {"level": "ok", "reason": "enough", "count": 2}}},
        "citations": [],
    }

    with pytest.raises(RuntimeError, match="llm down"):
        graph._node_answer_compose(state)
