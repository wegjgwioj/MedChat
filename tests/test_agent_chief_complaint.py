# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace


def test_build_chief_complaint_from_slots_normalizes_core_case() -> None:
    from app.agent.state import Slots, build_chief_complaint_from_slots

    complaint = build_chief_complaint_from_slots(
        Slots(
            age=24,
            sex="女",
            symptoms=["咳嗽", "发热"],
            duration="3天",
            severity="中度",
            location="咽喉",
        )
    )

    assert complaint == "24岁女，咽喉咳嗽、发热3天，中度。"


def test_node_rag_retrieve_uses_structured_chief_complaint_for_triage(monkeypatch) -> None:
    from app.agent import graph
    from app.agent.state import AgentSessionState, Slots
    import app.rag.retriever as retriever
    import app.rag.rag_core as rag_core

    captured = {}

    def fake_retrieve(rag_query, top_k=5, top_n=30, department=None, use_rerank=None):
        captured["rag_query"] = rag_query
        return []

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(retriever, "get_last_retrieval_meta", lambda: {})
    monkeypatch.setattr(
        rag_core,
        "get_stats",
        lambda: SimpleNamespace(
            backend="faiss-hnsw",
            device="cpu",
            collection="pytest",
            count=0,
            persist_dir="pytest",
            embed_model="pytest",
            rerank_model=None,
            updated_at="",
        ),
    )

    state = {
        "session": AgentSessionState(
            session_id="chief-complaint",
            slots=Slots(
                age=24,
                sex="女",
                symptoms=["咳嗽", "发热"],
                duration="3天",
                severity="中度",
                location="咽喉",
            ),
            summary="年龄24岁；性别女；症状：咳嗽、发热；时长：3天；严重程度：中度",
            record_summary="过敏：青霉素过敏",
        ),
        "mode": "answer",
        "user_message": "我咳嗽三天了还有点发烧",
        "top_k": 5,
        "top_n": 30,
        "use_rerank": False,
        "trace": {"planner_strategy": "triage"},
    }

    graph._node_rag_retrieve(state)

    assert captured["rag_query"] == "过敏：青霉素过敏；主诉：24岁女，咽喉咳嗽、发热3天，中度。"
    assert state["trace"]["chief_complaint"] == "24岁女，咽喉咳嗽、发热3天，中度。"


def test_node_rag_retrieve_keeps_raw_question_for_kb_qa(monkeypatch) -> None:
    from app.agent import graph
    from app.agent.state import AgentSessionState
    import app.rag.retriever as retriever
    import app.rag.rag_core as rag_core

    captured = {}

    def fake_retrieve(rag_query, **kwargs):
        captured["rag_query"] = rag_query
        return []

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(retriever, "get_last_retrieval_meta", lambda: {})
    monkeypatch.setattr(
        rag_core,
        "get_stats",
        lambda: SimpleNamespace(
            backend="faiss-hnsw",
            device="cpu",
            collection="pytest",
            count=0,
            persist_dir="pytest",
            embed_model="pytest",
            rerank_model=None,
            updated_at="",
        ),
    )

    state = {
        "session": AgentSessionState(session_id="kb-qa"),
        "mode": "answer",
        "user_message": "风疹病毒是怎么感染的？",
        "top_k": 5,
        "top_n": 30,
        "use_rerank": False,
        "trace": {"planner_strategy": "kb_qa"},
    }

    graph._node_rag_retrieve(state)

    assert captured["rag_query"] == "问题：风疹病毒是怎么感染的？"


def test_run_chat_v2_turn_records_chief_complaint_in_trace_and_rag_query(monkeypatch, tmp_path) -> None:
    from app.agent import graph
    import app.agent.record_index as record_index
    from app.agent.state import Slots
    from app.agent.storage_sqlite import SqliteSessionStore
    import app.rag.retriever as retriever
    import app.rag.rag_core as rag_core

    captured = {}

    graph._GRAPH = None
    graph._STORE = SqliteSessionStore(tmp_path / "agent_sessions.sqlite3")
    monkeypatch.setattr(
        record_index,
        "_compute_record_similarity",
        lambda left, right: 1.0 if left == right else 0.05,
    )
    monkeypatch.setattr(
        graph,
        "_extract_slots_with_llm",
        lambda _msg: Slots(
            age=24,
            sex="女",
            symptoms=["咳嗽", "发热"],
            fever="yes",
            duration="3天",
            severity="中度",
            location="咽喉",
        ),
    )

    def fake_retrieve(rag_query, top_k=5, top_n=30, department=None, use_rerank=None):
        captured["rag_query"] = rag_query
        return []

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(retriever, "get_last_retrieval_meta", lambda: {})
    monkeypatch.setattr(
        rag_core,
        "get_stats",
        lambda: SimpleNamespace(
            backend="faiss-hnsw",
            device="cpu",
            collection="pytest",
            count=0,
            persist_dir="pytest",
            embed_model="pytest",
            rerank_model=None,
            updated_at="",
        ),
    )

    resp = graph.run_chat_v2_turn(
        session_id="chief-complaint-e2e",
        user_message="我咳嗽三天了还有点发烧",
        top_k=5,
        top_n=30,
        use_rerank=False,
    )

    assert resp["mode"] == "answer"
    assert resp["trace"]["planner_strategy"] == "triage"
    assert resp["trace"]["chief_complaint"] == "24岁女，咽喉咳嗽、发热3天，中度。"
    assert captured["rag_query"] == "年龄24岁；性别女；主诉：24岁女，咽喉咳嗽、发热3天，中度。"
