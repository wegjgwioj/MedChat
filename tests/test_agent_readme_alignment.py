# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from types import SimpleNamespace

from app.agent import graph
from app.agent.state import Slots
from app.agent.storage_redis import RedisSessionStore


class _FakeRedisClient:
    def __init__(self) -> None:
        self._data = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value):
        self._data[key] = value
        return True

    def delete(self, key: str):
        self._data.pop(key, None)
        return 1


def _fake_store() -> RedisSessionStore:
    return RedisSessionStore(redis_url="redis://unit-test", client=_FakeRedisClient())


def _capture_retrieval_query(bucket: dict, query: str, **_kwargs):
    bucket["query"] = query
    return [
        {
            "eid": "E1",
            "text": "咽痛可见于上呼吸道感染。",
            "source": "test-kb",
            "score": 0.92,
            "metadata": {"title": "内科知识库", "department": "内科"},
        }
    ]


def test_phase0_guardrail_blocks_ood_and_masks_pii_before_persist(monkeypatch):
    store = _fake_store()

    monkeypatch.setenv("AGENT_REDIS_URL", "redis://unit-test")
    monkeypatch.setattr(graph, "build_session_store", lambda: store)
    monkeypatch.setattr(graph, "_extract_slots_with_llm", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("phase0 blocked requests should not reach slot extraction")))
    graph._STORE = None
    graph._GRAPH = None

    out = graph.run_chat_v2_turn(
        session_id="phase0-ood",
        user_message="我叫张三，手机号 13800138000，帮我写一个 Python 爬虫。",
        top_k=1,
        top_n=1,
        use_rerank=False,
    )

    assert out["mode"] == "answer"
    assert "医疗健康相关" in out["answer"]
    assert out["trace"]["phase0"]["blocked"] is True
    assert out["trace"]["phase0"]["label"] == "out_of_domain"
    assert out["trace"]["phase0"]["pii_masked"] is True

    session = store.load_session("phase0-ood")
    assert session is not None
    persisted = "\n".join(message.content for message in session.messages)
    assert "13800138000" not in persisted
    assert "张三" not in persisted
    assert "<PHONE>" in persisted
    assert "<NAME>" in persisted
    assert "13800138000" not in json.dumps(out["trace"], ensure_ascii=False)
    assert "张三" not in json.dumps(out["trace"], ensure_ascii=False)


def test_phase1_slot_filling_asks_when_information_is_insufficient(monkeypatch):
    store = _fake_store()

    monkeypatch.setenv("AGENT_REDIS_URL", "redis://unit-test")
    monkeypatch.setattr(graph, "build_session_store", lambda: store)
    monkeypatch.setattr(
        graph,
        "_extract_slots_with_llm",
        lambda _message: Slots(symptoms=["头痛"]),
    )
    graph._STORE = None
    graph._GRAPH = None

    out = graph.run_chat_v2_turn(
        session_id="phase1-ask",
        user_message="我头痛",
        top_k=1,
        top_n=1,
        use_rerank=False,
    )

    assert out["mode"] == "ask"
    assert out["questions"]
    assert out["next_questions"]
    assert out["trace"]["phase1"]["decision"] == "ask"
    assert "duration" in out["trace"]["phase1"]["missing_slots"]
    assert "age" in out["trace"]["phase1"]["missing_slots"]
    assert "severity" in out["trace"]["phase1"]["missing_slots"]
    assert out["trace"]["phase1"]["planner_strategy"] == "triage"


def test_phase2_builds_structured_chief_complaint_and_retrieval_query(monkeypatch):
    store = _fake_store()
    captured = {}

    monkeypatch.setenv("AGENT_REDIS_URL", "redis://unit-test")
    monkeypatch.setattr(graph, "build_session_store", lambda: store)
    monkeypatch.setattr(
        graph,
        "_extract_slots_with_llm",
        lambda _message: Slots(
            age=28,
            sex="女",
            symptoms=["咽痛"],
            duration="2天",
            severity="中",
            fever="no",
        ),
    )
    monkeypatch.setattr(
        "app.rag.retriever.retrieve",
        lambda query, **kwargs: _capture_retrieval_query(captured, query, **kwargs),
    )
    monkeypatch.setattr("app.rag.retriever.get_last_retrieval_meta", lambda: {"backend": "opensearch"})
    monkeypatch.setattr(
        "app.rag.rag_core.get_stats",
        lambda: SimpleNamespace(
            backend="opensearch",
            device="cpu",
            collection="kb",
            count=10,
        ),
    )
    monkeypatch.setattr(graph, "is_low_evidence", lambda _quality: False)
    monkeypatch.setattr(graph, "_call_llm_text", lambda _system, _user: "建议先补液休息。")
    graph._STORE = None
    graph._GRAPH = None

    out = graph.run_chat_v2_turn(
        session_id="phase2-answer",
        user_message="我是28岁女性，咽痛2天，中等程度，没有发烧。",
        top_k=1,
        top_n=1,
        use_rerank=False,
    )

    assert out["mode"] == "answer"
    assert captured["query"] == "主诉：28岁女，咽痛2天，中。"
    assert out["trace"]["phase2"]["chief_complaint"] == "28岁女，咽痛2天，中。"
    assert out["trace"]["phase2"]["retrieval_query"] == "主诉：28岁女，咽痛2天，中。"
    assert out["trace"]["phase2"]["retrieval_started"] is True
