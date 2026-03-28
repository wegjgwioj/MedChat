# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest
import importlib


class _FakeRedisClient:
    def __init__(self) -> None:
        self._data = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: str) -> bool:
        self._data[key] = value
        return True

    def delete(self, key: str) -> int:
        return 1 if self._data.pop(key, None) is not None else 0


def test_redis_session_store_roundtrip_with_fake_client():
    from app.agent.state import AgentSessionState
    from app.agent.storage_redis import RedisSessionStore

    fake = _FakeRedisClient()
    store = RedisSessionStore(redis_url="redis://fake:6379/0", key_prefix="medchat:test:", client=fake)
    sess = AgentSessionState(session_id="redis-1")
    sess.append_message("user", "我头痛两天了")

    store.save_session(sess)
    loaded = store.load_session("redis-1")

    assert loaded is not None
    assert loaded.session_id == "redis-1"
    assert loaded.messages[-1].content == "我头痛两天了"
    assert store.storage_meta()["type"] == "redis"
    store.delete_session("redis-1")
    assert store.load_session("redis-1") is None


def test_build_session_store_is_redis_only_and_fails_fast(monkeypatch):
    from app.agent import storage

    monkeypatch.delenv("AGENT_REDIS_URL", raising=False)
    monkeypatch.setenv("AGENT_SESSION_STORE", "sqlite")

    store = storage.build_session_store()

    assert store.storage_meta()["type"] == "sqlite"


def test_build_session_store_falls_back_to_sqlite_when_redis_unavailable(monkeypatch):
    from app.agent import storage

    monkeypatch.setenv("AGENT_REDIS_URL", "redis://fake:6379/0")
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    monkeypatch.setattr(storage, "_build_redis_session_store", lambda: (_ for _ in ()).throw(RuntimeError("redis down")))

    store = storage.build_session_store()

    assert store.storage_meta()["type"] == "sqlite"


def test_graph_trace_uses_redis_store(monkeypatch):
    import app.agent.storage as storage
    from app.agent.storage_redis import RedisSessionStore

    monkeypatch.setattr(
        storage,
        "build_session_store",
        lambda: RedisSessionStore(redis_url="redis://fake:6379/0", key_prefix="medchat:test:", client=_FakeRedisClient()),
    )

    import app.agent.graph as graph
    graph = importlib.reload(graph)
    graph._GRAPH = None
    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)
    monkeypatch.setattr(
        graph,
        "_STORE",
        RedisSessionStore(redis_url="redis://fake:6379/0", key_prefix="medchat:test:", client=_FakeRedisClient()),
        raising=False,
    )

    out = graph.run_chat_v2_turn(
        session_id="redis-trace-1",
        user_message="我头疼两天了",
        top_k=3,
        top_n=10,
        use_rerank=False,
    )

    storage_meta = (out.get("trace") or {}).get("storage") or {}
    assert storage_meta["type"] == "redis"
