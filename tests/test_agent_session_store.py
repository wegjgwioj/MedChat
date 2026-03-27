# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path


def test_memory_session_store_roundtrip():
    from app.agent.state import AgentSessionState
    from app.agent.storage_memory import InMemorySessionStore

    store = InMemorySessionStore()
    sess = AgentSessionState(session_id="mem-1")
    sess.append_message("user", "我头痛两天了")

    store.save_session(sess)
    loaded = store.load_session("mem-1")

    assert loaded is not None
    assert loaded.session_id == "mem-1"
    assert loaded.messages[-1].content == "我头痛两天了"
    assert store.storage_meta()["type"] == "memory"


def test_build_session_store_falls_back_to_sqlite_when_redis_is_unavailable(monkeypatch, tmp_path):
    from app.agent import storage
    from app.agent.storage_sqlite import SqliteSessionStore

    monkeypatch.setenv("AGENT_SESSION_STORE", "redis")
    monkeypatch.setenv("AGENT_SESSION_DB_PATH", str(tmp_path / "agent.sqlite3"))
    monkeypatch.delenv("AGENT_SESSION_STORE_STRICT", raising=False)
    monkeypatch.setattr(storage, "_build_redis_session_store", lambda: (_ for _ in ()).throw(RuntimeError("redis down")))

    store = storage.build_session_store()

    assert isinstance(store, SqliteSessionStore)
    assert Path(store.db_path).name == "agent.sqlite3"


def test_graph_trace_uses_memory_store(monkeypatch):
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    import app.agent.graph as graph
    from app.agent.storage_memory import InMemorySessionStore

    graph._GRAPH = None
    monkeypatch.setattr(graph, "_STORE", InMemorySessionStore(), raising=False)

    out = graph.run_chat_v2_turn(
        session_id="mem-trace-1",
        user_message="我头疼两天了",
        top_k=3,
        top_n=10,
        use_rerank=False,
    )

    storage_meta = (out.get("trace") or {}).get("storage") or {}
    assert storage_meta["type"] == "memory"
