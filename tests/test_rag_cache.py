# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib

import pytest

from tests.conftest import FakeRagStore


class _FakeDoc:
    def __init__(self, text: str, source_file: str, department: str = "内科"):
        self.page_content = text
        self.metadata = {
            "department": department,
            "title": text,
            "row": 1,
            "source_file": source_file,
        }


def _fake_store():
    return FakeRagStore(count=3)


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


def test_retrieve_cache_hits_on_same_request(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setenv("RAG_CACHE_TTL_SECONDS", "300")
    monkeypatch.setenv("RAG_CACHE_MAX_ENTRIES", "8")
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "0")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)

    calls = {"n": 0}

    def fake_vector_search(query, top_n, department=None):
        calls["n"] += 1
        return [(_FakeDoc("喉咙痛常见于上呼吸道感染。", "a.md"), 0.08)]

    monkeypatch.setattr(rag_core, "_vector_search", fake_vector_search)

    first = rag_core.retrieve("喉咙痛怎么办", top_k=1, use_rerank=False)
    first_meta = rag_core.get_last_retrieval_meta()
    second = rag_core.retrieve("喉咙痛怎么办", top_k=1, use_rerank=False)
    second_meta = rag_core.get_last_retrieval_meta()

    assert first == second
    assert calls["n"] == 1
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True
    assert second_meta["cache_mode"] == "exact"


def test_retrieve_cache_bypasses_when_disabled(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_CACHE_ENABLED", "0")
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "0")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)

    calls = {"n": 0}

    def fake_vector_search(query, top_n, department=None):
        calls["n"] += 1
        return [(_FakeDoc("发烧时要监测体温。", "a.md"), 0.06)]

    monkeypatch.setattr(rag_core, "_vector_search", fake_vector_search)

    rag_core.retrieve("发烧怎么办", top_k=1, use_rerank=False)
    rag_core.retrieve("发烧怎么办", top_k=1, use_rerank=False)

    assert calls["n"] == 2


def test_retrieve_cache_isolated_by_department(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "0")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)

    calls = {"n": 0}

    def fake_vector_search(query, top_n, department=None):
        calls["n"] += 1
        dept = department or "内科"
        return [(_FakeDoc(f"{dept}证据", f"{dept}.md", department=dept), 0.05)]

    monkeypatch.setattr(rag_core, "_vector_search", fake_vector_search)

    internal = rag_core.retrieve("发烧怎么办", top_k=1, department="内科", use_rerank=False)
    derm = rag_core.retrieve("发烧怎么办", top_k=1, department="皮肤科", use_rerank=False)

    assert calls["n"] == 2
    assert internal[0]["metadata"]["department"] == "内科"
    assert derm[0]["metadata"]["department"] == "皮肤科"


def test_retrieve_cache_supports_semantic_overlap_hit(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setenv("RAG_CACHE_SIM_THRESHOLD", "0.6")
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "0")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)

    calls = {"n": 0}

    def fake_vector_search(query, top_n, department=None):
        calls["n"] += 1
        return [(_FakeDoc("喉咙痛先补水休息。", "a.md"), 0.09)]

    monkeypatch.setattr(rag_core, "_vector_search", fake_vector_search)

    rag_core.retrieve("喉咙痛怎么办", top_k=1, use_rerank=False)
    rag_core.retrieve("喉咙疼怎么办", top_k=1, use_rerank=False)
    meta = rag_core.get_last_retrieval_meta()

    assert calls["n"] == 1
    assert meta["cache_hit"] is True
    assert meta["cache_mode"] == "semantic"


def test_retrieve_cache_survives_runtime_reload_with_shared_redis(monkeypatch) -> None:
    import app.rag.rag_core as rag_core
    from app.rag.cache_redis import RedisSemanticCache

    shared_client = _FakeRedisClient()
    monkeypatch.setenv("AGENT_REDIS_URL", "redis://fake-rag-cache/0")
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "0")
    monkeypatch.setattr(RedisSemanticCache, "_create_client", staticmethod(lambda _redis_url: shared_client))

    rag_core.clear_runtime_state()
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)

    calls = {"n": 0}

    def fake_vector_search(query, top_n, department=None):
        calls["n"] += 1
        return [(_FakeDoc("喉咙痛先补水休息。", "a.md"), 0.09)]

    monkeypatch.setattr(rag_core, "_vector_search", fake_vector_search)

    rag_core.retrieve("喉咙痛怎么办", top_k=1, use_rerank=False)

    rag_core = importlib.reload(rag_core)
    monkeypatch.setattr(RedisSemanticCache, "_create_client", staticmethod(lambda _redis_url: shared_client))
    rag_core.clear_runtime_state()
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)
    monkeypatch.setattr(rag_core, "_vector_search", fake_vector_search)

    second = rag_core.retrieve("喉咙痛怎么办", top_k=1, use_rerank=False)
    meta = rag_core.get_last_retrieval_meta()

    assert calls["n"] == 1
    assert second[0]["text"] == "喉咙痛先补水休息。"
    assert meta["cache_hit"] is True
    assert meta["cache_mode"] == "exact"
    assert meta["cache_backend"] == "redis"


def test_retrieve_cache_gracefully_bypasses_without_redis_url(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    rag_core.clear_runtime_state()
    monkeypatch.delenv("RAG_REDIS_URL", raising=False)
    monkeypatch.delenv("AGENT_REDIS_URL", raising=False)
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "0")
    monkeypatch.setattr(rag_core, "get_vectordb", _fake_store)
    monkeypatch.setattr(rag_core, "_env_use_reranker", lambda: False)
    monkeypatch.setattr(
        rag_core,
        "_vector_search",
        lambda query, top_n, department=None: [(_FakeDoc("发烧时要监测体温。", "a.md"), 0.06)],
    )

    out = rag_core.retrieve("发烧怎么办", top_k=1, use_rerank=False)
    meta = rag_core.get_last_retrieval_meta()

    assert len(out) == 1
    assert out[0]["text"] == "发烧时要监测体温。"
    assert meta["cache_hit"] is False
    assert meta["cache_mode"] is None
