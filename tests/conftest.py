import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# Ensure repo root is on sys.path so `import app.*` works when running pytest
# from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ===== CI 环境自动配置 =====

def pytest_configure(config):
    """pytest 启动时自动配置环境变量（CI 兼容）"""
    # 如果没有设置 API 密钥，使用占位符避免导入错误
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "test-key-for-ci"

    # 默认使用规则模式（避免 API 调用）
    if not os.environ.get("AGENT_SLOT_EXTRACTOR"):
        os.environ["AGENT_SLOT_EXTRACTOR"] = "rules"

    if not os.environ.get("CHAT_SLOT_EXTRACTOR"):
        os.environ["CHAT_SLOT_EXTRACTOR"] = "rules"


@pytest.fixture(scope="session")
def offline_mode(monkeypatch_session):
    """强制离线模式的 fixture"""
    monkeypatch_session.setenv("AGENT_SLOT_EXTRACTOR", "rules")
    monkeypatch_session.setenv("CHAT_SLOT_EXTRACTOR", "rules")


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session 级别的 monkeypatch"""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


class _FakeRedisClient:
    def __init__(self):
        self._data = {}

    def ping(self):
        return True

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value
        return True

    def delete(self, key):
        self._data.pop(key, None)
        return 1


@pytest.fixture(autouse=True)
def fake_agent_session_redis(monkeypatch):
    monkeypatch.setenv("AGENT_REDIS_URL", "redis://pytest-session-store/0")

    try:
        from app.agent.storage_redis import RedisSessionStore
    except Exception:
        yield
        return

    monkeypatch.setattr(
        RedisSessionStore,
        "_create_client",
        staticmethod(lambda _redis_url: _FakeRedisClient()),
    )

    try:
        from app.rag.cache_redis import RedisSemanticCache

        monkeypatch.setattr(
            RedisSemanticCache,
            "_create_client",
            staticmethod(lambda _redis_url: _FakeRedisClient()),
        )
    except Exception:
        pass

    try:
        import app.agent.graph as graph

        monkeypatch.setattr(graph, "_STORE", None, raising=False)
        monkeypatch.setattr(graph, "_GRAPH", None, raising=False)
    except Exception:
        pass

    try:
        import app.rag.rag_core as rag_core

        rag_core.clear_runtime_state()
    except Exception:
        pass

    yield


class FakeRagStore:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None, count: Optional[int] = None):
        self._rows = list(rows or [])
        self._count = int(count) if count is not None else len(self._rows)
        self.backend_name = "faiss-hnsw"
        self.collection_name = "pytest_kb"
        self.persist_dir = "pytest"

    def count(self) -> int:
        return self._count

    def similarity_search_with_score(self, query: str, *, k: int, filter: Optional[Dict[str, Any]] = None):
        try:
            from langchain.schema import Document  # type: ignore
        except Exception:
            from langchain_core.documents import Document  # type: ignore

        out = []
        for row in self._rows:
            metadata = dict(row.get("metadata") or {})
            if filter and any(metadata.get(key) != value for key, value in filter.items()):
                continue
            out.append((Document(page_content=str(row.get("page_content") or ""), metadata=metadata), float(row.get("score") or 0.0)))
            if len(out) >= max(1, int(k)):
                break
        return out

    def get_documents(self, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for row in self._rows:
            metadata = dict(row.get("metadata") or {})
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            documents.append(str(row.get("page_content") or ""))
            metadatas.append(metadata)
        return {"documents": documents, "metadatas": metadatas}

    def updated_at(self) -> float:
        return 0.0

