# -*- coding: utf-8 -*-
"""Agent 会话存储抽象与工厂。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from app.agent.state import AgentSessionState
from app.agent.storage_memory import InMemorySessionStore
from app.agent.storage_redis import RedisSessionStore
from app.agent.storage_sqlite import SqliteSessionStore


class SessionStore(Protocol):
    def load_session(self, session_id: str) -> Optional[AgentSessionState]:
        ...

    def save_session(self, state: AgentSessionState) -> None:
        ...

    def delete_session(self, session_id: str) -> None:
        ...

    def storage_meta(self) -> Dict[str, Any]:
        ...


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_sqlite_session_store() -> SqliteSessionStore:
    raw = (os.getenv("AGENT_SESSION_DB_PATH") or "").strip()
    db_path = Path(raw) if raw else None
    return SqliteSessionStore(db_path=db_path)


def _build_redis_session_store() -> RedisSessionStore:
    redis_url = (os.getenv("AGENT_REDIS_URL") or "redis://127.0.0.1:6379/0").strip()
    key_prefix = (os.getenv("AGENT_REDIS_PREFIX") or "medchat:session:").strip() or "medchat:session:"
    return RedisSessionStore(redis_url=redis_url, key_prefix=key_prefix)


def build_session_store() -> SessionStore:
    backend = (os.getenv("AGENT_SESSION_STORE") or "sqlite").strip().lower()
    strict = _env_flag("AGENT_SESSION_STORE_STRICT", "0")

    if backend == "memory":
        return InMemorySessionStore()

    if backend == "redis":
        try:
            return _build_redis_session_store()
        except Exception:
            if strict:
                raise
            return _build_sqlite_session_store()

    return _build_sqlite_session_store()
