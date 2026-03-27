# -*- coding: utf-8 -*-
"""Agent 会话存储抽象与工厂。"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Protocol

from app.agent.state import AgentSessionState
from app.agent.storage_redis import RedisSessionStore


class SessionStore(Protocol):
    def load_session(self, session_id: str) -> Optional[AgentSessionState]:
        ...

    def save_session(self, state: AgentSessionState) -> None:
        ...

    def delete_session(self, session_id: str) -> None:
        ...

    def storage_meta(self) -> Dict[str, Any]:
        ...


def _build_redis_session_store() -> RedisSessionStore:
    redis_url = (os.getenv("AGENT_REDIS_URL") or "").strip()
    if not redis_url:
        raise RuntimeError("AGENT_REDIS_URL 未配置，无法初始化 Redis 会话存储。")
    key_prefix = (os.getenv("AGENT_REDIS_PREFIX") or "medchat:session:").strip() or "medchat:session:"
    return RedisSessionStore(redis_url=redis_url, key_prefix=key_prefix)


def build_session_store() -> SessionStore:
    return _build_redis_session_store()
