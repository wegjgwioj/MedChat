# -*- coding: utf-8 -*-
"""Redis 会话存储。可选依赖；未安装或连不上时由工厂决定是否回退。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.agent.state import AgentSessionState, utc_now_iso


class RedisSessionStore:
    def __init__(self, redis_url: str, key_prefix: str = "medchat:session:", client: Any = None):
        self._redis_url = str(redis_url or "").strip()
        self._key_prefix = str(key_prefix or "medchat:session:").strip() or "medchat:session:"
        self._client = client or self._create_client(self._redis_url)
        try:
            self._client.ping()
        except Exception as e:
            raise RuntimeError(f"Redis 不可用：{type(e).__name__}: {e}") from e

    @staticmethod
    def _create_client(redis_url: str) -> Any:
        try:
            import redis  # type: ignore
        except Exception as e:
            raise RuntimeError("未安装 redis 客户端，请先安装 redis 包") from e
        return redis.from_url(redis_url, decode_responses=False)

    def _key(self, session_id: str) -> str:
        sid = (session_id or "").strip()
        if not sid:
            raise RuntimeError("session_id 不能为空")
        return f"{self._key_prefix}{sid}"

    def storage_meta(self) -> Dict[str, Any]:
        return {"type": "redis", "redis_url": self._redis_url, "key_prefix": self._key_prefix}

    def load_session(self, session_id: str) -> Optional[AgentSessionState]:
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return AgentSessionState.model_validate_json(raw)
        return AgentSessionState.model_validate(raw)

    def save_session(self, state: AgentSessionState) -> None:
        if not isinstance(state, AgentSessionState):
            raise RuntimeError("save_session 入参不是 AgentSessionState")
        state.trim_messages(max_turns=20)
        state.last_update_ts = utc_now_iso()
        payload = json.dumps(state.model_dump(), ensure_ascii=False)
        self._client.set(self._key(state.session_id), payload)

    def delete_session(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))
