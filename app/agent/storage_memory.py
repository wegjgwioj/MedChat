# -*- coding: utf-8 -*-
"""进程内会话存储。适合测试与单进程本地运行。"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from app.agent.state import AgentSessionState, utc_now_iso


class InMemorySessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def storage_meta(self) -> Dict[str, Any]:
        return {"type": "memory"}

    def load_session(self, session_id: str) -> Optional[AgentSessionState]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        with self._lock:
            payload = self._sessions.get(sid)
        if payload is None:
            return None
        return AgentSessionState.model_validate(payload)

    def save_session(self, state: AgentSessionState) -> None:
        if not isinstance(state, AgentSessionState):
            raise RuntimeError("save_session 入参不是 AgentSessionState")
        state.trim_messages(max_turns=20)
        state.last_update_ts = utc_now_iso()
        with self._lock:
            self._sessions[state.session_id] = state.model_dump()

    def delete_session(self, session_id: str) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._sessions.pop(sid, None)
