# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import storage


def test_build_session_store_requires_redis_url_and_never_falls_back(monkeypatch):
    monkeypatch.delenv("AGENT_REDIS_URL", raising=False)
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    monkeypatch.delenv("AGENT_SQLITE_DB_PATH", raising=False)

    with pytest.raises(RuntimeError, match="AGENT_REDIS_URL"):
        storage.build_session_store()


def test_non_redis_session_store_modules_are_removed():
    root = Path(__file__).resolve().parents[1]

    assert not (root / "app" / "agent" / "storage_sqlite.py").exists()
    assert not (root / "app" / "agent" / "storage_memory.py").exists()
