# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import app.agent
import app.safety


def test_legacy_triage_stack_files_and_exports_are_removed():
    root = Path(__file__).resolve().parents[1]

    assert not (root / "app" / "triage_service.py").exists()
    assert not (root / "app" / "safety" / "record_guard.py").exists()
    assert not (root / "tests" / "test_triage_safety_fuse.py").exists()
    assert not hasattr(app.safety, "apply_confirmed_safety_fuse_to_triage_answer")


def test_agent_package_docstring_no_longer_mentions_legacy_routes_or_sqlite():
    doc = str(app.agent.__doc__ or "")

    assert "/v1/triage" not in doc
    assert "/v1/chat" not in doc
    assert "SQLite" not in doc
