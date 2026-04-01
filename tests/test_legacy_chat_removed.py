# -*- coding: utf-8 -*-

from __future__ import annotations

from fastapi.testclient import TestClient

from app import api_server


def test_legacy_chat_route_and_helpers_are_removed():
    client = TestClient(api_server.app)
    response = client.post(
        "/v1/chat",
        json={
            "session_id": "legacy-chat",
            "patient_message": "我头痛两天",
            "top_k": 1,
            "mode": "safe",
        },
    )

    assert response.status_code == 404
    assert not hasattr(api_server, "ChatRequest")
    assert not hasattr(api_server, "_session_file_path")
    assert not hasattr(api_server, "_load_or_create_session")
    assert not hasattr(api_server, "_save_session")
    assert not hasattr(api_server, "_get_chat_graph")
