# -*- coding: utf-8 -*-

from __future__ import annotations

from fastapi.testclient import TestClient

from app import api_server


def test_readme_external_public_routes_return_404_and_are_not_in_openapi():
    client = TestClient(api_server.app)

    for method, path in (
        ("post", "/v1/triage"),
        ("get", "/v1/rag/stats"),
        ("post", "/v1/rag/retrieve"),
        ("post", "/v1/ocr/ingest"),
        ("get", "/v1/ocr/status/test-task"),
    ):
        response = getattr(client, method)(path)
        assert response.status_code == 404

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert set(openapi.json()["paths"].keys()) == {
        "/health",
        "/v1/agent/chat_v2",
        "/v1/agent/chat_v2/stream",
    }


def test_readme_external_public_symbols_are_removed_from_api_server_module():
    for name in (
        "TriageRequest",
        "RagRetrieveRequest",
        "triage",
        "rag_stats",
        "rag_retrieve",
        "ocr_ingest",
        "ocr_status",
        "_prune_routes_to_readme_whitelist",
        "_README_PUBLIC_PATHS",
    ):
        assert not hasattr(api_server, name)
