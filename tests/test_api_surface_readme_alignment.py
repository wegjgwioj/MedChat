# -*- coding: utf-8 -*-

from __future__ import annotations

from app import api_server


def test_public_api_surface_matches_readme_whitelist():
    paths = {
        route.path
        for route in api_server.app.routes
        if route.path == "/health" or route.path.startswith("/v1/")
    }

    assert paths == {
        "/health",
        "/v1/agent/chat_v2",
        "/v1/agent/chat_v2/stream",
    }
