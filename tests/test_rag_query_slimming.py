# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import types


def test_query_slimming_uses_legacy_openai_chatcompletion(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    class FakeResponse:
        choices = [types.SimpleNamespace(message={"content": "咳嗽 发热"})]

    class FakeChatCompletion:
        @staticmethod
        def create(**kwargs):
            assert kwargs["model"] == "deepseek-chat"
            return FakeResponse()

    fake_openai = types.SimpleNamespace(
        api_key=None,
        api_base=None,
        ChatCompletion=FakeChatCompletion,
    )

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")

    assert rag_core._rewrite_query_slimming("我最近咳嗽还有一点发热") == "咳嗽 发热"
