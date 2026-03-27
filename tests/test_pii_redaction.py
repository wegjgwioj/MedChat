# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def test_redact_pii_for_llm_masks_phone_id_and_address() -> None:
    from app.privacy.pii import redact_pii_for_llm

    text = "我手机号13800138000，身份证110101199001011234，住址北京市朝阳区酒仙桥路10号。"
    redacted = redact_pii_for_llm(text)

    assert "13800138000" not in redacted
    assert "110101199001011234" not in redacted
    assert "北京市朝阳区酒仙桥路10号" not in redacted
    assert "<PHONE>" in redacted
    assert "<IDCARD>" in redacted
    assert "<ADDRESS>" in redacted


def test_triage_assess_redacts_user_text_before_llm(monkeypatch) -> None:
    import app.triage_service as triage_service

    captured = {}

    class FakeChain:
        def run(self, payload):
            captured["user_text"] = payload["user_text"]
            return json.dumps(
                {
                    "triage_level": "ROUTINE",
                    "red_flags": [],
                    "immediate_actions": ["先观察并补充信息[E1]"],
                    "what_not_to_do": [],
                    "key_questions": [],
                    "reasoning": "根据[E1]资料先做初步观察。",
                    "uncertainty": "",
                    "safety_notice": "本回答仅供信息参考。",
                    "citations_used": ["E1"],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(triage_service._ENGINE, "_initialized", True, raising=False)
    monkeypatch.setattr(triage_service._ENGINE, "chain_suggest", FakeChain(), raising=False)

    triage_service.triage_step_assess(
        user_text="我手机号13800138000，身份证110101199001011234，地址上海市浦东新区世纪大道100号。",
        evidence_list=[
            {
                "eid": "E1",
                "text": "测试证据",
                "source": "kb.md",
                "chunk_id": "kb:1",
                "score": 0.1,
                "rerank_score": 0.9,
                "metadata": {"department": "内科", "title": "测试", "row": 1, "source_file": "kb.md"},
            }
        ],
        trace=[],
    )

    assert "13800138000" not in captured["user_text"]
    assert "110101199001011234" not in captured["user_text"]
    assert "世纪大道100号" not in captured["user_text"]
    assert "<PHONE>" in captured["user_text"]
    assert "<IDCARD>" in captured["user_text"]
    assert "<ADDRESS>" in captured["user_text"]


def test_agent_slot_extraction_redacts_user_text_before_llm(monkeypatch) -> None:
    import app.agent.graph as graph

    captured = {}

    def fake_call_llm_text(system: str, user: str) -> str:
        captured["user"] = user
        return json.dumps(
            {
                "age": None,
                "sex": "",
                "symptoms": [],
                "duration": "",
                "severity": "",
                "fever": "unknown",
                "location": "",
                "pregnancy": "unknown",
                "meds": "",
                "allergy": "",
                "history": "",
                "red_flags": [],
                "department_guess": None,
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(graph, "_call_llm_text", fake_call_llm_text)

    graph._extract_slots_with_llm("我电话13800138000，身份证110101199001011234，住在杭州市西湖区文三路90号")

    assert "13800138000" not in captured["user"]
    assert "110101199001011234" not in captured["user"]
    assert "文三路90号" not in captured["user"]
    assert "<PHONE>" in captured["user"]
    assert "<IDCARD>" in captured["user"]
    assert "<ADDRESS>" in captured["user"]


def test_agent_answer_compose_redacts_user_text_before_llm(monkeypatch, tmp_path) -> None:
    import app.agent.graph as graph
    import app.rag.retriever as retriever
    import app.rag.rag_core as rag_core
    from app.agent.storage_sqlite import SqliteSessionStore

    graph._GRAPH = None
    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)
    monkeypatch.setattr(graph, "_STORE", SqliteSessionStore(Path(tmp_path) / "agent.sqlite3"))

    captured = {}

    def fake_call_llm_text(system: str, user: str) -> str:
        captured["user"] = user
        return "根据[E1][E2]资料，建议先观察。\n\n引用：[E1][E2]\n免责声明：本回答仅供信息参考，不能替代医生面诊。"

    def fake_retrieve(*args, **kwargs):
        return [
            {
                "eid": "E1",
                "text": "证据一",
                "source": "a.md",
                "chunk_id": "a:1",
                "score": 0.1,
                "rerank_score": 0.95,
                "metadata": {"department": "内科", "title": "A", "row": 1, "source_file": "a.md"},
            },
            {
                "eid": "E2",
                "text": "证据二",
                "source": "b.md",
                "chunk_id": "b:1",
                "score": 0.12,
                "rerank_score": 0.93,
                "metadata": {"department": "内科", "title": "B", "row": 2, "source_file": "b.md"},
            },
        ]

    monkeypatch.setattr(graph, "_call_llm_text", fake_call_llm_text)
    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        rag_core,
        "get_stats",
        lambda: types.SimpleNamespace(
            device="cpu",
            collection="pytest",
            count=2,
            persist_dir="pytest",
            embed_model="pytest",
            rerank_model=None,
            updated_at="",
        ),
    )

    out = graph.run_chat_v2_turn(
        session_id="pii-answer-session",
        user_message="我电话13800138000，住址深圳市南山区科技园科苑路15号，风疹病毒是怎么感染的？",
        top_k=5,
        top_n=30,
        use_rerank=True,
    )

    assert out["mode"] == "answer"
    assert "13800138000" not in captured["user"]
    assert "科苑路15号" not in captured["user"]
    assert "<PHONE>" in captured["user"]
    assert "<ADDRESS>" in captured["user"]


def test_query_slimming_redacts_user_text_before_openai(monkeypatch) -> None:
    import app.rag.rag_core as rag_core

    class FakeResponse:
        choices = [types.SimpleNamespace(message={"content": "风疹 感染 传播"})]

    class FakeChatCompletion:
        @staticmethod
        def create(**kwargs):
            user_content = kwargs["messages"][1]["content"]
            assert "13800138000" not in user_content
            assert "110101199001011234" not in user_content
            assert "酒仙桥路10号" not in user_content
            assert "<PHONE>" in user_content
            assert "<IDCARD>" in user_content
            assert "<ADDRESS>" in user_content
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

    result = rag_core._rewrite_query_slimming("我电话13800138000，身份证110101199001011234，住在北京市朝阳区酒仙桥路10号，风疹病毒怎么感染？")
    assert result == "风疹 感染 传播"
