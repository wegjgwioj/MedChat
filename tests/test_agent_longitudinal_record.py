# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


def test_upsert_longitudinal_records_applies_importance_threshold(monkeypatch):
    import app.agent.record_index as record_index
    from app.agent.state import Slots

    monkeypatch.setenv("AGENT_RECORD_MIN_IMPORTANCE", "0.9")

    records, stats = record_index.upsert_longitudinal_records(
        [],
        Slots(age=24, sex="男", meds="维生素", allergy="青霉素过敏"),
    )

    assert len(records) == 1
    assert records[0].category == "allergy"
    assert records[0].importance_score >= 0.9
    assert stats["added"] == 1
    assert stats["skipped"] >= 1


def test_upsert_longitudinal_records_merges_semantic_duplicates(monkeypatch):
    import app.agent.record_index as record_index
    from app.agent.state import Slots

    monkeypatch.setattr(
        record_index,
        "_compute_record_similarity",
        lambda left, right: 0.96 if "过敏" in left and "过敏" in right else 0.05,
    )

    records, _ = record_index.upsert_longitudinal_records([], Slots(allergy="青霉素过敏"))
    records, stats = record_index.upsert_longitudinal_records(records, Slots(allergy="阿莫西林过敏"))

    allergy_records = [record for record in records if record.category == "allergy"]

    assert len(allergy_records) == 1
    assert stats["merged"] == 1
    assert stats["added"] == 0


def test_memory_update_builds_longitudinal_records_and_summary(monkeypatch, tmp_path):
    import app.agent.record_index as record_index
    from app.agent import graph
    from app.agent.state import AgentSessionState
    from app.agent.storage_sqlite import SqliteSessionStore

    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)
    monkeypatch.setattr(graph, "_STORE", SqliteSessionStore(tmp_path / "agent_sessions.sqlite3"), raising=False)
    monkeypatch.setattr(
        record_index,
        "_compute_record_similarity",
        lambda left, right: 0.96 if "过敏" in left and "过敏" in right else 0.05,
    )

    first = graph._node_memory_update(
        {
            "session": AgentSessionState(session_id="record-session-1"),
            "user_message": "我24岁，男，青霉素过敏，有哮喘史。",
            "trace": {},
        }
    )["session"]

    second_state = {
        "session": first,
        "user_message": "我对阿莫西林也过敏，目前在吃维生素。",
        "trace": {},
    }
    second = graph._node_memory_update(second_state)["session"]

    allergy_records = [record for record in second.longitudinal_records if record.category == "allergy"]
    med_records = [record for record in second.longitudinal_records if record.category == "medication"]

    assert len(allergy_records) == 1
    assert len(med_records) == 1
    assert "过敏：" in second.record_summary
    assert "用药：维生素" in second.record_summary
    assert second_state["trace"]["record_admission"]["merged"] == 1


def test_full_chain_confirmed_ocr_allergy_blocks_later_medication(monkeypatch):
    import app.api_server as api_server
    from app.agent import graph
    import app.agent.record_index as record_index
    from app.agent.storage_sqlite import SqliteSessionStore
    import app.rag.rag_core as rag_core
    import app.rag.retriever as retriever
    from fastapi.testclient import TestClient

    db_dir = Path(".pytest-tmp") / "full-chain"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"agent_sessions_{uuid4().hex}.sqlite3"
    store = SqliteSessionStore(db_path)

    monkeypatch.setattr(api_server, "_OCR_STORE", store, raising=False)
    monkeypatch.setattr(graph, "_STORE", store, raising=False)
    monkeypatch.setenv("AGENT_SESSION_STORE", "sqlite")
    monkeypatch.setenv("AGENT_SQLITE_DB_PATH", str(db_path))
    monkeypatch.setattr(graph, "_extract_slots_with_llm", graph._rule_extract_slots)
    monkeypatch.setattr(
        record_index,
        "_compute_record_similarity",
        lambda left, right: 1.0 if left == right else 0.05,
    )
    monkeypatch.setattr(
        retriever,
        "retrieve",
        lambda rag_query, top_k=5, top_n=30, department=None, use_rerank=None: [
            {
                "eid": "E1",
                "text": "咽痛可对症处理。",
                "source": "kb1",
                "chunk_id": "kb1:1",
                "score": 0.01,
                "rerank_score": 0.93,
                "metadata": {},
            },
            {
                "eid": "E2",
                "text": "发热时可考虑对乙酰氨基酚。",
                "source": "kb2",
                "chunk_id": "kb2:1",
                "score": 0.02,
                "rerank_score": 0.91,
                "metadata": {},
            },
        ],
    )
    monkeypatch.setattr(retriever, "get_last_retrieval_meta", lambda: {})
    monkeypatch.setattr(
        rag_core,
        "get_stats",
        lambda: SimpleNamespace(
            backend="faiss-hnsw",
            device="cpu",
            collection="pytest",
            count=2,
            persist_dir="pytest",
            embed_model="pytest",
            rerank_model=None,
            updated_at="",
        ),
    )
    monkeypatch.setattr(
        graph,
        "_call_llm_text",
        lambda system, user: (
            "建议先口服阿莫西林；如发热可考虑对乙酰氨基酚。[E1]\n\n"
            "引用：[E1]\n免责声明：本回答仅供信息参考，不能替代医生面诊。"
        ),
    )
    graph._GRAPH = None

    store.upsert_ocr_task(
        task_id="task-chain-ocr",
        session_id="s-chain",
        source_url="https://example.com/report.pdf",
        source_name="report.pdf",
        source_kind="url",
        status="pending",
        trace_id="trace-chain",
    )
    monkeypatch.setattr(
        api_server,
        "_get_mineru_task_status",
        lambda task_id: {
            "task_id": task_id,
            "status": "done",
            "done": True,
            "trace_id": "trace-chain",
            "full_zip_url": "https://example.com/full.zip",
        },
        raising=False,
    )
    monkeypatch.setattr(api_server, "_download_mineru_result_zip", lambda url: b"zip-bytes", raising=False)
    monkeypatch.setattr(
        api_server,
        "_extract_mineru_text_from_zip",
        lambda zip_bytes: ("过敏史：青霉素过敏。", {"picked": "result.md"}),
        raising=False,
    )

    class FakeVS:
        def add_documents(self, documents):
            return None

        def persist(self):
            return None

    monkeypatch.setattr(api_server, "_get_vectordb_for_ocr", lambda: FakeVS(), raising=False)

    client = TestClient(api_server.app)
    ocr_resp = client.get("/v1/ocr/status/task-chain-ocr")
    assert ocr_resp.status_code == 200

    confirm = graph.run_chat_v2_turn(
        session_id="s-chain",
        user_message="是，我对青霉素过敏",
        use_rerank=False,
    )
    assert confirm["trace"]["record_confirmation"]["confirmed_count"] == 1

    out = graph.run_chat_v2_turn(
        session_id="s-chain",
        user_message="我24岁，女，喉咙痛两天了，3/10，不发烧。",
        use_rerank=False,
    )

    assert "建议先口服阿莫西林" not in out["answer"]
    assert "对乙酰氨基酚" in out["answer"]
    assert "record_confirmation" in out["trace"]
    assert "medication_safety" in out["trace"]
