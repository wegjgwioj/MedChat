# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _setup_ocr_store(api_server, tmp_path: Path):
    from app.agent.storage_sqlite import SqliteSessionStore

    store = SqliteSessionStore(tmp_path / "ocr_tasks.sqlite3")
    return store


def test_ocr_ingest_url_creates_task_record(monkeypatch, tmp_path: Path):
    import app.api_server as api_server

    store = _setup_ocr_store(api_server, tmp_path)
    monkeypatch.setattr(api_server, "_OCR_STORE", store, raising=False)
    monkeypatch.setattr(
        api_server,
        "_create_mineru_task_from_url",
        lambda file_url: {"task_id": "task-url-1", "trace_id": "trace-url-1", "source_url": file_url},
        raising=False,
    )

    client = TestClient(api_server.app)
    resp = client.post(
        "/v1/ocr/ingest",
        json={"session_id": "s-url", "file_url": "https://example.com/report.pdf"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s-url"
    assert body["task_id"] == "task-url-1"
    assert body["status"] == "pending"

    rec = store.get_ocr_task("task-url-1")
    assert rec is not None
    assert rec["session_id"] == "s-url"
    assert rec["source_url"] == "https://example.com/report.pdf"
    assert rec["source_kind"] == "url"


def test_ocr_status_is_idempotent_for_same_task(monkeypatch, tmp_path: Path):
    import app.api_server as api_server

    store = _setup_ocr_store(api_server, tmp_path)
    monkeypatch.setattr(api_server, "_OCR_STORE", store, raising=False)
    store.upsert_ocr_task(
        task_id="task-done-1",
        session_id="s1",
        source_url="https://example.com/report.pdf",
        source_name="report.pdf",
        source_kind="url",
        status="pending",
        trace_id="trace-1",
    )

    monkeypatch.setattr(
        api_server,
        "_get_mineru_task_status",
        lambda task_id: {"task_id": task_id, "status": "done", "done": True, "trace_id": "trace-1", "full_zip_url": "https://example.com/full.zip"},
        raising=False,
    )
    monkeypatch.setattr(api_server, "_download_mineru_result_zip", lambda url: b"zip-bytes", raising=False)
    monkeypatch.setattr(
        api_server,
        "_extract_mineru_text_from_zip",
        lambda zip_bytes: ("有效 OCR 文本内容，足够入库。", {"picked": "result.md"}),
        raising=False,
    )

    calls = {"add": 0, "persist": 0}

    class FakeVS:
        def add_documents(self, documents):
            calls["add"] += 1

        def persist(self):
            calls["persist"] += 1

    monkeypatch.setattr(api_server, "_get_vectordb_for_ocr", lambda: FakeVS(), raising=False)

    client = TestClient(api_server.app)
    resp1 = client.get("/v1/ocr/status/task-done-1")
    resp2 = client.get("/v1/ocr/status/task-done-1")

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["ingested"] is True
    assert resp2.json()["ingested"] is True
    assert calls["add"] == 1
    assert calls["persist"] == 1


def test_ocr_status_chunks_long_text_before_ingest(monkeypatch, tmp_path: Path):
    import app.api_server as api_server

    monkeypatch.setenv("RAG_CHUNK_SIZE", "4")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "1")

    store = _setup_ocr_store(api_server, tmp_path)
    monkeypatch.setattr(api_server, "_OCR_STORE", store, raising=False)
    store.upsert_ocr_task(
        task_id="task-long-ocr",
        session_id="s-long",
        source_url="https://example.com/long.pdf",
        source_name="long.pdf",
        source_kind="url",
        status="pending",
        trace_id="trace-long",
    )

    monkeypatch.setattr(
        api_server,
        "_get_mineru_task_status",
        lambda task_id: {"task_id": task_id, "status": "done", "done": True, "trace_id": "trace-long", "full_zip_url": "https://example.com/full.zip"},
        raising=False,
    )
    monkeypatch.setattr(api_server, "_download_mineru_result_zip", lambda url: b"zip-bytes", raising=False)
    monkeypatch.setattr(
        api_server,
        "_extract_mineru_text_from_zip",
        lambda zip_bytes: (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
            {"picked": "result.md"},
        ),
        raising=False,
    )

    captured = {"chunks": 0}

    class FakeVS:
        def add_documents(self, documents):
            captured["chunks"] = len(documents)

        def persist(self):
            return None

    monkeypatch.setattr(api_server, "_get_vectordb_for_ocr", lambda: FakeVS(), raising=False)

    client = TestClient(api_server.app)
    resp = client.get("/v1/ocr/status/task-long-ocr")

    assert resp.status_code == 200
    assert resp.json()["ingested"] is True
    assert captured["chunks"] >= 2


def test_ocr_ingest_accepts_file_upload(monkeypatch, tmp_path: Path):
    import app.api_server as api_server

    store = _setup_ocr_store(api_server, tmp_path)
    monkeypatch.setattr(api_server, "_OCR_STORE", store, raising=False)

    captured = {}

    def fake_create_from_upload(file_name: str, content_type: str | None, file_bytes: bytes):
        captured["file_name"] = file_name
        captured["content_type"] = content_type
        captured["size"] = len(file_bytes)
        return {"task_id": "task-file-1", "trace_id": "trace-file-1", "source_url": file_name}

    monkeypatch.setattr(api_server, "_create_mineru_task_from_upload", fake_create_from_upload, raising=False)

    client = TestClient(api_server.app)
    resp = client.post(
        "/v1/ocr/ingest",
        data={"session_id": "s-file"},
        files={"file": ("report.pdf", b"%PDF-1.7 mock", "application/pdf")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "task-file-1"
    assert captured == {
        "file_name": "report.pdf",
        "content_type": "application/pdf",
        "size": 13,
    }

    rec = store.get_ocr_task("task-file-1")
    assert rec is not None
    assert rec["source_kind"] == "upload"
    assert rec["source_name"] == "report.pdf"


def test_ocr_status_writes_pending_record_fact_into_agent_session(monkeypatch, tmp_path):
    import app.api_server as api_server
    from app.agent.storage_sqlite import SqliteSessionStore

    db_path = tmp_path / "agent_sessions.sqlite3"
    store = SqliteSessionStore(db_path)
    monkeypatch.setattr(api_server, "_OCR_STORE", store, raising=False)
    monkeypatch.setenv("AGENT_SESSION_STORE", "sqlite")
    monkeypatch.setenv("AGENT_SQLITE_DB_PATH", str(db_path))

    store.upsert_ocr_task(
        task_id="task-done-ocr-fact",
        session_id="s-ocr",
        source_url="https://example.com/report.pdf",
        source_name="report.pdf",
        source_kind="url",
        status="pending",
        trace_id="trace-ocr",
    )

    monkeypatch.setattr(
        api_server,
        "_get_mineru_task_status",
        lambda task_id: {
            "task_id": task_id,
            "status": "done",
            "done": True,
            "trace_id": "trace-ocr",
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
    resp = client.get("/v1/ocr/status/task-done-ocr-fact")

    session = store.load_session("s-ocr")
    assert resp.status_code == 200
    assert session is not None
    assert session.pending_record_facts[0].value == "青霉素过敏"
    assert session.pending_record_facts[0].status == "pending"
    assert session.record_summary == ""
