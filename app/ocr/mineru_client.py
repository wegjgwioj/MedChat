# -*- coding: utf-8 -*-
"""MinerU OCR client helpers."""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx


MINERU_DEFAULT_BASE = "https://mineru.net"


@dataclass
class MineruTask:
    task_id: str
    trace_id: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    source_url: Optional[str] = None


@dataclass
class MineruTaskStatus:
    task_id: str
    state: str
    trace_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    full_zip_url: Optional[str] = None

    @property
    def is_done(self) -> bool:
        return str(self.state or "").strip().lower() in {"done", "completed", "finished", "success", "succeeded"}


class MineruError(RuntimeError):
    pass


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _base_url() -> str:
    return _env_str("MINERU_BASE_URL", MINERU_DEFAULT_BASE).rstrip("/")


def _timeout() -> float:
    raw = _env_str("MINERU_TIMEOUT", "20")
    try:
        return float(raw)
    except Exception:
        return 20.0


def _auth_headers() -> Dict[str, str]:
    token = _env_str("MINERU_TOKEN", "")
    if not token:
        raise MineruError("MINERU_TOKEN is required")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _first_list_item(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first
    if isinstance(payload, dict):
        return payload
    return {}


def _extract_task_id(data: Dict[str, Any]) -> str:
    return str(
        data.get("batch_id")
        or data.get("task_id")
        or data.get("id")
        or ""
    ).strip()


def create_task_from_url(
    file_url: str,
    *,
    file_name: str = "",
    is_ocr: bool = True,
    language: str = "ch",
    enable_table: bool = True,
    enable_formula: bool = False,
) -> MineruTask:
    payload = {
        "language": language,
        "enable_table": bool(enable_table),
        "enable_formula": bool(enable_formula),
        "files": [
            {
                "url": file_url,
                "name": file_name or os.path.basename(file_url) or "remote-file",
                "is_ocr": bool(is_ocr),
            }
        ],
    }
    url = f"{_base_url()}/api/v4/extract/task/batch"
    with httpx.Client(timeout=_timeout()) as client:
        resp = client.post(url, headers=_auth_headers(), json=payload)
    if resp.status_code >= 400:
        raise MineruError(f"MinerU create URL task failed: HTTP {resp.status_code}")
    body = resp.json() if resp.content else {}
    data = body.get("data") or {}
    task_id = _extract_task_id(data)
    if not task_id:
        raise MineruError("MinerU create URL task failed: missing batch_id")
    return MineruTask(task_id=task_id, trace_id=str(body.get("trace_id") or "") or None, raw=body, source_url=file_url)


def create_upload_target(file_name: str, content_type: str | None = None) -> MineruTask:
    payload = {
        "enable_url_extraction": True,
        "files": [
            {
                "name": file_name or "upload.bin",
                "content_type": content_type or "application/octet-stream",
            }
        ],
    }
    url = f"{_base_url()}/api/v4/file-urls/batch"
    with httpx.Client(timeout=_timeout()) as client:
        resp = client.post(url, headers=_auth_headers(), json=payload)
    if resp.status_code >= 400:
        raise MineruError(f"MinerU create upload target failed: HTTP {resp.status_code}")
    body = resp.json() if resp.content else {}
    data = body.get("data") or {}
    task_id = _extract_task_id(data)
    file_info = _first_list_item(data.get("file_urls") or data.get("files") or [])
    upload_url = str(file_info.get("url") or file_info.get("upload_url") or file_info.get("put_url") or "").strip()
    if not task_id or not upload_url:
        raise MineruError("MinerU create upload target failed: missing batch_id or upload url")
    return MineruTask(task_id=task_id, trace_id=str(body.get("trace_id") or "") or None, raw={"upload_url": upload_url, **body}, source_url=file_name)


def upload_file_to_presigned_url(upload_url: str, file_bytes: bytes, content_type: str | None = None) -> None:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    with httpx.Client(timeout=_timeout()) as client:
        resp = client.put(upload_url, content=file_bytes, headers=headers)
    if resp.status_code >= 400:
        raise MineruError(f"MinerU upload failed: HTTP {resp.status_code}")


def get_task_status(task_id: str) -> MineruTaskStatus:
    url = f"{_base_url()}/api/v4/extract-results/batch/{task_id}"
    with httpx.Client(timeout=_timeout()) as client:
        resp = client.get(url, headers=_auth_headers())
    if resp.status_code >= 400:
        raise MineruError(f"MinerU task status failed: HTTP {resp.status_code}")

    body = resp.json() if resp.content else {}
    data = body.get("data") or {}
    result = _first_list_item(
        data.get("extract_result")
        or data.get("extract_results")
        or data.get("results")
        or data.get("data")
        or {}
    )
    state = str(result.get("state") or data.get("state") or "").strip()
    full_zip_url = str(result.get("full_zip_url") or data.get("full_zip_url") or "").strip() or None
    return MineruTaskStatus(
        task_id=task_id,
        state=state,
        trace_id=str(body.get("trace_id") or "") or None,
        data=result or data,
        full_zip_url=full_zip_url,
    )


def download_result_zip(full_zip_url: str) -> bytes:
    with httpx.Client(timeout=_timeout()) as client:
        resp = client.get(full_zip_url)
    if resp.status_code >= 400:
        raise MineruError(f"Download result failed: HTTP {resp.status_code}")
    return resp.content


def extract_best_text_from_zip(zip_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"picked": None, "candidates": []}
    if not zip_bytes:
        return "", meta

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        meta["candidates"] = names
        md_files = [n for n in names if n.lower().endswith(".md")]
        txt_files = [n for n in names if n.lower().endswith(".txt")]
        json_files = [n for n in names if n.lower().endswith(".json")]

        for bucket in (md_files, txt_files):
            if not bucket:
                continue
            picked = bucket[0]
            meta["picked"] = picked
            raw = zf.read(picked)
            return raw.decode("utf-8", errors="ignore"), meta

        if json_files:
            picked = json_files[0]
            meta["picked"] = picked
            raw = zf.read(picked)
            try:
                obj = json.loads(raw.decode("utf-8", errors="ignore"))
            except Exception:
                return raw.decode("utf-8", errors="ignore"), meta

            if isinstance(obj, dict):
                for key in ("markdown", "text", "content"):
                    value = obj.get(key)
                    if isinstance(value, str) and value.strip():
                        return value, meta
            return raw.decode("utf-8", errors="ignore"), meta

    return "", meta
