# -*- coding: utf-8 -*-
"""
api_server.py
阶段2：FastAPI接口包装
- POST /v1/triage
- GET  /health
"""

import logging
import os
import contextvars
import json
import hashlib
import time
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Optional, Literal, Any, Dict

from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

try:
    # Prefer package-relative import so `import app.api_server` works in tests.
    from .triage_service import triage_once  # type: ignore
except Exception:
    # Fallback for running with different PYTHONPATH / working directory.
    from triage_service import triage_once  # type: ignore

try:
    # Stepwise triage helpers for LangGraph orchestration.
    from .triage_service import (  # type: ignore
        build_rag_query,
        triage_step_retrieve,
        triage_step_assess,
        triage_step_safety,
        triage_step_build_payload,
    )
except Exception:
    from triage_service import (  # type: ignore
        build_rag_query,
        triage_step_retrieve,
        triage_step_assess,
        triage_step_safety,
        triage_step_build_payload,
    )


logger = logging.getLogger(__name__)


_TRACE_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)

from app.agent.storage_sqlite import SqliteSessionStore

_OCR_STORE = SqliteSessionStore()


class _TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _TRACE_ID.get() or "-"
        return True


logger.addFilter(_TraceIdFilter())


def _get_cors_allow_origins() -> list[str]:
    """Return CORS allowlist.

    Environment:
      TRIAGE_CORS_ORIGINS: comma-separated origins.
        Example: "http://localhost:5173,http://127.0.0.1:5173"

    Defaults (for local demo):
      - http://localhost:5173
      - http://127.0.0.1:5173
    """

    raw = (os.getenv("TRIAGE_CORS_ORIGINS") or "").strip()
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        # De-dup while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for o in origins:
            if o not in seen:
                seen.add(o)
                out.append(o)
        return out

    return ["http://localhost:5173", "http://127.0.0.1:5173"]


class TriageRequest(BaseModel):
    user_text: str = Field(..., description="用户描述(一句话/一段话都可以)")
    top_k: int = Field(5, ge=1, le=20, description="RAG召回数量")
    mode: Literal["fast", "safe"] = Field("fast", description="fast更快；safe会跑安全审查链更慢更稳")
    clinical_record_path: Optional[str] = Field(
        None,
        description="可选：病历/规则摘要文本路径（不强依赖；可留空）。",
    )


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="会话ID；不传则自动生成")
    patient_message: str = Field(..., description="患者输入")
    top_k: int = Field(5, ge=1, le=20, description="RAG召回数量（进入DIAGNOSIS时使用）")
    mode: Literal["fast", "safe"] = Field("fast", description="同 /v1/triage；默认会被API策略强制为safe")


class RagRetrieveRequest(BaseModel):
    query: str = Field(..., description="检索 query")
    top_k: int = Field(5, ge=1, le=20, description="最终返回条数")
    top_n: int = Field(30, ge=1, le=200, description="第一阶段向量召回条数")
    department: Optional[str] = Field(None, description="可选：科室过滤（严格等值匹配）")
    use_rerank: Optional[bool] = Field(None, description="可选：是否启用 rerank（默认按环境变量）")


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_choice(name: str, default: str) -> str:
    v = (os.getenv(name) or "").strip().lower()
    return v or default


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "README.md").exists() and (p / "app").is_dir():
            return p
    return start.parent


def _get_output_dir() -> Path:
    repo_root = _find_repo_root(Path(__file__))
    raw = (os.getenv("OUTPUT_DIR") or "outputs").strip()
    out = Path(raw)
    if not out.is_absolute():
        out = repo_root / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def _create_mineru_task_from_url(file_url: str) -> Dict[str, Any]:
    from app.ocr.mineru_client import create_task_from_url

    task = create_task_from_url(file_url)
    return {
        "task_id": task.task_id,
        "trace_id": task.trace_id,
        "source_url": task.source_url or file_url,
    }


def _create_mineru_task_from_upload(file_name: str, content_type: Optional[str], file_bytes: bytes) -> Dict[str, Any]:
    from app.ocr.mineru_client import create_upload_target, upload_file_to_presigned_url

    task = create_upload_target(file_name=file_name, content_type=content_type)
    upload_url = str((task.raw or {}).get("upload_url") or "").strip()
    if not upload_url:
        raise RuntimeError("MinerU 未返回上传地址")
    upload_file_to_presigned_url(upload_url, file_bytes=file_bytes, content_type=content_type)
    return {
        "task_id": task.task_id,
        "trace_id": task.trace_id,
        "source_url": file_name,
    }


def _get_mineru_task_status(task_id: str) -> Dict[str, Any]:
    from app.ocr.mineru_client import get_task_status

    st = get_task_status(task_id)
    return {
        "task_id": st.task_id,
        "status": st.state,
        "done": st.is_done,
        "trace_id": st.trace_id,
        "full_zip_url": st.full_zip_url,
        "data": st.data or {},
    }


def _download_mineru_result_zip(full_zip_url: str) -> bytes:
    from app.ocr.mineru_client import download_result_zip

    return download_result_zip(full_zip_url)


def _extract_mineru_text_from_zip(zip_bytes: bytes) -> tuple[str, Dict[str, Any]]:
    from app.ocr.mineru_client import extract_best_text_from_zip

    return extract_best_text_from_zip(zip_bytes)


def _get_vectordb_for_ocr():
    from app.rag.rag_core import get_vectordb

    return get_vectordb()


async def _parse_ocr_ingest_payload(request: Request) -> Dict[str, Any]:
    ctype = (request.headers.get("content-type") or "").lower()

    session_id = ""
    file_url = ""
    source = ""
    upload_name = ""
    upload_type = ""
    upload_bytes = b""

    if "application/json" in ctype:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("OCR 请求体必须是 JSON object")
        session_id = str(body.get("session_id") or "").strip()
        file_url = str(body.get("file_url") or "").strip()
        source = str(body.get("source") or "").strip()
    else:
        form = await request.form()
        session_id = str(form.get("session_id") or "").strip()
        file_url = str(form.get("file_url") or "").strip()
        source = str(form.get("source") or "").strip()
        file_obj = form.get("file")
        if isinstance(file_obj, StarletteUploadFile):
            upload_name = str(file_obj.filename or "").strip()
            upload_type = str(file_obj.content_type or "").strip()
            upload_bytes = await file_obj.read()

    return {
        "session_id": session_id,
        "file_url": file_url,
        "source": source,
        "upload_name": upload_name,
        "upload_type": upload_type,
        "upload_bytes": upload_bytes,
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def _safe_query_for_log(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "(empty)"
    prefix = q[:100]
    if len(q) <= 100:
        return prefix
    return f"{prefix}…(sha256={_sha256_text(q)[:12]})"


def _text_meta(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {"present": False, "length": 0, "sha256": None}
    return {"present": True, "length": len(text), "sha256": _sha256_text(text)}


def _session_file_path(session_id: str) -> Path:
    base = _get_output_dir() / "sessions"
    base.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9._-]", "_", session_id)[:128] or "session"
    return base / f"{safe_id}.json"


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_or_create_session(session_id: str) -> Dict[str, Any]:
    p = _session_file_path(session_id)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "session_id": session_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "turns": [],
        "intake_slots": {},
    }


def _save_session(session: Dict[str, Any]) -> None:
    session["updated_at"] = _utc_now()
    p = _session_file_path(str(session.get("session_id") or "session"))
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def _looks_like_explanation_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or "?" not in t:
        return False
    return any(k in t for k in ["what is", "what's", "meaning", "define", "explain", "why", "how does"])


def _looks_like_recommendation_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or "?" not in t:
        return False
    return any(k in t for k in ["should i", "can i", "what should", "treat", "medicine", "medication", "diet", "exercise"])


def _looks_like_greeting_only(text: str) -> bool:
    """Return True if the message is essentially a greeting and contains no symptom info."""

    t = (text or "").strip().lower()
    if not t:
        return True

    # Remove common punctuation/emoji-like chars
    t2 = re.sub(r"[\s\t\n\r\-_,.!?，。！？、~…]+", "", t)
    if not t2:
        return True

    greetings = {
        "你好",
        "您好",
        "哈喽",
        "嗨",
        "在吗",
        "早",
        "早上好",
        "下午好",
        "晚上好",
        "hello",
        "hi",
        "hey",
    }

    if t2 in greetings:
        return True

    # Short variants like "你好呀" / "hi!" etc.
    for g in ["你好", "您好", "哈喽", "hello", "hi", "hey"]:
        if t2 == g or t2 == f"{g}呀" or t2 == f"{g}啊" or t2 == f"{g}呢":
            return True

    return False


def _slot_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"unknown", "n/a", "na"}
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _infer_onset_from_duration_if_missing(intake_slots: Dict[str, Any]) -> None:
    """Infer onset like "10分钟前" from a known duration when onset is missing.

    This is a best-effort shortcut to reduce redundant questions.
    """

    try:
        if not isinstance(intake_slots, dict):
            return
        if not _slot_is_missing(intake_slots.get("onset")):
            return
        duration = intake_slots.get("duration")
        if _slot_is_missing(duration):
            return

        d = str(duration).strip()
        if not d:
            return

        # Avoid making up onset from vague durations.
        if d in {"较久", "不久", "不确定"}:
            return

        # If duration already contains "前"/"开始" etc, treat as onset.
        if any(x in d for x in ["前", "开始", "起"]):
            intake_slots["onset"] = d
            return

        # Common formats: "10分钟" "2小时" "3天" "一周左右"
        if re.search(r"(分钟|小时|天|周|个月|月|年)", d):
            intake_slots["onset"] = f"{d}前"
    except Exception:
        return


def _select_chat_act(patient_message: str, intake_slots: Dict[str, Any], turn_index: int, max_turns: int) -> str:
    if _looks_like_explanation_request(patient_message):
        return "EXPLANATION"
    if _looks_like_recommendation_request(patient_message):
        return "RECOMMENDATION"

    if max_turns > 0 and turn_index >= max_turns:
        return "DIAGNOSIS"

    red_flags = intake_slots.get("red_flags")
    if isinstance(red_flags, list) and any(str(x).strip() for x in red_flags):
        return "DIAGNOSIS"

    # Short-circuit policy for demo UX:
    # If we already have a chief complaint AND any time-course info (onset/duration/pattern),
    # proceed to DIAGNOSIS. Missing details can be returned as key_questions in triage JSON.
    if not _slot_is_missing(intake_slots.get("chief_complaint")):
        has_time = any(
            not _slot_is_missing(intake_slots.get(k)) for k in ["onset", "duration", "pattern"]
        )
        if has_time:
            return "DIAGNOSIS"

    # Otherwise keep asking essentials.
    essential_keys = ["chief_complaint", "onset"]
    missing = [k for k in essential_keys if _slot_is_missing(intake_slots.get(k))]
    if _slot_is_missing(intake_slots.get("duration")) and _slot_is_missing(intake_slots.get("pattern")):
        missing.append("duration")
    return "INQUIRY" if missing else "DIAGNOSIS"


def _first_missing_slot(intake_slots: Dict[str, Any], *, skip: Optional[set[str]] = None) -> Optional[str]:
    skip = skip or set()
    # Deterministic single-question policy (no extra LLM call needed)
    order = ["chief_complaint", "onset", "duration", "pattern", "severity", "associated_symptoms", "red_flags"]
    for k in order:
        if k in skip:
            continue
        if _slot_is_missing(intake_slots.get(k)):
            return k
    return None


def _next_question_for_missing(intake_slots: Dict[str, Any], *, force_slot: Optional[str] = None, repeat_count: int = 0) -> str:
    k = force_slot or _first_missing_slot(intake_slots)
    if not k:
        return "还有哪些你觉得重要的信息想补充吗？"

    if k == "chief_complaint":
        return "请用一句话描述你最主要的不适是什么？"
    if k == "onset":
        return "这些症状是从什么时候开始的？（例如：今天/昨天/3天前）"
    if k == "duration":
        if repeat_count >= 1:
            return "我可能没听清：症状大概持续多久？可以这样回答：‘3小时/2天/一周左右’。如果说不准，也可以说‘不确定’。"
        return "症状大概持续多久？（例如：几小时/几天/一周左右）"
    if k == "pattern":
        if repeat_count >= 1:
            return "症状是一直都有，还是一阵一阵反复发作？可以回答：‘一直都有’或‘间歇发作’。"
        return "症状是持续存在还是间歇发作？（例如：一直都有 / 时好时坏）"
    if k == "severity":
        return "请用0-10分描述严重程度（0不痛/10最严重），目前大概几分？"
    if k == "associated_symptoms":
        return "还有没有伴随症状（比如发烧、胸痛、呼吸困难、呕吐、皮疹等）？"
    if k == "red_flags":
        return "有没有出现危险信号：胸痛/呼吸困难/意识改变/抽搐/严重过敏/持续高热等？"
    return "还有哪些你觉得重要的信息想补充吗？"


def _update_intake_slots_from_message(intake_slots: Dict[str, Any], patient_message: str) -> None:
    """Very small rule-based slot updates to make /v1/chat progress.

    Notes:
    - Keep deterministic and lightweight (no extra LLM calls).
    - Store short strings only; do not attempt to normalize too hard.
    """

    msg = (patient_message or "").strip()
    if not msg:
        return

    low = msg.lower()

    # severity: patterns like "7/10" or "7分" or "0-10"
    if _slot_is_missing(intake_slots.get("severity")):
        m = re.search(r"\b(10|[0-9])\s*/\s*10\b", low)
        if m:
            intake_slots["severity"] = int(m.group(1))
        else:
            m2 = re.search(r"(10|[0-9])\s*分", msg)
            if m2:
                intake_slots["severity"] = int(m2.group(1))

    # duration: prefer concrete time spans
    if _slot_is_missing(intake_slots.get("duration")):
        # Direct compact answers like "3小时" / "2天" / "一周左右"
        m0 = re.match(r"^\s*([0-9一二三四五六七八九十]+)\s*(分钟|小时|天|周|个月|月|年)\s*(左右|多|余|来)?\s*$", msg)
        if m0:
            intake_slots["duration"] = f"{m0.group(1)}{m0.group(2)}"

        m = re.search(r"(已经|持续|大概|约)\s*([0-9一二三四五六七八九十]+)\s*(分钟|小时|天|周|个月|月|年)", msg)
        if m:
            intake_slots["duration"] = f"{m.group(2)}{m.group(3)}"
        else:
            # looser expressions
            for pat, val in [
                (r"(好几天|几天|数天)", "几天"),
                (r"(好几周|几周)", "几周"),
                (r"(好几个月|几个月)", "几个月"),
                (r"(很久|挺久|很长时间)", "较久"),
                (r"(刚开始|刚刚|不久)", "不久"),
            ]:
                if re.search(pat, msg):
                    intake_slots["duration"] = val
                    break

    # onset: "X天前开始" / "从...开始" / "今天/昨天" 等
    if _slot_is_missing(intake_slots.get("onset")):
        m = re.search(r"([0-9一二三四五六七八九十]+)\s*(分钟|小时|天|周|个月|月|年)\s*前(开始|出现)?", msg)
        if m:
            intake_slots["onset"] = f"{m.group(1)}{m.group(2)}前"
        elif "从" in msg and "开始" in msg:
            m2 = re.search(r"从(.{1,40}?)开始", msg)
            if m2:
                intake_slots["onset"] = m2.group(1)
        else:
            for kw in [
                "今天",
                "昨天",
                "前天",
                "刚刚",
                "刚才",
                "昨晚",
                "今晚",
                "今早",
                "今天早上",
                "今天上午",
                "今天中午",
                "今天下午",
                "今天晚上",
                "上周",
                "本周",
                "上个月",
                "最近",
            ]:
                if kw in msg:
                    intake_slots["onset"] = kw
                    break

    # pattern: continuous vs intermittent (fuzzy synonyms)
    if _slot_is_missing(intake_slots.get("pattern")):
        pattern_map = [
            (r"(间歇性|间歇|断断续续|时有时无|一阵一阵|反反复复|反复发作|时好时坏|偶尔|有时候|有时)", "间歇"),
            (r"(持续性|一直|持续|不停|连续|每天都|一直都有|一直在)", "持续"),
        ]
        for pat, val in pattern_map:
            if re.search(pat, msg):
                intake_slots["pattern"] = val
                break

    # red flags (very small keyword list)
    red_flag_keywords = [
        "胸痛",
        "呼吸困难",
        "喘不上气",
        "意识不清",
        "昏迷",
        "抽搐",
        "剧烈头痛",
        "颈部僵硬",
        "口唇发紫",
        "严重过敏",
        "喉头水肿",
        "偏瘫",
        "说话不清",
    ]
    if _slot_is_missing(intake_slots.get("red_flags")):
        hits = [k for k in red_flag_keywords if k in msg]
        if hits:
            intake_slots["red_flags"] = hits[:6]


def _coerce_llm_slots(raw: Any) -> Dict[str, Any]:
    """Coerce LLM-returned slots into a safe, small dict."""

    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Any] = {}

    def _pick_str(key: str, max_len: int = 80) -> Optional[str]:
        v = raw.get(key)
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in {"unknown", "n/a", "na", "不确定", "不知道"}:
            return None
        return s[:max_len]

    for k in ["chief_complaint", "onset", "duration", "pattern", "associated_symptoms"]:
        s = _pick_str(k, max_len=200 if k == "chief_complaint" else 80)
        if s is not None:
            out[k] = s

    sev = raw.get("severity")
    try:
        if isinstance(sev, str):
            m = re.search(r"\b(10|[0-9])\b", sev)
            sev2 = int(m.group(1)) if m else None
        elif isinstance(sev, (int, float)):
            sev2 = int(sev)
        else:
            sev2 = None
        if sev2 is not None:
            sev2 = max(0, min(10, sev2))
            out["severity"] = sev2
    except Exception:
        pass

    rf = raw.get("red_flags")
    if isinstance(rf, list):
        cleaned = [str(x).strip() for x in rf if str(x).strip()]
        if cleaned:
            out["red_flags"] = cleaned[:8]
    elif isinstance(rf, str):
        # allow comma/、 separated
        parts = [p.strip() for p in re.split(r"[,，、;；]\s*", rf) if p.strip()]
        if parts:
            out["red_flags"] = parts[:8]

    return out


def _update_intake_slots_with_llm(intake_slots: Dict[str, Any], patient_message: str) -> bool:
    """Try to update slots using LLM; return True if succeeded.

    This is best-effort and must never raise.
    """

    msg = (patient_message or "").strip()
    if not msg:
        return False

    try:
        # Local import: avoid hard dependency at import time.
        try:
            from . import config_llm  # type: ignore
        except Exception:
            import config_llm  # type: ignore

        llm = config_llm.get_llm(temperature=0.0)

        prompt = (
            "你是医疗问诊助手。请从患者一句话中抽取结构化信息。\n"
            "只输出严格 JSON，不要输出其它文字。\n\n"
            "字段：\n"
            "- chief_complaint: 主要不适（短句）\n"
            "- onset: 起病时间（如 今天/昨天/3天前/上周/不确定）\n"
            "- duration: 持续多久（如 3小时/2天/一周左右/不确定）\n"
            "- pattern: 持续或间歇（持续/间歇/不确定）\n"
            "- severity: 0-10 数字（可为空）\n"
            "- associated_symptoms: 伴随症状（短句，可为空）\n"
            "- red_flags: 危险信号关键词列表（可为空）\n\n"
            f"患者输入：{msg}\n"
        )

        # LangChain ChatOpenAI: prefer predict() for older versions.
        raw_text = None
        if hasattr(llm, "predict"):
            raw_text = llm.predict(prompt)  # type: ignore[attr-defined]
        else:
            raw_text = str(llm(prompt))

        text = (raw_text or "").strip()
        # Extract first JSON object if model wrapped it.
        m = re.search(r"\{.*\}", text, flags=re.S)
        json_text = m.group(0) if m else text
        parsed = json.loads(json_text)
        slots = _coerce_llm_slots(parsed)
        if not slots:
            return False

        # Only fill missing slots to avoid overwriting earlier user-provided data.
        for k, v in slots.items():
            if _slot_is_missing(intake_slots.get(k)):
                intake_slots[k] = v
        return True
    except Exception as e:
        logger.warning("LLM slot extraction failed; falling back to rules: %s", e)
        return False


def _format_doctor_reply_from_answer(answer: Any) -> str:
    """Format a patient-facing reply from canonical triage answer JSON.

    Keep it deterministic and readable (no extra LLM call).
    """

    if not isinstance(answer, dict):
        return "我还需要更多信息才能给出初步建议。"

    triage_level = str(answer.get("triage_level") or "").strip() or "ROUTINE"

    def _list(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        return [s] if s else []

    red_flags = _list(answer.get("red_flags"))
    immediate_actions = _list(answer.get("immediate_actions"))
    what_not_to_do = _list(answer.get("what_not_to_do"))
    reasoning = str(answer.get("reasoning") or "").strip()
    uncertainty = str(answer.get("uncertainty") or "").strip()
    safety_notice = str(answer.get("safety_notice") or "").strip()

    level_cn = {
        "EMERGENCY_NOW": "立即急诊/呼叫急救",
        "URGENT_24H": "建议24小时内就医",
        "ROUTINE": "常规就诊/观察",
        "SELF_CARE": "居家护理为主",
    }.get(triage_level, triage_level)

    lines: list[str] = []
    lines.append(f"分诊级别：{level_cn}")

    if red_flags:
        lines.append("")
        lines.append("危险信号（如出现请尽快就医/急救）：")
        for x in red_flags:
            lines.append(f"- {x}")

    if immediate_actions:
        lines.append("")
        lines.append("现在可以做的事：")
        for x in immediate_actions:
            lines.append(f"- {x}")

    if what_not_to_do:
        lines.append("")
        lines.append("避免做的事：")
        for x in what_not_to_do:
            lines.append(f"- {x}")

    if reasoning:
        lines.append("")
        lines.append("简要解释：")
        lines.append(reasoning)

    if uncertainty:
        lines.append("")
        lines.append(f"不确定性/需要补充：{uncertainty}")

    if safety_notice:
        lines.append("")
        lines.append(safety_notice)

    return "\n".join(lines).strip() + "\n"


class _ChatGraphState(TypedDict, total=False):
    patient_message: str
    top_k: int
    mode: str
    forced_safe: bool
    max_turns: int
    turns_len: int
    intake_slots: Dict[str, Any]
    inquiry_state: Dict[str, Any]

    greeting_only: bool
    handled_reply: bool
    act: str
    doctor_reply: str
    triage_payload: Optional[Dict[str, Any]]

    rag_query: str
    evidence_list: list
    evidence_block: str
    answer_json: Dict[str, Any]
    rag_status: str
    rag_error: Optional[Dict[str, Any]]
    triage_trace: list


def _append_chat_trace(trace: Any, step: str, status: str, t0: float, **info) -> list:
    if not isinstance(trace, list):
        trace = []
    try:
        ms = int((time.perf_counter() - t0) * 1000)
    except Exception:
        ms = None
    entry: Dict[str, Any] = {"step": step, "status": status}
    if ms is not None:
        entry["ms"] = ms
    for k, v in info.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            entry[k] = v
        elif isinstance(v, list):
            entry[k] = v
        elif isinstance(v, dict):
            entry[k] = v
    trace.append(entry)
    return trace


_CHAT_GRAPH = None


def _get_chat_graph():
    """Build and cache the LangGraph chat state machine.

    This wraps the existing deterministic control-flow (slot fill -> decide -> inquiry/triage)
    into a graph so we can evolve it without changing the API contract.
    """

    global _CHAT_GRAPH
    if _CHAT_GRAPH is not None:
        return _CHAT_GRAPH

    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LangGraph 未安装或不可用。请在当前环境中安装：pip install langgraph==0.0.20"
        ) from e

    def _node_preprocess(state: _ChatGraphState) -> dict:
        t0 = time.perf_counter()
        patient_message = (state.get("patient_message") or "").strip()
        intake_slots = state.get("intake_slots")
        if not isinstance(intake_slots, dict):
            intake_slots = {}

        trace = state.get("triage_trace")
        if not isinstance(trace, list):
            trace = []

        greeting_only = _looks_like_greeting_only(patient_message)
        state["greeting_only"] = greeting_only

        # If user only greeted / provided no symptom info, do NOT treat it as chief complaint.
        if greeting_only and _slot_is_missing(intake_slots.get("chief_complaint")):
            trace = _append_chat_trace(trace, "INQUIRY", "ok", t0, note="greeting_only")
            return {
                "intake_slots": intake_slots,
                "greeting_only": True,
                "handled_reply": True,
                "act": "INQUIRY",
                "doctor_reply": "你好！请用一句话描述你最主要的不适是什么？比如：哪里不舒服、持续多久、严重程度大概几分。",
                "triage_payload": None,
                "triage_trace": trace,
            }

        # Minimal heuristic update: fill chief_complaint if missing using first meaningful message.
        if _slot_is_missing(intake_slots.get("chief_complaint")) and patient_message:
            intake_slots["chief_complaint"] = patient_message[:200]

        # Slot extraction policy:
        # - Default: use LLM to extract slots (more flexible for user phrasing)
        # - On any failure: fall back to deterministic heuristics
        # - To force rules only: set CHAT_SLOT_EXTRACTOR=rules
        extractor = _env_choice("CHAT_SLOT_EXTRACTOR", "llm")
        used_llm = False
        if extractor not in {"rules", "rule", "heuristic"}:
            used_llm = _update_intake_slots_with_llm(intake_slots, patient_message)
        if not used_llm:
            _update_intake_slots_from_message(intake_slots, patient_message)

        _infer_onset_from_duration_if_missing(intake_slots)

        max_turns = int((state.get("max_turns") or 8))
        turns_len = int((state.get("turns_len") or 0))
        act = _select_chat_act(patient_message, intake_slots, turn_index=turns_len + 1, max_turns=max_turns)

        trace = _append_chat_trace(trace, "PREPROCESS", "ok", t0, act=act)

        return {
            "intake_slots": intake_slots,
            "greeting_only": greeting_only,
            "handled_reply": False,
            "act": act,
            "triage_trace": trace,
        }

    def _route_after_preprocess(state: _ChatGraphState) -> str:
        if state.get("handled_reply"):
            return END
        act = str(state.get("act") or "")
        if act == "INQUIRY":
            return "inquiry"
        return "triage"

    def _node_inquiry(state: _ChatGraphState) -> dict:
        t0 = time.perf_counter()
        intake_slots = state.get("intake_slots")
        if not isinstance(intake_slots, dict):
            intake_slots = {}

        trace = state.get("triage_trace")
        if not isinstance(trace, list):
            trace = []

        inquiry_state = state.get("inquiry_state")
        if not isinstance(inquiry_state, dict):
            inquiry_state = {}

        slot = _first_missing_slot(intake_slots)
        last_slot = str(inquiry_state.get("last_slot") or "")
        repeat_count = int(inquiry_state.get("repeat_count") or 0)
        if slot and slot == last_slot:
            repeat_count += 1
        else:
            repeat_count = 0

        # If we already asked the same slot twice and it's still missing, move on to keep the demo flowing.
        if slot and repeat_count >= 2:
            slot2 = _first_missing_slot(intake_slots, skip={slot})
            if slot2:
                slot = slot2
                repeat_count = 0

        inquiry_state["last_slot"] = slot
        inquiry_state["repeat_count"] = repeat_count

        doctor_reply = _next_question_for_missing(intake_slots, force_slot=slot, repeat_count=repeat_count)
        trace = _append_chat_trace(trace, "INQUIRY", "ok", t0, slot=slot or "")
        return {
            "intake_slots": intake_slots,
            "inquiry_state": inquiry_state,
            "doctor_reply": doctor_reply,
            "triage_payload": None,
            "triage_trace": trace,
        }

    def _node_triage_retrieve(state: _ChatGraphState) -> dict:
        t0 = time.perf_counter()
        patient_message = (state.get("patient_message") or "").strip()
        top_k = int((state.get("top_k") or 5))
        trace = state.get("triage_trace")
        if not isinstance(trace, list):
            trace = []

        rag_query = build_rag_query(patient_message, "")
        out = triage_step_retrieve(rag_query=rag_query, top_k=top_k, trace=trace)
        evidence_list = out.get("evidence") if isinstance(out, dict) else []

        trace = _append_chat_trace(trace, "RETRIEVE", "ok", t0, top_k=top_k)

        return {
            "rag_query": rag_query,
            "evidence_list": evidence_list if isinstance(evidence_list, list) else [],
            "rag_status": str((out or {}).get("rag_status") or "unknown"),
            "rag_error": (out or {}).get("rag_error"),
            "triage_trace": trace,
        }

    def _node_triage_assess(state: _ChatGraphState) -> dict:
        t0 = time.perf_counter()
        patient_message = (state.get("patient_message") or "").strip()
        evidence_list = state.get("evidence_list")
        if not isinstance(evidence_list, list):
            evidence_list = []
        trace = state.get("triage_trace")
        if not isinstance(trace, list):
            trace = []

        out = triage_step_assess(user_text=patient_message, evidence_list=evidence_list, trace=trace)
        answer_json = out.get("answer") if isinstance(out, dict) else {}

        trace = _append_chat_trace(trace, "ASSESS", "ok", t0)

        return {
            "evidence_block": str((out or {}).get("evidence_block") or ""),
            "answer_json": answer_json if isinstance(answer_json, dict) else {},
            "triage_trace": trace,
        }

    def _route_after_assess(state: _ChatGraphState) -> str:
        mode = str(state.get("mode") or "safe").strip().lower()
        if mode == "safe":
            return "safety"
        return "build_payload"

    def _node_triage_safety(state: _ChatGraphState) -> dict:
        t0 = time.perf_counter()
        answer_json = state.get("answer_json")
        if not isinstance(answer_json, dict):
            answer_json = {}
        evidence_block = str(state.get("evidence_block") or "")
        evidence_list = state.get("evidence_list")
        if not isinstance(evidence_list, list):
            evidence_list = []
        trace = state.get("triage_trace")
        if not isinstance(trace, list):
            trace = []

        out = triage_step_safety(answer_json=answer_json, evidence_block=evidence_block, evidence_list=evidence_list, trace=trace)
        new_answer = out.get("answer") if isinstance(out, dict) else {}
        trace = _append_chat_trace(trace, "SAFETY", "ok", t0)
        return {"answer_json": new_answer if isinstance(new_answer, dict) else {}, "triage_trace": trace}

    def _node_triage_build_payload(state: _ChatGraphState) -> dict:
        t0 = time.perf_counter()
        rag_query = str(state.get("rag_query") or "")
        evidence_list = state.get("evidence_list")
        if not isinstance(evidence_list, list):
            evidence_list = []
        answer_json = state.get("answer_json")
        if not isinstance(answer_json, dict):
            answer_json = {}
        mode = str(state.get("mode") or "safe").strip().lower()
        forced_safe = bool(state.get("forced_safe") or False)
        rag_status = str(state.get("rag_status") or "unknown")
        rag_error = state.get("rag_error") if isinstance(state.get("rag_error"), dict) else None
        trace = state.get("triage_trace")
        if not isinstance(trace, list):
            trace = []

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        triage_payload = triage_step_build_payload(
            answer_json=answer_json,
            evidence_list=evidence_list,
            rag_query=rag_query,
            mode=mode,
            created_at=created_at,
            rag_status=rag_status,
            rag_error=rag_error,
            trace=trace,
        )

        trace = _append_chat_trace(trace, "BUILD_PAYLOAD", "ok", t0)

        if forced_safe and isinstance(triage_payload, dict):
            meta = triage_payload.get("meta")
            if isinstance(meta, dict):
                meta["forced_safe"] = True

        return {"triage_payload": triage_payload, "triage_trace": trace}

    def _node_format_reply(state: _ChatGraphState) -> dict:
        t0 = time.perf_counter()
        triage_payload = state.get("triage_payload")
        answer = triage_payload.get("answer") if isinstance(triage_payload, dict) else {}
        trace = state.get("triage_trace")
        trace = _append_chat_trace(trace, "FORMAT", "ok", t0)
        return {"doctor_reply": _format_doctor_reply_from_answer(answer), "triage_trace": trace}

    graph = StateGraph(_ChatGraphState)
    graph.add_node("preprocess", _node_preprocess)
    graph.add_node("inquiry", _node_inquiry)
    graph.add_node("triage_retrieve", _node_triage_retrieve)
    graph.add_node("triage_assess", _node_triage_assess)
    graph.add_node("triage_safety", _node_triage_safety)
    graph.add_node("triage_build_payload", _node_triage_build_payload)
    graph.add_node("format", _node_format_reply)
    graph.set_entry_point("preprocess")
    graph.add_conditional_edges(
        "preprocess",
        _route_after_preprocess,
        {
            "inquiry": "inquiry",
            "triage": "triage_retrieve",
            END: END,
        },
    )
    graph.add_edge("inquiry", END)
    graph.add_edge("triage_retrieve", "triage_assess")
    graph.add_conditional_edges(
        "triage_assess",
        _route_after_assess,
        {
            "safety": "triage_safety",
            "build_payload": "triage_build_payload",
        },
    )
    graph.add_edge("triage_safety", "triage_build_payload")
    graph.add_edge("triage_build_payload", "format")
    graph.add_edge("format", END)

    _CHAT_GRAPH = graph.compile()
    return _CHAT_GRAPH


app = FastAPI(
    title="Medical Triage API (RAG + LLM)",
    version="0.1.0",
)


# M2：AgentOrchestration（LangGraph）路由挂载。
# 采用延迟导入，避免启动阶段强制加载模型或引入不必要依赖。
try:
    from app.agent.router import router as agent_router  # type: ignore

    app.include_router(agent_router)
except Exception as e:
    logger.warning("M2 agent router not loaded: %s", e)


@app.middleware("http")
async def add_trace_id_middleware(request: Request, call_next):
    trace_id = str(uuid4())
    request.state.trace_id = trace_id
    token = _TRACE_ID.set(trace_id)
    try:
        logger.info("request start trace_id=%s method=%s path=%s", trace_id, request.method, request.url.path)
        response = await call_next(request)
        return response
    finally:
        _TRACE_ID.reset(token)


def _get_trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if isinstance(trace_id, str) and trace_id:
        return trace_id
    return str(uuid4())

# 允许前端本地开发跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/v1/rag/stats")
def rag_stats() -> Dict[str, Any]:
    """RAG 底座状态。

    注意：该接口不涉及会话上下文，不会输出用户原始会话文本。
    """

    from app.rag.rag_core import get_stats  # 延迟导入，避免启动时强制加载模型

    st = get_stats()
    return {
        "collection": st.collection,
        "count": st.count,
        "persist_dir": st.persist_dir,
        "device": st.device,
        "embed_model": st.embed_model,
        "rerank_model": st.rerank_model,
        "updated_at": st.updated_at,
    }


@app.post("/v1/rag/retrieve")
def rag_retrieve(
    request: Request,
    req: RagRetrieveRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    """独立 RAG 检索接口。

    日志安全：仅打印 query 前 100 字符或哈希。
    """

    trace_id = _get_trace_id(request)

    auth_resp = _auth_guard(x_api_key, trace_id)
    if auth_resp is not None:
        return auth_resp  # type: ignore[return-value]

    from app.rag.evidence_policy import summarize_evidence_quality
    from app.rag.rag_core import get_last_retrieval_meta, get_stats, retrieve

    q = (req.query or "").strip()
    logger.info("RAG_RETRIEVE trace_id=%s query=%s top_k=%s top_n=%s dept=%s", trace_id, _safe_query_for_log(q), req.top_k, req.top_n, (req.department or ""))

    evidence = retrieve(
        q,
        top_k=req.top_k,
        top_n=req.top_n,
        department=req.department,
        use_rerank=req.use_rerank,
    )
    retrieval_meta = get_last_retrieval_meta()
    st = get_stats()

    effective_use_rerank = req.use_rerank if req.use_rerank is not None else _env_flag("RAG_USE_RERANKER", "1")

    return {
        "query": q,
        "top_k": int(req.top_k),
        "top_n": int(req.top_n),
        "use_rerank": bool(effective_use_rerank),
        "evidence": evidence,
        "evidence_quality": summarize_evidence_quality(evidence),
        "retrieval_meta": retrieval_meta,
        "stats": {
            "collection": st.collection,
            "count": st.count,
            "device": st.device,
            "embed_model": st.embed_model,
            "rerank_model": st.rerank_model,
        },
    }


@app.post("/v1/ocr/ingest")
async def ocr_ingest(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    trace_id = _get_trace_id(request)

    auth_resp = _auth_guard(x_api_key, trace_id)
    if auth_resp is not None:
        return auth_resp  # type: ignore[return-value]

    payload = await _parse_ocr_ingest_payload(request)
    session_id = payload["session_id"] or str(uuid4())
    file_url = payload["file_url"]
    source = payload["source"]
    upload_name = payload["upload_name"]
    upload_type = payload["upload_type"] or None
    upload_bytes = payload["upload_bytes"]

    if bool(file_url) == bool(upload_bytes):
        raise HTTPException(status_code=400, detail="必须且只能提供 file_url 或 file")

    try:
        if file_url:
            task = _create_mineru_task_from_url(file_url)
            source_url = str(task.get("source_url") or file_url)
            source_name = source or os.path.basename(source_url) or source_url
            source_kind = "url"
        else:
            task = _create_mineru_task_from_upload(upload_name or "upload.bin", upload_type, upload_bytes)
            source_url = str(task.get("source_url") or upload_name or "upload.bin")
            source_name = source or upload_name or "upload.bin"
            source_kind = "upload"
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR任务创建失败: {type(e).__name__}")

    _OCR_STORE.upsert_ocr_task(
        task_id=str(task.get("task_id") or ""),
        session_id=session_id,
        source_url=source_url,
        source_name=source_name,
        source_kind=source_kind,
        status="pending",
        trace_id=str(task.get("trace_id") or ""),
    )

    return {
        "session_id": session_id,
        "task_id": str(task.get("task_id") or ""),
        "status": "pending",
        "trace_id": str(task.get("trace_id") or "") or None,
        "source_url": source_url,
        "source_kind": source_kind,
    }


@app.get("/v1/ocr/status/{task_id}")
def ocr_status(
    task_id: str,
    request: Request,
    session_id: Optional[str] = None,
    source_url: Optional[str] = None,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    trace_id = _get_trace_id(request)

    auth_resp = _auth_guard(x_api_key, trace_id)
    if auth_resp is not None:
        return auth_resp  # type: ignore[return-value]

    rec = _OCR_STORE.get_ocr_task(task_id) or {}
    sid = str(rec.get("session_id") or session_id or "").strip() or str(uuid4())
    src_url = str(rec.get("source_url") or source_url or "").strip() or "ocr"
    src_name = str(rec.get("source_name") or src_url or "ocr_upload").strip() or "ocr_upload"
    src_kind = str(rec.get("source_kind") or "url").strip() or "url"

    if bool(rec.get("ingested")):
        return {
            "task_id": task_id,
            "status": str(rec.get("status") or "done"),
            "done": True,
            "ingested": True,
            "session_id": sid,
            "trace_id": str(rec.get("trace_id") or "") or None,
            "picked": str(rec.get("picked") or "") or None,
        }

    try:
        st = _get_mineru_task_status(task_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MinerU 查询失败: {type(e).__name__}")

    status = str(st.get("status") or "").strip() or "pending"
    _OCR_STORE.upsert_ocr_task(
        task_id=task_id,
        session_id=sid,
        source_url=src_url,
        source_name=src_name,
        source_kind=src_kind,
        status=status,
        trace_id=str(st.get("trace_id") or rec.get("trace_id") or ""),
        ingested=bool(rec.get("ingested")),
        chunk_id=str(rec.get("chunk_id") or ""),
        picked=str(rec.get("picked") or ""),
        message=str(rec.get("message") or ""),
    )

    if not bool(st.get("done")):
        return {
            "task_id": task_id,
            "status": status,
            "done": False,
            "trace_id": str(st.get("trace_id") or rec.get("trace_id") or "") or None,
        }

    full_zip_url = str(st.get("full_zip_url") or "").strip()
    if not full_zip_url:
        message = "full_zip_url 为空"
        _OCR_STORE.upsert_ocr_task(
            task_id=task_id,
            session_id=sid,
            source_url=src_url,
            source_name=src_name,
            source_kind=src_kind,
            status=status,
            trace_id=str(st.get("trace_id") or rec.get("trace_id") or ""),
            ingested=False,
            chunk_id=str(rec.get("chunk_id") or ""),
            picked=str(rec.get("picked") or ""),
            message=message,
        )
        return {
            "task_id": task_id,
            "status": status,
            "done": True,
            "ingested": False,
            "trace_id": str(st.get("trace_id") or rec.get("trace_id") or "") or None,
            "message": message,
        }

    zip_bytes = _download_mineru_result_zip(full_zip_url)
    text, pick_meta = _extract_mineru_text_from_zip(zip_bytes)

    from app.rag.ingest_kb import _hard_gate, _sanitize_text

    clean = _sanitize_text(text)
    if not _hard_gate(clean):
        message = "OCR文本不符合入库要求"
        _OCR_STORE.upsert_ocr_task(
            task_id=task_id,
            session_id=sid,
            source_url=src_url,
            source_name=src_name,
            source_kind=src_kind,
            status=status,
            trace_id=str(st.get("trace_id") or rec.get("trace_id") or ""),
            ingested=False,
            chunk_id=str(rec.get("chunk_id") or ""),
            picked=str(pick_meta.get("picked") or ""),
            message=message,
        )
        return {
            "task_id": task_id,
            "status": status,
            "done": True,
            "ingested": False,
            "trace_id": str(st.get("trace_id") or rec.get("trace_id") or "") or None,
            "message": message,
        }

    chunk_id = str(rec.get("chunk_id") or "").strip() or f"ocr:{task_id}:{_sha256_text(clean)[:8]}"
    metadata = {
        "source_file": src_name[:120],
        "source": src_url[:240],
        "page": None,
        "section": "ocr",
        "department": "",
        "title": src_name[:120],
        "row": None,
        "domain": "ocr",
        "chunk_id": chunk_id,
        "source_kind": src_kind,
        "session_id": sid,
        "picked": pick_meta.get("picked"),
    }

    vs = _get_vectordb_for_ocr()
    try:
        vs.add_texts([clean], metadatas=[metadata])
    except Exception:
        from langchain.schema import Document  # type: ignore

        vs.add_documents([Document(page_content=clean, metadata=metadata)])
    if hasattr(vs, "persist"):
        try:
            vs.persist()
        except Exception:
            pass

    _OCR_STORE.upsert_ocr_task(
        task_id=task_id,
        session_id=sid,
        source_url=src_url,
        source_name=src_name,
        source_kind=src_kind,
        status=status,
        trace_id=str(st.get("trace_id") or rec.get("trace_id") or ""),
        ingested=True,
        chunk_id=chunk_id,
        picked=str(pick_meta.get("picked") or ""),
        message="",
    )

    return {
        "task_id": task_id,
        "status": status,
        "done": True,
        "ingested": True,
        "session_id": sid,
        "trace_id": str(st.get("trace_id") or rec.get("trace_id") or "") or None,
        "picked": str(pick_meta.get("picked") or "") or None,
    }


def _auth_guard(x_api_key: Optional[str], trace_id: str) -> Optional[JSONResponse]:
    """Minimal API-key auth guard for /v1/triage.

    - Reads TRIAGE_API_KEY from environment.
    - If set: requires request header X-API-Key to match.
    - If not set: allows requests but emits a warning log.

    Returns:
        - None if request is allowed.
        - JSONResponse(401, {code,message}) if unauthorized.
    """

    expected = (os.getenv("TRIAGE_API_KEY") or "").strip()
    if not expected:
        logger.warning("鉴权未启用：未设置环境变量 TRIAGE_API_KEY")
        return None

    provided = (x_api_key or "").strip()
    if not provided or provided != expected:
        # 这里不要泄漏更多信息（例如 expected）
        return JSONResponse(
            status_code=401,
            content={
                "code": "UNAUTHORIZED",
                "message": "Missing or invalid X-API-Key",
                "trace_id": trace_id,
            },
        )
    return None


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = _get_trace_id(request)
    logger.warning("BAD_REQUEST trace_id=%s validation_errors=%s", trace_id, exc.errors())
    return JSONResponse(
        status_code=400,
        content={
            "code": "BAD_REQUEST",
            "message": "请求参数不合法",
            "trace_id": trace_id,
        },
    )


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    trace_id = _get_trace_id(request)
    logger.warning("BAD_REQUEST trace_id=%s error=%s", trace_id, str(exc))
    return JSONResponse(
        status_code=400,
        content={
            "code": "BAD_REQUEST",
            "message": str(exc),
            "trace_id": trace_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    trace_id = _get_trace_id(request)

    code = "HTTP_ERROR"
    message = ""
    if isinstance(exc.detail, dict):
        code = str(exc.detail.get("code") or code)
        message = str(exc.detail.get("message") or "")
    elif exc.detail is not None:
        message = str(exc.detail)

    logger.warning("HTTPException trace_id=%s status=%s code=%s message=%s", trace_id, exc.status_code, code, message)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": code,
            "message": message,
            "trace_id": trace_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    trace_id = _get_trace_id(request)
    logger.exception("INTERNAL_ERROR trace_id=%s", trace_id)
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "服务内部错误",
            "trace_id": trace_id,
        },
    )


def _is_localhost(host: Optional[str]) -> bool:
    return (host or "") in {"127.0.0.1", "::1"}


def _apply_mode_policy(requested_mode: str, client_host: Optional[str]) -> tuple[str, bool]:
    """Apply API-layer policy for mode.

    Policy:
    - By default, do NOT allow mode=fast to bypass safety chain.
    - fast is allowed only when:
        - client host is localhost (127.0.0.1/::1)
        - and env var ALLOW_FAST_MODE == "1"
    - Otherwise, force safe.

    Returns:
        (final_mode, forced_safe)
    """

    mode = (requested_mode or "fast").strip().lower()
    if mode not in ("fast", "safe"):
        mode = "fast"

    if mode != "fast":
        return mode, False

    allow_fast = (os.getenv("ALLOW_FAST_MODE") or "").strip() == "1"
    if allow_fast and _is_localhost(client_host):
        return "fast", False

    logger.warning(
        "fast模式已被API层强制改写为safe：host=%s, ALLOW_FAST_MODE=%s",
        client_host,
        os.getenv("ALLOW_FAST_MODE"),
    )
    return "safe", True


@app.post("/v1/triage")
def triage(
    request: Request,
    req: TriageRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    trace_id = _get_trace_id(request)

    auth_resp = _auth_guard(x_api_key, trace_id)
    if auth_resp is not None:
        return auth_resp  # type: ignore[return-value]

    client_host = request.client.host if request.client else None

    final_mode, forced_safe = _apply_mode_policy(req.mode, client_host)

    out = triage_once(
        user_text=req.user_text,
        top_k=req.top_k,
        mode=final_mode,
        clinical_record_path=req.clinical_record_path,
    )

    # Keep canonical payload shape; add a small meta flag when policy forced safe.
    if forced_safe and isinstance(out, dict):
        meta = out.get("meta")
        if isinstance(meta, dict):
            meta["forced_safe"] = True

    return out


@app.post("/v1/chat")
def chat(
    request: Request,
    req: ChatRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    """Multi-turn patient chat with persistent sessions.

    Persistence:
      - Writes per-session JSON to OUTPUT_DIR/sessions/{session_id}.json
      - Default does NOT save raw patient/doctor text; stores {present,length,sha256} only.
      - Set ALLOW_SAVE_SESSION_RAW_TEXT=1 to store raw messages.
    """

    trace_id = _get_trace_id(request)

    auth_resp = _auth_guard(x_api_key, trace_id)
    if auth_resp is not None:
        return auth_resp  # type: ignore[return-value]

    session_id = (req.session_id or str(uuid4())).strip() or str(uuid4())
    session = _load_or_create_session(session_id)
    intake_slots = session.get("intake_slots")
    if not isinstance(intake_slots, dict):
        intake_slots = {}

    turns = session.get("turns")
    if not isinstance(turns, list):
        turns = []

    doctor_reply = ""
    triage_payload: Optional[Dict[str, Any]] = None
    handled_reply = False

    inquiry_state = session.get("inquiry_state")
    if not isinstance(inquiry_state, dict):
        inquiry_state = {}

    allow_save_raw = _env_flag("ALLOW_SAVE_SESSION_RAW_TEXT", "0")

    client_host = request.client.host if request.client else None
    final_mode, forced_safe = _apply_mode_policy(req.mode, client_host)

    max_turns = int((os.getenv("CHAT_MAX_TURNS") or "8").strip() or "8")
    graph = _get_chat_graph()
    gstate: _ChatGraphState = {
        "patient_message": req.patient_message,
        "top_k": req.top_k,
        "mode": final_mode,
        "forced_safe": forced_safe,
        "max_turns": max_turns,
        "turns_len": len(turns),
        "intake_slots": intake_slots,
        "inquiry_state": inquiry_state,
        "triage_trace": [],
    }
    result = graph.invoke(gstate)

    # Pull results back out to keep persistence and API response stable.
    if isinstance(result, dict):
        intake_slots = result.get("intake_slots") if isinstance(result.get("intake_slots"), dict) else intake_slots
        inquiry_state = result.get("inquiry_state") if isinstance(result.get("inquiry_state"), dict) else inquiry_state
        session["inquiry_state"] = inquiry_state

        act = str(result.get("act") or "INQUIRY")
        doctor_reply = str(result.get("doctor_reply") or "")
        triage_payload = result.get("triage_payload") if isinstance(result.get("triage_payload"), dict) else None
        handled_reply = bool(result.get("handled_reply") or False)
        debug_trace = result.get("triage_trace") if isinstance(result.get("triage_trace"), list) else []
    else:
        # Extremely defensive fallback.
        act = "INQUIRY"
        doctor_reply = "我还需要更多信息才能给出初步建议。"
        triage_payload = None
        handled_reply = True
        debug_trace = []

    # Append turn
    turn_record: Dict[str, Any] = {
        "at": _utc_now(),
        "trace_id": trace_id,
        "act": act,
        "patient_message_meta": _text_meta(req.patient_message),
        "doctor_reply_meta": _text_meta(doctor_reply),
        "intake_slots_snapshot": intake_slots,
    }
    if allow_save_raw:
        turn_record["patient_message"] = req.patient_message
        turn_record["doctor_reply"] = doctor_reply
    if isinstance(triage_payload, dict):
        # Store canonical triage payload (already avoids raw inputs).
        turn_record["triage"] = triage_payload

    turns.append(turn_record)
    session["turns"] = turns
    session["intake_slots"] = intake_slots

    _save_session(session)

    return {
        "session_id": session_id,
        "act": act,
        "doctor_reply": doctor_reply,
        "intake_slots": intake_slots,
        "triage": triage_payload,
        "trace_id": trace_id,
        "debug": {"trace": debug_trace},
    }
