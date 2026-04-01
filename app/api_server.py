# -*- coding: utf-8 -*-
"""FastAPI app boundary aligned to README public API whitelist."""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse


logger = logging.getLogger(__name__)

_TRACE_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)


class _TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _TRACE_ID.get() or "-"
        return True


logger.addFilter(_TraceIdFilter())


def _get_cors_allow_origins() -> list[str]:
    raw = (os.getenv("TRIAGE_CORS_ORIGINS") or "").strip()
    if raw:
        origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for origin in origins:
            if origin in seen:
                continue
            seen.add(origin)
            deduped.append(origin)
        return deduped

    return ["http://localhost:5173", "http://127.0.0.1:5173"]


app = FastAPI(
    title="MedChat Backend API",
    version="0.1.0",
)


try:
    from app.agent.router import router as agent_router  # type: ignore

    app.include_router(agent_router)
except Exception as exc:  # pragma: no cover - only relevant during broken imports
    logger.warning("agent router not loaded: %s", exc)


@app.middleware("http")
async def add_trace_id_middleware(request: Request, call_next):
    trace_id = str(uuid4())
    request.state.trace_id = trace_id
    token = _TRACE_ID.set(trace_id)
    try:
        logger.info("request start trace_id=%s method=%s path=%s", trace_id, request.method, request.url.path)
        return await call_next(request)
    finally:
        _TRACE_ID.reset(token)


def _get_trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if isinstance(trace_id, str) and trace_id:
        return trace_id
    return str(uuid4())


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


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
