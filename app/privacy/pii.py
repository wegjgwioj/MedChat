# -*- coding: utf-8 -*-

from __future__ import annotations

import re


_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_IDCARD_RE = re.compile(r"(?<![\dA-Za-z])(?:\d{17}[\dXx]|\d{15})(?![\dA-Za-z])")
_ADDRESS_RE = re.compile(r"((?:住址|地址|现住址?|家住|住在)\s*[:：]?\s*)([^，。；;\n]{4,80})")
_CONTACT_NAME_RE = re.compile(r"((?:我叫|姓名|联系人|患者姓名)\s*[:：]?\s*)([\u4e00-\u9fffA-Za-z·]{2,20})")


def redact_pii_for_llm(text: str) -> str:
    """Redact common personal identifiers before sending text to LLMs."""

    s = str(text or "")
    if not s:
        return s

    s = _CONTACT_NAME_RE.sub(r"\1<NAME>", s)
    s = _PHONE_RE.sub("<PHONE>", s)
    s = _IDCARD_RE.sub("<IDCARD>", s)
    s = _ADDRESS_RE.sub(r"\1<ADDRESS>", s)
    return s
