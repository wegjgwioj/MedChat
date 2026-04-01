# -*- coding: utf-8 -*-
"""Thin retrieval facade over the README-aligned OpenSearch hybrid core."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.rag.rag_core import get_last_retrieval_meta as _get_last_retrieval_meta
from app.rag.rag_core import retrieve as _retrieve_core


def retrieve(
    query: str,
    top_k: int = 5,
    *,
    top_n: Optional[int] = None,
    department: Optional[str] = None,
    use_rerank: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """兼容入口：

    - 保持 `retrieve(query, top_k=5)` 仍可用，以兼容当前 agent 主链路。
    - 额外支持 M2 需要的参数：top_n/department/use_rerank。
    """

    return _retrieve_core(
        query,
        top_k=top_k,
        top_n=top_n,
        department=department,
        use_rerank=use_rerank,
    )


def get_last_retrieval_meta() -> Dict[str, Any]:
    return _get_last_retrieval_meta()
