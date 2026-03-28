# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Any, Dict, List


# 单测默认关闭 reranker：更快、更稳定（仍覆盖 evidence 契约与 top_k 行为）。
os.environ.setdefault("RAG_USE_RERANKER", "0")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def test_rag_retrieve_contract_basic() -> None:
    """验证 evidence 契约字段齐全、eid 连续、text 非空、top_k 生效。

    说明：该测试依赖本地已完成入库的 Faiss-HNSW 持久化目录（默认 app/rag/kb_store）。
    """

    from app.rag.rag_core import retrieve

    top_k = 5
    evidence: List[Dict[str, Any]] = retrieve("咳嗽发热怎么办", top_k=top_k)

    assert isinstance(evidence, list)
    assert len(evidence) <= top_k

    required_fields = {
        "eid",
        "text",
        "source",
        "score",
        "rerank_score",
        "metadata",
        "chunk_id",
    }

    for idx, e in enumerate(evidence, start=1):
        assert isinstance(e, dict)
        missing = required_fields.difference(e.keys())
        assert not missing, f"missing fields: {missing}"

        assert e["eid"] == f"E{idx}"

        text = str(e.get("text") or "").strip()
        assert text

        assert isinstance(e.get("source"), str)
        assert e["source"].strip()

        assert _is_number(e.get("score"))
        # rerank_score 允许 None（例如关闭 reranker）或数值
        assert e.get("rerank_score") is None or _is_number(e.get("rerank_score"))

        assert isinstance(e.get("metadata"), dict)
        md = e.get("metadata") or {}
        for k in ["department", "title", "row", "source_file"]:
            assert k in md
        assert isinstance(e.get("chunk_id"), str)
        assert e["chunk_id"].strip()


def test_rag_retrieve_top_k_effective() -> None:
    """同一 query 下，top_k=1 应该返回 <=1 条。"""

    from app.rag.rag_core import retrieve

    evidence = retrieve("咳嗽发热怎么办", top_k=1)
    assert isinstance(evidence, list)
    assert len(evidence) <= 1
