# -*- coding: utf-8 -*-
"""rag_core.py

M1：RAGService 核心实现（进程内直连版本）。

目标：
- 统一入库与检索的 embedding 配置（默认 BCEmbedding bce-embedding-base_v1，GPU 优先）。
- 两阶段检索：Chroma 向量召回 top_n -> 可选 BCEmbedding reranker 重排 -> 返回 top_k。
- 固定 evidence 契约：retrieve() 返回 List[Dict] 且字段齐全，可直接被 triage_service.py 使用。

说明：
- 该模块不依赖会话上下文，不会保存或输出用户历史文本。
- 日志仅允许输出 query 的前 100 字符（或哈希），避免泄漏。
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.rag.utils.rag_shared import (
    DEFAULT_COLLECTION,
    EmbeddingInfo,
    apply_windows_openmp_workaround,
    env_flag,
    env_int,
    env_str,
    make_embeddings,
    resolve_app_dir,
    resolve_persist_dir,
    resolve_embedding_device,
)


@dataclass(frozen=True)
class RagStats:
    collection: str
    count: int
    persist_dir: str
    device: str
    embed_model: str
    rerank_model: Optional[str]
    updated_at: str


_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()
_vs_lock = threading.Lock()

_cached_embed: Optional[Tuple[Any, EmbeddingInfo]] = None
_cached_reranker: Optional[Any] = None
_cached_vs: Optional[Any] = None


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


def _env_provider() -> str:
    # 新变量优先，兼容旧变量
    v = (env_str("RAG_PROVIDER", "") or env_str("RAG_EMBEDDINGS_PROVIDER", "") or "bce").strip().lower()
    if v in {"bce", "bcembedding"}:
        return "bce"
    if v in {"hf", "huggingface"}:
        return "hf"
    return v


def _env_embed_model_default() -> str:
    # 由 rag_shared.DEFAULT_BCE_MODEL 提供默认值；此处只负责读取新变量。
    return env_str("RAG_EMBED_MODEL", "").strip()


def _env_rerank_model() -> str:
    return (env_str("RAG_RERANK_MODEL", "") or "maidalun1020/bce-reranker-base_v1").strip()


def _env_use_reranker() -> bool:
    # 入参 use_rerank 允许覆盖；否则按环境变量默认开启
    return env_flag("RAG_USE_RERANKER", default="1")


def _env_top_n_default() -> int:
    n = env_int("RAG_TOP_N", default=30)
    if not n:
        return 30
    return max(1, int(n))


def get_embedder() -> Tuple[Any, EmbeddingInfo]:
    """获取 embedding（单例缓存），确保与 ingest 使用同一套逻辑。"""
    global _cached_embed
    if _cached_embed is not None:
        return _cached_embed

    with _embed_lock:
        if _cached_embed is not None:
            return _cached_embed

        apply_windows_openmp_workaround()

        # make_embeddings 内部已处理 provider/model/device 的新旧变量兼容
        emb, info = make_embeddings()
        _cached_embed = (emb, info)
        return _cached_embed


class _CrossEncoderWrapper:
    """Wrapper to make CrossEncoder compatible with BCEmbedding RerankerModel API."""

    def __init__(self, model_name: str, device: str):
        from sentence_transformers import CrossEncoder  # type: ignore

        self._model = CrossEncoder(model_name, device=device)
        self._device = device
        print(f"[RAG_RERANK] INFO model={model_name} device={device}", flush=True)

    def compute_score(self, pairs, enable_tqdm: bool = False):
        """Compatible with BCEmbedding RerankerModel.compute_score()."""
        return self._model.predict(pairs, show_progress_bar=enable_tqdm)


def _try_load_reranker(model_name: str, device: str):
    """尝试加载 reranker 模型，返回 (model, actual_device)。使用 CrossEncoder 支持 MPS。"""
    return _CrossEncoderWrapper(model_name, device), device


def get_reranker() -> Optional[Any]:
    """获取 reranker（单例缓存）。

    说明：
    - 默认启用（RAG_USE_RERANKER=1）。
    - 如果明确关闭（RAG_USE_RERANKER=0），返回 None。
    - 如果启用但模型/依赖不可用，抛出可读错误信息。
    - 支持 CUDA、MPS、CPU 设备，MPS 不可用时自动降级到 CPU。
    - 使用 sentence-transformers CrossEncoder 代替 BCEmbedding，以支持 MPS。
    """
    global _cached_reranker

    if not _env_use_reranker():
        return None

    if _cached_reranker is not None:
        return _cached_reranker

    with _rerank_lock:
        if _cached_reranker is not None:
            return _cached_reranker

        apply_windows_openmp_workaround()

        model_name = _env_rerank_model()
        device = resolve_embedding_device()

        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "已启用 RAG_USE_RERANKER=1，但无法导入 sentence-transformers CrossEncoder。\n"
                "解决：确认已安装 sentence-transformers，或设置 RAG_USE_RERANKER=0 关闭重排。\n"
                f"导入错误：{type(e).__name__}: {e}"
            ) from e

        try:
            _cached_reranker, actual_device = _try_load_reranker(model_name, device)
            if actual_device != device:
                print(
                    f"[RAG_RERANK] INFO device_fallback from={device} to={actual_device}",
                    flush=True,
                )
        except Exception as e:
            # MPS 可能不完全支持某些操作，尝试降级到 CPU
            if device == "mps":
                print(
                    f"[RAG_RERANK] WARN mps_fallback reason={type(e).__name__}: {e}",
                    flush=True,
                )
                try:
                    _cached_reranker, _ = _try_load_reranker(model_name, "cpu")
                    print("[RAG_RERANK] INFO fallback_to_cpu success", flush=True)
                except Exception as e2:
                    raise RuntimeError(
                        "Reranker 模型加载失败（MPS 和 CPU 均失败）。\n"
                        f"model={model_name}\n"
                        f"MPS 错误：{type(e).__name__}: {e}\n"
                        f"CPU 错误：{type(e2).__name__}: {e2}"
                    ) from e2
            else:
                raise RuntimeError(
                    "Reranker 模型加载失败。\n"
                    f"model={model_name} device={device}\n"
                    f"错误：{type(e).__name__}: {e}"
                ) from e

        return _cached_reranker


def get_vectordb() -> Any:
    """获取 Chroma 向量库对象（单例缓存）。"""
    global _cached_vs
    if _cached_vs is not None:
        return _cached_vs

    with _vs_lock:
        if _cached_vs is not None:
            return _cached_vs

        app_dir = resolve_app_dir(Path(__file__))
        persist_dir = resolve_persist_dir(app_dir)
        if not persist_dir.exists():
            raise FileNotFoundError(
                f"找不到 Chroma 持久化目录：{persist_dir}。请先运行 app/rag/ingest_kb.py 完成入库。"
            )

        collection_name = env_str("RAG_COLLECTION", "") or DEFAULT_COLLECTION
        embeddings, _ = get_embedder()

        try:
            from langchain.vectorstores import Chroma  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "缺少 langchain 依赖，无法构建 Chroma 向量库。\n"
                "解决：安装 requirements.txt 中的依赖（langchain、chromadb）。"
            ) from e

        _cached_vs = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        return _cached_vs


def _vector_search(
    query: str,
    *,
    top_n: int,
    department: Optional[str] = None,
) -> List[Tuple[Any, float]]:
    """第一阶段向量检索：返回 (doc, score) 列表。

    department 过滤：优先尝试使用 Chroma filter；若不支持则降级为客户端过滤。
    """
    vs = get_vectordb()

    k = max(1, int(top_n))
    dept = (department or "").strip()

    # 优先尝试原生 filter
    if dept:
        try:
            return vs.similarity_search_with_score(query, k=k, filter={"department": dept})
        except TypeError:
            pass
        except Exception:
            # 某些版本在 filter 参数上会抛运行时错误，降级处理。
            pass

    docs_scores: List[Tuple[Any, float]] = vs.similarity_search_with_score(query, k=k)

    if not dept:
        return docs_scores

    # 客户端过滤：尽量补足到 k
    filtered: List[Tuple[object, float]] = []
    for doc, score in docs_scores:
        md = getattr(doc, "metadata", None) or {}
        if str(md.get("department") or "").strip() == dept:
            filtered.append((doc, score))

    if len(filtered) >= k:
        return filtered[:k]

    # 可能需要更大范围召回再过滤补足
    try:
        docs_scores2 = vs.similarity_search_with_score(query, k=min(max(k * 4, 50), 200))
    except Exception:
        return filtered

    for doc, score in docs_scores2:
        md = getattr(doc, "metadata", None) or {}
        if str(md.get("department") or "").strip() == dept:
            filtered.append((doc, score))
        if len(filtered) >= k:
            break

    return filtered[:k]


def _normalize_evidence_item(
    *,
    idx: int,
    doc: object,
    score: float,
) -> Dict[str, Any]:
    page_content = (getattr(doc, "page_content", None) or "").strip()
    md0 = dict(getattr(doc, "metadata", None) or {})

    source_file = (md0.get("source_file") or md0.get("source") or md0.get("source_url") or "").strip()
    if not source_file:
        source_file = "unknown"

    chunk_id = (md0.get("chunk_id") or "").strip()
    if not chunk_id:
        row = md0.get("row", None)
        chunk_id = f"{source_file}:{row}:{idx}"

    department = md0.get("department", "")
    title = md0.get("title", "")
    row = md0.get("row", None)

    row_out: Optional[int]
    try:
        row_out = int(row) if row is not None else None
    except Exception:
        row_out = None

    return {
        "eid": f"E{idx}",
        "text": page_content,
        "source": source_file,
        "chunk_id": chunk_id,
        "score": float(score) if score is not None else None,
        "rerank_score": None,
        "metadata": {
            "department": (str(department).strip() if department is not None else ""),
            "title": (str(title).strip() if title is not None else ""),
            "row": row_out,
            "source_file": source_file,
        },
    }


def _apply_rerank(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """第二阶段重排：为每条 evidence 填充 rerank_score 并按其降序排序。"""
    reranker = get_reranker()
    if reranker is None:
        return items

    passages = [str(it.get("text") or "") for it in items]
    # compute_score 接收 (query, passage) 对
    pairs = [(query, p) for p in passages]

    try:
        scores = reranker.compute_score(pairs, enable_tqdm=False)
    except TypeError:
        # 兼容旧签名
        scores = reranker.compute_score(pairs)
    except Exception as e:
        raise RuntimeError(
            "Reranker 计算失败。\n"
            f"query={_safe_query_for_log(query)}\n"
            f"错误：{type(e).__name__}: {e}"
        ) from e

    # scores 可能是 list/ndarray/tensor
    try:
        scores_list = list(scores)
    except Exception:
        scores_list = []

    for it, s in zip(items, scores_list):
        try:
            it["rerank_score"] = float(s)
        except Exception:
            it["rerank_score"] = None

    items.sort(key=lambda x: (x.get("rerank_score") is not None, x.get("rerank_score") or float("-inf")), reverse=True)
    return items


def retrieve(
    query: str,
    top_k: int = 5,
    *,
    top_n: Optional[int] = None,
    department: Optional[str] = None,
    use_rerank: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """对外统一检索入口。

    兼容性：
    - 保持 `retrieve(query, top_k=5)` 可用，以兼容 triage_service.py 的现有调用。
    """
    q = (query or "").strip()
    if not q:
        return []

    k = int(top_k) if top_k and int(top_k) > 0 else 5
    n = int(top_n) if top_n is not None else _env_top_n_default()
    if n < k:
        n = k

    docs_scores = _vector_search(q, top_n=n, department=department)

    items: List[Dict[str, Any]] = []
    for i, (doc, score) in enumerate(docs_scores, start=1):
        it = _normalize_evidence_item(idx=i, doc=doc, score=float(score) if score is not None else score)
        # text 不能为空（契约要求）
        if not str(it.get("text") or "").strip():
            continue
        items.append(it)

    # 是否启用 rerank：入参优先，其次环境变量
    do_rerank = _env_use_reranker() if use_rerank is None else bool(use_rerank)
    if do_rerank and items:
        items = _apply_rerank(q, items)

    # 截断 top_k，并重建 eid 连续
    items = items[:k]
    for i, it in enumerate(items, start=1):
        it["eid"] = f"E{i}"

    return items


def get_stats() -> RagStats:
    """返回当前 RAG 底座状态。"""
    vs = get_vectordb()

    app_dir = resolve_app_dir(Path(__file__))
    persist_dir = resolve_persist_dir(app_dir)
    collection_name = env_str("RAG_COLLECTION", "") or DEFAULT_COLLECTION

    try:
        count = int(vs._collection.count())  # type: ignore[attr-defined]
    except Exception:
        count = 0

    _, emb_info = get_embedder()

    rerank_model = _env_rerank_model() if _env_use_reranker() else None

    # updated_at：优先 ingest_progress.json，其次 sqlite
    updated_at = ""
    try:
        candidates = [
            persist_dir / "ingest_progress.json",
            persist_dir / "chroma.sqlite3",
        ]
        ts = 0.0
        for p in candidates:
            if p.exists():
                ts = max(ts, p.stat().st_mtime)
        if ts > 0:
            updated_at = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        updated_at = ""

    return RagStats(
        collection=collection_name,
        count=count,
        persist_dir=str(persist_dir),
        device=str(emb_info.device or resolve_embedding_device()),
        embed_model=str(emb_info.model_name),
        rerank_model=rerank_model,
        updated_at=updated_at,
    )
