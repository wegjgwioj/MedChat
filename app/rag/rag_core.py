# -*- coding: utf-8 -*-
"""rag_core.py

M1：RAGService 核心实现（Faiss-HNSW 版本）。

目标：
- 统一入库与检索的 embedding 配置（默认 BCEmbedding bce-embedding-base_v1，GPU 优先）。
- 两阶段检索：Faiss-HNSW dense 召回 top_n -> 可选 BCEmbedding reranker 重排 -> 返回 top_k。
- 固定 evidence 契约：retrieve() 返回 List[Dict] 且字段齐全，可直接被 triage_service.py 使用。

说明：
- 该模块不依赖会话上下文，不会保存或输出用户历史文本。
- 日志仅允许输出 query 的前 100 字符（或哈希），避免泄漏。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.rag.cache_redis import RedisSemanticCache
from app.rag.faiss_store import FaissHNSWStore

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
from app.privacy import redact_pii_for_llm


@dataclass(frozen=True)
class RagStats:
    backend: str
    collection: str
    count: int
    persist_dir: str
    device: str
    embed_model: str
    rerank_model: Optional[str]
    updated_at: str


@dataclass(frozen=True)
class _DocLike:
    page_content: str
    metadata: Dict[str, Any]


_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()
_vs_lock = threading.Lock()
_cache_lock = threading.Lock()
_runtime_lock = threading.Lock()

_cached_embed: Optional[Tuple[Any, EmbeddingInfo]] = None
_cached_reranker: Optional[Any] = None
_cached_vs: Optional[Any] = None
_cache_backend: Optional[RedisSemanticCache] = None
_last_retrieval_meta: Dict[str, Any] = {}


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


def _rewrite_query_slimming(query: str) -> str:
    """
    利用 DeepSeek 大模型将患者的碎碎念转化为医学核心关键词。
    """
    try:
        import os
        import openai  # openai==0.28.x

        openai.api_key = os.getenv("DEEPSEEK_API_KEY")
        openai.api_base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        # 2. 读取prompt模板
        from app.agent.prompts import QUERY_SLIMMING_SYSTEM, QUERY_SLIMMING_USER_TEMPLATE

        # 3. 构造用户消息
        user_message = QUERY_SLIMMING_USER_TEMPLATE.format(user_message=redact_pii_for_llm(query))

        # 4. 调用API
        response = openai.ChatCompletion.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": QUERY_SLIMMING_SYSTEM},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=10
        )

        first_choice = response.choices[0]
        message = getattr(first_choice, "message", None)
        if isinstance(message, dict):
            slimming_q = str(message.get("content") or "").strip()
        else:
            slimming_q = str(getattr(message, "content", "") or "").strip()

        # 5. 演示日志
        print("\n" + "—" * 40)
        print("【M1 检索优化：DeepSeek 语义提纯】")
        print(f"原始提问: {query[:50]}...")
        print(f"瘦身结果: {slimming_q}")
        print("—" * 40 + "\n")

        return slimming_q if slimming_q else query

    except Exception as e:
        print(f"⚠️  瘦身功能异常 ({type(e).__name__}: {str(e)[:100]})，已切换至原始检索。")
        return query


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


def _env_float(name: str, default: float = 0.0) -> float:
    raw = env_str(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_rerank_min_score() -> float:
    return max(0.0, _env_float("RAG_RERANK_MIN_SCORE", 0.0))


def _env_vector_max_score() -> Optional[float]:
    raw = env_str("RAG_VECTOR_MAX_SCORE", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if value >= 0 else None


def _env_hybrid_enabled() -> bool:
    return env_flag("RAG_HYBRID_ENABLED", default="1")


def _env_hybrid_alpha() -> float:
    value = _env_float("RAG_HYBRID_ALPHA", 0.60)
    return min(1.0, max(0.0, value))


def _env_hybrid_token_min_len() -> int:
    value = env_int("RAG_HYBRID_TOKEN_MIN_LEN", default=2)
    if not value:
        return 2
    return max(1, int(value))


def _env_cache_enabled() -> bool:
    return env_flag("RAG_CACHE_ENABLED", default="0")


def _env_cache_ttl_seconds() -> int:
    value = env_int("RAG_CACHE_TTL_SECONDS", default=300)
    if not value:
        return 300
    return max(1, int(value))


def _env_cache_max_entries() -> int:
    value = env_int("RAG_CACHE_MAX_ENTRIES", default=128)
    if not value:
        return 128
    return max(1, int(value))


def _env_cache_sim_threshold() -> float:
    value = _env_float("RAG_CACHE_SIM_THRESHOLD", 0.85)
    return min(1.0, max(0.0, value))


def clear_runtime_state() -> None:
    """Reset module runtime caches for tests and local debugging."""

    global _cached_embed, _cached_reranker, _cached_vs
    global _cache_backend, _last_retrieval_meta

    with _embed_lock:
        _cached_embed = None
    with _rerank_lock:
        _cached_reranker = None
    with _vs_lock:
        _cached_vs = None
    with _cache_lock:
        _cache_backend = None
    with _runtime_lock:
        _last_retrieval_meta = {}


def get_last_retrieval_meta() -> Dict[str, Any]:
    with _runtime_lock:
        return dict(_last_retrieval_meta)


def _set_last_retrieval_meta(meta: Dict[str, Any]) -> None:
    global _last_retrieval_meta
    with _runtime_lock:
        _last_retrieval_meta = dict(meta)


def _normalize_query_for_cache(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())


def _tokenize_text(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", str(text or ""))
    if not raw:
        return []

    token_min_len = _env_hybrid_token_min_len()
    tokens: List[str] = []
    for token in raw:
        if re.fullmatch(r"[\u4e00-\u9fff]", token):
            tokens.append(token)
            continue
        t = token.lower()
        if len(t) >= token_min_len:
            tokens.append(t)
    return tokens


def _normalize_dense_relevance(items: List[Dict[str, Any]]) -> List[float]:
    scores: List[float] = []
    for item in items:
        score = item.get("hybrid_dense_score", item.get("score"))
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            scores.append(float(score))
        else:
            scores.append(1.0)

    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(max_score, min_score):
        return [1.0 for _ in scores]

    span = max_score - min_score
    return [1.0 - ((score - min_score) / span) for score in scores]


def _normalize_positive_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]
    if math.isclose(max_score, min_score):
        return [1.0 for _ in scores]
    span = max_score - min_score
    return [(score - min_score) / span for score in scores]


def _compute_sparse_scores(query: str, items: List[Dict[str, Any]]) -> List[float]:
    query_tokens = _tokenize_text(query)
    if not query_tokens or not items:
        return [0.0 for _ in items]

    docs_tokens = [_tokenize_text(str(item.get("text") or "")) for item in items]
    if not any(docs_tokens):
        return [0.0 for _ in items]

    query_set = set(query_tokens)
    doc_freq: Dict[str, int] = {}
    for tokens in docs_tokens:
        for token in set(tokens):
            if token in query_set:
                doc_freq[token] = doc_freq.get(token, 0) + 1

    avgdl = sum(len(tokens) for tokens in docs_tokens) / max(len(docs_tokens), 1)
    if avgdl <= 0:
        avgdl = 1.0

    k1 = 1.2
    b = 0.75
    n_docs = len(docs_tokens)
    scores: List[float] = []
    for tokens in docs_tokens:
        tf: Dict[str, int] = {}
        for token in tokens:
            if token in query_set:
                tf[token] = tf.get(token, 0) + 1

        doc_len = max(len(tokens), 1)
        score = 0.0
        for token in query_set:
            freq = tf.get(token, 0)
            if freq <= 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
            denom = freq + k1 * (1.0 - b + b * (doc_len / avgdl))
            score += idf * ((freq * (k1 + 1.0)) / denom)
        scores.append(score)

    return scores


def _load_sparse_docs(*, department: Optional[str] = None) -> List[_DocLike]:
    vs = get_vectordb()
    dept = (department or "").strip()
    where = {"department": dept} if dept else None
    result = vs.get_documents(where=where)
    if not isinstance(result, dict):
        raise RuntimeError("Faiss store.get_documents 返回格式非法。")

    documents = list(result.get("documents") or [])
    metadatas = list(result.get("metadatas") or [])

    out: List[_DocLike] = []
    for idx, raw_text in enumerate(documents):
        text = str(raw_text or "").strip()
        if not text:
            continue
        raw_meta = metadatas[idx] if idx < len(metadatas) else {}
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        if dept and str(meta.get("department") or "").strip() != dept:
            continue
        out.append(_DocLike(page_content=text, metadata=meta))
    return out


def _sparse_search(
    query: str,
    *,
    top_n: int,
    department: Optional[str] = None,
) -> List[Tuple[Any, float]]:
    docs = _load_sparse_docs(department=department)
    if not docs:
        return []

    items = [
        {
            "text": str(doc.page_content or "").strip(),
            "score": 0.0,
        }
        for doc in docs
    ]
    sparse_scores = _compute_sparse_scores(query, items)
    ranked: List[Tuple[float, int, _DocLike]] = []
    for idx, (doc, score) in enumerate(zip(docs, sparse_scores)):
        if float(score) <= 0:
            continue
        ranked.append((float(score), -idx, doc))

    ranked.sort(reverse=True)
    limit = max(1, int(top_n))
    return [(doc, score) for score, _, doc in ranked[:limit]]


def _apply_hybrid_ranking(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not _env_hybrid_enabled() or len(items) <= 1:
        return items

    sparse_scores = _compute_sparse_scores(query, items)
    if not any(score > 0 for score in sparse_scores):
        return items

    dense_scores = _normalize_dense_relevance(items)
    sparse_relevance = _normalize_positive_scores(sparse_scores)
    alpha = _env_hybrid_alpha()

    ranked: List[Tuple[float, float, float, int, Dict[str, Any]]] = []
    for idx, item in enumerate(items):
        dense_value = dense_scores[idx] if idx < len(dense_scores) else 0.0
        sparse_value = sparse_relevance[idx] if idx < len(sparse_relevance) else 0.0
        hybrid_value = alpha * dense_value + (1.0 - alpha) * sparse_value
        ranked.append((hybrid_value, sparse_value, -float(item.get("score") or 0.0), -idx, item))

    ranked.sort(reverse=True)
    return [item for _, _, _, _, item in ranked]


def _merge_hybrid_candidates(
    *,
    dense_items: List[Dict[str, Any]],
    sparse_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for item in dense_items + sparse_items:
        chunk_id = str(item.get("chunk_id") or "").strip()
        source = str(item.get("source") or "").strip()
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_file = str(metadata.get("source_file") or source).strip()
        row = metadata.get("row")
        row_key = f"{source_file}:{row}" if source_file and row is not None else ""
        key = row_key or chunk_id or f"{source}|{str(item.get('text') or '').strip()}"
        if key not in merged:
            merged[key] = dict(item)
            order.append(key)
            continue

        existing = merged[key]
        incoming_is_sparse_only = (
            isinstance(item.get("hybrid_dense_score"), (int, float))
            and float(item.get("hybrid_dense_score") or 0.0) >= 1.0
            and float(item.get("score") or 0.0) == 0.0
        )
        existing_has_real_dense = not (
            isinstance(existing.get("hybrid_dense_score"), (int, float))
            and float(existing.get("hybrid_dense_score") or 0.0) >= 1.0
            and float(existing.get("score") or 0.0) == 0.0
        )
        if not incoming_is_sparse_only or not existing_has_real_dense:
            if not isinstance(existing.get("score"), (int, float)) or float(item.get("score") or 0.0) < float(existing.get("score") or 0.0):
                existing["score"] = float(item.get("score") or 0.0)

        existing_text = str(existing.get("text") or "").strip()
        new_text = str(item.get("text") or "").strip()
        if len(new_text) > len(existing_text):
            existing["text"] = new_text

        existing_meta = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
        new_meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if isinstance(existing_meta, dict) and isinstance(new_meta, dict):
            merged_meta = dict(existing_meta)
            for mk, mv in new_meta.items():
                if mk not in merged_meta or merged_meta[mk] in {"", None}:
                    merged_meta[mk] = mv
            existing["metadata"] = merged_meta

    return [merged[key] for key in order]


def _build_cache_request_key(
    *,
    top_k: int,
    top_n: int,
    department: Optional[str],
    use_rerank: bool,
    hybrid_enabled: bool,
) -> str:
    dept = (department or "").strip().lower()
    vector_max_score = _env_vector_max_score()
    return "|".join(
        [
            "v2",
            f"top_k={top_k}",
            f"top_n={top_n}",
            f"dept={dept}",
            f"use_rerank={int(use_rerank)}",
            f"hybrid={int(hybrid_enabled)}",
            f"hybrid_alpha={_env_hybrid_alpha():.3f}",
            f"rerank_min={_env_rerank_min_score():.3f}",
            f"vector_max={'' if vector_max_score is None else f'{vector_max_score:.3f}'}",
        ]
    )


def _resolve_cache_redis_url() -> str:
    return (env_str("RAG_REDIS_URL", "").strip() or str(os.getenv("AGENT_REDIS_URL") or "").strip())


def _get_cache_backend() -> RedisSemanticCache:
    global _cache_backend

    with _cache_lock:
        if _cache_backend is not None:
            return _cache_backend

        redis_url = _resolve_cache_redis_url()
        if not redis_url:
            raise RuntimeError("已启用 RAG Redis 语义缓存，但未配置 RAG_REDIS_URL 或 AGENT_REDIS_URL。")

        key_prefix = (env_str("RAG_CACHE_KEY_PREFIX", "").strip() or "medchat:rag-cache:")
        _cache_backend = RedisSemanticCache(redis_url=redis_url, key_prefix=key_prefix)
        return _cache_backend


def _lookup_cache(
    *,
    normalized_query: str,
    query_tokens: Tuple[str, ...],
    request_key: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    if not _env_cache_enabled():
        return None, {"cache_hit": False, "cache_mode": None, "cache_backend": None}

    try:
        cache = _get_cache_backend()
    except Exception:
        return None, {"cache_hit": False, "cache_mode": None, "cache_backend": None}
    return cache.lookup(
        normalized_query=normalized_query,
        query_tokens=query_tokens,
        request_key=request_key,
        sim_threshold=_env_cache_sim_threshold(),
        max_entries=_env_cache_max_entries(),
    )


def _store_cache(
    *,
    normalized_query: str,
    query_tokens: Tuple[str, ...],
    request_key: str,
    items: List[Dict[str, Any]],
) -> None:
    if not _env_cache_enabled():
        return

    try:
        cache = _get_cache_backend()
    except Exception:
        return
    cache.store(
        normalized_query=normalized_query,
        query_tokens=query_tokens,
        request_key=request_key,
        items=items,
        ttl_seconds=float(_env_cache_ttl_seconds()),
        max_entries=_env_cache_max_entries(),
    )


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


def get_vectordb() -> FaissHNSWStore:
    """获取 Faiss-HNSW 向量库对象（单例缓存）。"""
    global _cached_vs
    if _cached_vs is not None:
        return _cached_vs

    with _vs_lock:
        if _cached_vs is not None:
            return _cached_vs

        app_dir = resolve_app_dir(Path(__file__))
        persist_dir = resolve_persist_dir(app_dir)
        collection_name = env_str("RAG_COLLECTION", "") or DEFAULT_COLLECTION
        embeddings, _ = get_embedder()
        _cached_vs = FaissHNSWStore(
            persist_dir=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        return _cached_vs


def _vector_search(
    query: str,
    *,
    top_n: int,
    department: Optional[str] = None,
) -> List[Tuple[Any, float]]:
    """第一阶段向量检索：返回 (doc, score) 列表。

    department 过滤：由 store 接口统一实现。
    """
    vs = get_vectordb()

    k = max(1, int(top_n))
    dept = (department or "").strip()

    if dept:
        return vs.similarity_search_with_score(query, k=k, filter={"department": dept})
    return vs.similarity_search_with_score(query, k=k)


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


def _apply_score_thresholds(items: List[Dict[str, Any]], *, use_rerank: bool) -> List[Dict[str, Any]]:
    """Apply optional score thresholds before returning final evidence."""

    vector_max_score = _env_vector_max_score()
    rerank_min_score = _env_rerank_min_score()

    filtered: List[Dict[str, Any]] = []
    for item in items:
        score = item.get("score")
        if vector_max_score is not None:
            if not isinstance(score, (int, float)) or isinstance(score, bool) or float(score) > vector_max_score:
                continue

        if use_rerank and rerank_min_score > 0:
            rerank_score = item.get("rerank_score")
            if not isinstance(rerank_score, (int, float)) or isinstance(rerank_score, bool):
                continue
            if float(rerank_score) < rerank_min_score:
                continue

        filtered.append(item)

    return filtered


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
        _set_last_retrieval_meta(
            {
                "query": "",
                "cache_hit": False,
                "cache_mode": None,
                "cache_backend": "redis" if _env_cache_enabled() else None,
                "hybrid_enabled": _env_hybrid_enabled(),
            }
        )
        return []

    try:
        vs = get_vectordb()
        if int(vs.count()) <= 0:
            _set_last_retrieval_meta(
                {
                    "query": q,
                    "search_query": q,
                    "top_k": int(top_k) if top_k and int(top_k) > 0 else 5,
                    "top_n": int(top_n) if top_n is not None else _env_top_n_default(),
                    "department": (department or "").strip(),
                    "use_rerank": _env_use_reranker() if use_rerank is None else bool(use_rerank),
                    "hybrid_enabled": _env_hybrid_enabled(),
                    "cache_hit": False,
                    "cache_mode": None,
                    "cache_backend": "redis" if _env_cache_enabled() else None,
                    "hits": 0,
                }
            )
            return []
    except Exception:
        pass

    # === [M1 检索优化：触发瘦身逻辑] ===
    # 策略：长度超过 15 个字才瘦身，短句直接检索
    search_q = q
    if len(q) > 15:
        search_q = _rewrite_query_slimming(q)
    # =============================

    k = int(top_k) if top_k and int(top_k) > 0 else 5
    n = int(top_n) if top_n is not None else _env_top_n_default()
    if n < k:
        n = k
    do_rerank = _env_use_reranker() if use_rerank is None else bool(use_rerank)
    hybrid_enabled = _env_hybrid_enabled()
    normalized_query = _normalize_query_for_cache(search_q)
    query_tokens = tuple(_tokenize_text(search_q))
    request_key = _build_cache_request_key(
        top_k=k,
        top_n=n,
        department=department,
        use_rerank=do_rerank,
        hybrid_enabled=hybrid_enabled,
    )

    base_meta: Dict[str, Any] = {
        "query": q,
        "search_query": search_q,
        "top_k": k,
        "top_n": n,
        "department": (department or "").strip(),
        "use_rerank": do_rerank,
        "hybrid_enabled": hybrid_enabled,
        "dense_hits": 0,
        "sparse_hits": 0,
    }

    cached_items, cache_meta = _lookup_cache(
        normalized_query=normalized_query,
        query_tokens=query_tokens,
        request_key=request_key,
    )
    if cached_items is not None:
        hit_meta = dict(base_meta)
        hit_meta.update(cache_meta)
        hit_meta["hits"] = len(cached_items)
        _set_last_retrieval_meta(hit_meta)
        return cached_items

    dense_docs_scores = _vector_search(search_q, top_n=n, department=department)
    dense_items: List[Dict[str, Any]] = []
    for i, (doc, score) in enumerate(dense_docs_scores, start=1):
        it = _normalize_evidence_item(idx=i, doc=doc, score=float(score) if score is not None else score)
        if not str(it.get("text") or "").strip():
            continue
        it["hybrid_dense_score"] = float(it.get("score") or 0.0)
        dense_items.append(it)

    sparse_items: List[Dict[str, Any]] = []
    if hybrid_enabled:
        sparse_docs_scores = _sparse_search(search_q, top_n=n, department=department)
        for i, (doc, sparse_score) in enumerate(sparse_docs_scores, start=1):
            it = _normalize_evidence_item(idx=i, doc=doc, score=0.0)
            if not str(it.get("text") or "").strip():
                continue
            it["hybrid_dense_score"] = 1.0
            it["hybrid_sparse_score"] = float(sparse_score)
            sparse_items.append(it)
        base_meta["sparse_hits"] = len(sparse_items)

    base_meta["dense_hits"] = len(dense_items)

    items = _merge_hybrid_candidates(dense_items=dense_items, sparse_items=sparse_items)
    items = _apply_hybrid_ranking(search_q, items)
    if do_rerank and items:
        # 重排建议使用原句 q，因为精排模型需要上下文
        items = _apply_rerank(q, items)

    items = _apply_score_thresholds(items, use_rerank=do_rerank)

    # 截断 top_k，并重建 eid 连续
    items = items[:k]
    for i, it in enumerate(items, start=1):
        it["eid"] = f"E{i}"
        it.pop("hybrid_dense_score", None)
        it.pop("hybrid_sparse_score", None)

    _store_cache(
        normalized_query=normalized_query,
        query_tokens=query_tokens,
        request_key=request_key,
        items=items,
    )
    miss_meta = dict(base_meta)
    miss_meta.update(cache_meta)
    miss_meta.update({"cache_hit": False, "cache_mode": None, "hits": len(items)})
    _set_last_retrieval_meta(miss_meta)
    return items


def get_stats() -> RagStats:
    """返回当前 RAG 底座状态。"""
    vs = get_vectordb()

    app_dir = resolve_app_dir(Path(__file__))
    persist_dir = resolve_persist_dir(app_dir)
    collection_name = env_str("RAG_COLLECTION", "") or DEFAULT_COLLECTION

    try:
        count = int(vs.count())
    except Exception:
        count = 0

    _, emb_info = get_embedder()

    rerank_model = _env_rerank_model() if _env_use_reranker() else None

    backend = getattr(vs, "backend_name", "faiss-hnsw")
    updated_at = ""
    try:
        ts = float(getattr(vs, "updated_at", lambda: 0.0)())
        if ts > 0:
            updated_at = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        updated_at = ""

    return RagStats(
        backend=str(backend),
        collection=collection_name,
        count=count,
        persist_dir=str(persist_dir),
        device=str(emb_info.device or resolve_embedding_device()),
        embed_model=str(emb_info.model_name),
        rerank_model=rerank_model,
        updated_at=updated_at,
    )
