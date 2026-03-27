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
import math
import re
import threading
import time
from collections import OrderedDict
from copy import deepcopy
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
from app.privacy import redact_pii_for_llm


@dataclass(frozen=True)
class RagStats:
    collection: str
    count: int
    persist_dir: str
    device: str
    embed_model: str
    rerank_model: Optional[str]
    updated_at: str


@dataclass
class _CacheEntry:
    cache_key: str
    request_key: str
    normalized_query: str
    query_tokens: Tuple[str, ...]
    created_at: float
    expires_at: float
    items: List[Dict[str, Any]]


_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()
_vs_lock = threading.Lock()
_cache_lock = threading.Lock()
_runtime_lock = threading.Lock()

_cached_embed: Optional[Tuple[Any, EmbeddingInfo]] = None
_cached_reranker: Optional[Any] = None
_cached_vs: Optional[Any] = None
_retrieval_cache: "OrderedDict[str, _CacheEntry]" = OrderedDict()
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
    global _retrieval_cache, _last_retrieval_meta

    with _embed_lock:
        _cached_embed = None
    with _rerank_lock:
        _cached_reranker = None
    with _vs_lock:
        _cached_vs = None
    with _cache_lock:
        _retrieval_cache = OrderedDict()
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


def _token_jaccard(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _normalize_dense_relevance(items: List[Dict[str, Any]]) -> List[float]:
    scores: List[float] = []
    for item in items:
        score = item.get("score")
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


def _build_cache_key(normalized_query: str, request_key: str) -> str:
    return f"{request_key}|query={normalized_query}"


def _prune_cache(now: float) -> None:
    expired = [key for key, entry in _retrieval_cache.items() if entry.expires_at <= now]
    for key in expired:
        _retrieval_cache.pop(key, None)

    max_entries = _env_cache_max_entries()
    while len(_retrieval_cache) > max_entries:
        _retrieval_cache.popitem(last=False)


def _lookup_cache(
    *,
    normalized_query: str,
    query_tokens: Tuple[str, ...],
    request_key: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    if not _env_cache_enabled():
        return None, {"cache_hit": False, "cache_mode": None}

    now = time.monotonic()
    exact_key = _build_cache_key(normalized_query, request_key)

    with _cache_lock:
        _prune_cache(now)

        exact_entry = _retrieval_cache.get(exact_key)
        if exact_entry is not None:
            _retrieval_cache.move_to_end(exact_key)
            return deepcopy(exact_entry.items), {"cache_hit": True, "cache_mode": "exact", "cache_similarity": 1.0}

        sim_threshold = _env_cache_sim_threshold()
        for key in list(_retrieval_cache.keys())[::-1]:
            entry = _retrieval_cache.get(key)
            if entry is None or entry.request_key != request_key:
                continue
            sim = _token_jaccard(query_tokens, entry.query_tokens)
            if sim >= sim_threshold:
                _retrieval_cache.move_to_end(key)
                return deepcopy(entry.items), {"cache_hit": True, "cache_mode": "semantic", "cache_similarity": round(sim, 4)}

    return None, {"cache_hit": False, "cache_mode": None}


def _store_cache(
    *,
    normalized_query: str,
    query_tokens: Tuple[str, ...],
    request_key: str,
    items: List[Dict[str, Any]],
) -> None:
    if not _env_cache_enabled():
        return

    now = time.monotonic()
    ttl_seconds = float(_env_cache_ttl_seconds())
    entry = _CacheEntry(
        cache_key=_build_cache_key(normalized_query, request_key),
        request_key=request_key,
        normalized_query=normalized_query,
        query_tokens=query_tokens,
        created_at=now,
        expires_at=now + ttl_seconds,
        items=deepcopy(items),
    )
    with _cache_lock:
        _retrieval_cache[entry.cache_key] = entry
        _retrieval_cache.move_to_end(entry.cache_key)
        _prune_cache(now)


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
        _set_last_retrieval_meta({"query": "", "cache_hit": False, "cache_mode": None, "hybrid_enabled": _env_hybrid_enabled()})
        return []

    try:
        vs = get_vectordb()
        if int(vs._collection.count()) <= 0:  # type: ignore[attr-defined]
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

    # 使用 search_q 进行向量召回
    docs_scores = _vector_search(search_q, top_n=n, department=department)

    items: List[Dict[str, Any]] = []
    for i, (doc, score) in enumerate(docs_scores, start=1):
        it = _normalize_evidence_item(idx=i, doc=doc, score=float(score) if score is not None else score)
        # text 不能为空（契约要求）
        if not str(it.get("text") or "").strip():
            continue
        items.append(it)

    items = _apply_hybrid_ranking(search_q, items)
    if do_rerank and items:
        # 重排建议使用原句 q，因为精排模型需要上下文
        items = _apply_rerank(q, items)

    items = _apply_score_thresholds(items, use_rerank=do_rerank)

    # 截断 top_k，并重建 eid 连续
    items = items[:k]
    for i, it in enumerate(items, start=1):
        it["eid"] = f"E{i}"

    _store_cache(
        normalized_query=normalized_query,
        query_tokens=query_tokens,
        request_key=request_key,
        items=items,
    )
    miss_meta = dict(base_meta)
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
