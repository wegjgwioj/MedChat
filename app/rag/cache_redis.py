# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class CacheEntry:
    cache_key: str
    request_key: str
    normalized_query: str
    query_tokens: Tuple[str, ...]
    created_at: float
    expires_at: float
    items: List[Dict[str, Any]]


def _token_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = {str(item) for item in a if str(item)}
    sb = {str(item) for item in b if str(item)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


class RedisSemanticCache:
    def __init__(self, redis_url: str, key_prefix: str = "medchat:rag-cache:", client: Any = None):
        self._redis_url = str(redis_url or "").strip()
        if not self._redis_url:
            raise RuntimeError("RAG Redis 语义缓存缺少 redis_url，请配置 RAG_REDIS_URL 或 AGENT_REDIS_URL。")
        self._key_prefix = str(key_prefix or "medchat:rag-cache:").strip() or "medchat:rag-cache:"
        self._client = client or self._create_client(self._redis_url)
        try:
            self._client.ping()
        except Exception as e:
            raise RuntimeError(f"RAG Redis 语义缓存不可用：{type(e).__name__}: {e}") from e

    @staticmethod
    def _create_client(redis_url: str) -> Any:
        try:
            import redis  # type: ignore
        except Exception as e:
            raise RuntimeError("未安装 redis 客户端，无法初始化 RAG Redis 语义缓存。") from e
        return redis.from_url(redis_url, decode_responses=False)

    def backend_name(self) -> str:
        return "redis"

    def clear(self) -> None:
        self._client.delete(self._entries_key())

    def lookup(
        self,
        *,
        normalized_query: str,
        query_tokens: Tuple[str, ...],
        request_key: str,
        sim_threshold: float,
        max_entries: int,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        now = time.time()
        exact_key = self._build_cache_key(normalized_query, request_key)
        entries = self._prune_entries(self._load_entries(), now=now, max_entries=max_entries)
        exact_entry = self._find_exact(entries, exact_key=exact_key)
        if exact_entry is not None:
            entries = self._move_to_tail(entries, exact_entry.cache_key)
            self._save_entries(entries)
            return deepcopy(exact_entry.items), {
                "cache_hit": True,
                "cache_mode": "exact",
                "cache_similarity": 1.0,
                "cache_backend": self.backend_name(),
            }

        semantic_entry = self._find_semantic(
            entries,
            request_key=request_key,
            query_tokens=query_tokens,
            sim_threshold=sim_threshold,
        )
        if semantic_entry is not None:
            entries = self._move_to_tail(entries, semantic_entry.cache_key)
            self._save_entries(entries)
            return deepcopy(semantic_entry.items), {
                "cache_hit": True,
                "cache_mode": "semantic",
                "cache_similarity": round(_token_jaccard(query_tokens, semantic_entry.query_tokens), 4),
                "cache_backend": self.backend_name(),
            }
        return None, {"cache_hit": False, "cache_mode": None, "cache_backend": self.backend_name()}

    def store(
        self,
        *,
        normalized_query: str,
        query_tokens: Tuple[str, ...],
        request_key: str,
        items: List[Dict[str, Any]],
        ttl_seconds: float,
        max_entries: int,
    ) -> None:
        now = time.time()
        entries = self._prune_entries(self._load_entries(), now=now, max_entries=max_entries)
        cache_key = self._build_cache_key(normalized_query, request_key)
        entries = [entry for entry in entries if entry.cache_key != cache_key]
        entries.append(
            CacheEntry(
                cache_key=cache_key,
                request_key=request_key,
                normalized_query=normalized_query,
                query_tokens=tuple(query_tokens),
                created_at=now,
                expires_at=now + float(ttl_seconds),
                items=deepcopy(items),
            )
        )
        entries = self._prune_entries(entries, now=now, max_entries=max_entries)
        self._save_entries(entries)

    def _entries_key(self) -> str:
        return f"{self._key_prefix}entries"

    def _build_cache_key(self, normalized_query: str, request_key: str) -> str:
        return f"{request_key}|query={normalized_query}"

    def _load_entries(self) -> List[CacheEntry]:
        raw = self._client.get(self._entries_key())
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if not isinstance(raw, str) or not raw.strip():
            return []

        try:
            parsed = json.loads(raw)
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []

        out: List[CacheEntry] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                out.append(
                    CacheEntry(
                        cache_key=str(item.get("cache_key") or "").strip(),
                        request_key=str(item.get("request_key") or "").strip(),
                        normalized_query=str(item.get("normalized_query") or "").strip(),
                        query_tokens=tuple(str(token) for token in (item.get("query_tokens") or []) if str(token)),
                        created_at=float(item.get("created_at") or 0.0),
                        expires_at=float(item.get("expires_at") or 0.0),
                        items=deepcopy(list(item.get("items") or [])),
                    )
                )
            except Exception:
                continue
        return out

    def _save_entries(self, entries: List[CacheEntry]) -> None:
        payload = [
            {
                "cache_key": entry.cache_key,
                "request_key": entry.request_key,
                "normalized_query": entry.normalized_query,
                "query_tokens": list(entry.query_tokens),
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "items": deepcopy(entry.items),
            }
            for entry in entries
        ]
        self._client.set(self._entries_key(), json.dumps(payload, ensure_ascii=False))

    def _prune_entries(self, entries: List[CacheEntry], *, now: float, max_entries: int) -> List[CacheEntry]:
        active = [entry for entry in entries if entry.expires_at > now]
        if max_entries <= 0:
            return []
        if len(active) > max_entries:
            active = active[-max_entries:]
        return active

    def _find_exact(self, entries: List[CacheEntry], *, exact_key: str) -> Optional[CacheEntry]:
        for entry in reversed(entries):
            if entry.cache_key == exact_key:
                return entry
        return None

    def _find_semantic(
        self,
        entries: List[CacheEntry],
        *,
        request_key: str,
        query_tokens: Tuple[str, ...],
        sim_threshold: float,
    ) -> Optional[CacheEntry]:
        for entry in reversed(entries):
            if entry.request_key != request_key:
                continue
            sim = _token_jaccard(query_tokens, entry.query_tokens)
            if sim >= sim_threshold:
                return entry
        return None

    def _move_to_tail(self, entries: List[CacheEntry], cache_key: str) -> List[CacheEntry]:
        matched: Optional[CacheEntry] = None
        remaining: List[CacheEntry] = []
        for entry in entries:
            if matched is None and entry.cache_key == cache_key:
                matched = entry
                continue
            remaining.append(entry)
        if matched is not None:
            remaining.append(matched)
        return remaining
