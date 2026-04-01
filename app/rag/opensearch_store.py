# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class OpenSearchConfig:
    url: str
    index_name: str
    vector_dim: int
    username: str = ""
    password: str = ""
    verify_ssl: bool = False
    knn_k: int = 30
    num_candidates: int = 128
    hybrid_mode: str = "app_rrf"


class OpenSearchHybridStore:
    backend_name = "opensearch"

    def __init__(self, config: OpenSearchConfig, client: Optional[Any] = None) -> None:
        self.config = config
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            from opensearchpy import OpenSearch  # type: ignore
        except Exception as exc:
            raise RuntimeError("缺少 opensearch-py 依赖，无法连接 OpenSearch。") from exc

        options: Dict[str, Any] = {
            "hosts": [self.config.url],
            "use_ssl": self.config.url.startswith("https://"),
            "verify_certs": bool(self.config.verify_ssl),
        }
        if self.config.username or self.config.password:
            options["http_auth"] = (self.config.username, self.config.password)
        return OpenSearch(**options)

    def build_index_body(self) -> Dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "knn": True,
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "source_file": {"type": "keyword"},
                    "department": {"type": "keyword"},
                    "title": {"type": "keyword"},
                    "row": {"type": "integer"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": int(self.config.vector_dim),
                    },
                }
            },
        }

    def build_bulk_operations(self, docs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        operations: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            payload = dict(doc or {})
            chunk_id = str(payload.get("chunk_id") or "").strip() or f"chunk-{idx}"
            operations.append({"index": {"_index": self.config.index_name, "_id": chunk_id}})
            payload["chunk_id"] = chunk_id
            operations.append(payload)
        return operations

    def build_search_body(
        self,
        *,
        query: str,
        query_vector: List[float],
        top_k: int,
        department: Optional[str] = None,
    ) -> Dict[str, Any]:
        bool_query: Dict[str, Any] = {
            "should": [
                {
                    "match": {
                        "text": {
                            "query": query,
                        }
                    }
                },
                {
                    "knn": {
                        "embedding": {
                            "vector": list(query_vector or []),
                            "k": max(1, int(top_k)),
                            "num_candidates": max(1, int(self.config.num_candidates)),
                        }
                    }
                },
            ],
            "minimum_should_match": 1,
        }
        dept = str(department or "").strip()
        if dept:
            bool_query["filter"] = [{"term": {"department": dept}}]

        return {
            "size": max(1, int(top_k)),
            "query": {"bool": bool_query},
        }

    def ensure_index(self) -> None:
        indices = getattr(self.client, "indices", None)
        if indices is None:
            return
        try:
            exists = bool(indices.exists(index=self.config.index_name))
        except Exception:
            exists = False
        if not exists:
            indices.create(index=self.config.index_name, body=self.build_index_body())

    def bulk_upsert(self, docs: Iterable[Dict[str, Any]]) -> int:
        operations = self.build_bulk_operations(docs)
        if not operations:
            return 0
        self.ensure_index()
        self.client.bulk(body=operations, refresh=True)
        return len(operations) // 2

    def search_hybrid(
        self,
        *,
        query: str,
        query_vector: List[float],
        top_k: int,
        department: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self.build_search_body(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            department=department,
        )
        response = self.client.search(index=self.config.index_name, body=body)
        hits = list(((response or {}).get("hits") or {}).get("hits") or [])
        return {
            "hits": hits,
            "meta": {
                "backend": "opensearch",
                "hybrid_mode": self.config.hybrid_mode,
                "bm25_hits": len(hits),
                "knn_hits": len(hits),
                "fusion_hits": len(hits),
                "took_ms": int((response or {}).get("took") or 0),
                "index_name": self.config.index_name,
            },
        }

    def count(self) -> int:
        try:
            response = self.client.count(index=self.config.index_name)
        except Exception:
            return 0
        try:
            return int((response or {}).get("count") or 0)
        except Exception:
            return 0
