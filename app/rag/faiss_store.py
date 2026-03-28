# -*- coding: utf-8 -*-
"""Faiss-HNSW persistent vector store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _import_document():
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        from langchain_core.documents import Document  # type: ignore
    return Document


class FaissHNSWStore:
    def __init__(
        self,
        *,
        persist_dir: Path,
        embedding_function: Any,
        collection_name: str,
        hnsw_m: int = 32,
        ef_search: int = 128,
        ef_construction: int = 200,
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._embedding_function = embedding_function
        self._collection_name = str(collection_name or "").strip() or "medical_kb"
        self._hnsw_m = max(8, int(hnsw_m))
        self._ef_search = max(16, int(ef_search))
        self._ef_construction = max(16, int(ef_construction))

        self._index = None
        self._dim = 0
        self._docs: List[Dict[str, Any]] = []
        self._dirty = False
        self._load()

    @property
    def backend_name(self) -> str:
        return "faiss-hnsw"

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def persist_dir(self) -> Path:
        return self._persist_dir

    def _index_path(self) -> Path:
        return self._persist_dir / "index.faiss"

    def _docs_path(self) -> Path:
        return self._persist_dir / "docs.jsonl"

    def _meta_path(self) -> Path:
        return self._persist_dir / "meta.json"

    def _load(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        docs_path = self._docs_path()
        if docs_path.exists():
            loaded_docs: List[Dict[str, Any]] = []
            for line in docs_path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                data = json.loads(raw)
                if isinstance(data, dict):
                    loaded_docs.append(
                        {
                            "page_content": str(data.get("page_content") or ""),
                            "metadata": dict(data.get("metadata") or {}),
                        }
                    )
            self._docs = loaded_docs

        meta_path = self._meta_path()
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._dim = int(data.get("dim") or 0)
                self._collection_name = str(data.get("collection_name") or self._collection_name)

        index_path = self._index_path()
        if index_path.exists():
            try:
                import faiss  # type: ignore
            except Exception as e:
                raise RuntimeError("缺少 faiss-cpu，无法加载 Faiss-HNSW 索引。") from e
            self._index = faiss.read_index(str(index_path))
            self._index.hnsw.efSearch = self._ef_search
            self._dim = int(self._index.d)

    def _ensure_index(self, dim: int) -> Any:
        if self._index is not None:
            return self._index
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError("缺少 faiss-cpu，无法创建 Faiss-HNSW 索引。") from e
        self._dim = int(dim)
        self._index = faiss.IndexHNSWFlat(self._dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efSearch = self._ef_search
        self._index.hnsw.efConstruction = self._ef_construction
        return self._index

    def _as_document(self, payload: Dict[str, Any]) -> Any:
        Document = _import_document()
        return Document(
            page_content=str(payload.get("page_content") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _iter_filtered_docs(self, where: Optional[Dict[str, Any]] = None) -> Iterable[Tuple[int, Dict[str, Any]]]:
        where = where or {}
        for idx, payload in enumerate(self._docs):
            metadata = dict(payload.get("metadata") or {})
            matched = True
            for key, expected in where.items():
                if metadata.get(key) != expected:
                    matched = False
                    break
            if matched:
                yield idx, payload

    def count(self) -> int:
        return len(self._docs)

    def add_documents(self, documents: List[Any]) -> None:
        docs = list(documents or [])
        if not docs:
            return

        texts = [str(getattr(doc, "page_content", "") or "") for doc in docs]
        vectors = self._embedding_function.embed_documents(texts)
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise RuntimeError("embedding_function.embed_documents 返回了非法形状。")

        if matrix.shape[1] <= 0:
            raise RuntimeError("embedding 维度非法。")

        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError("缺少 faiss-cpu，无法写入 Faiss-HNSW 索引。") from e

        faiss.normalize_L2(matrix)
        index = self._ensure_index(int(matrix.shape[1]))
        index.add(matrix)

        for doc in docs:
            self._docs.append(
                {
                    "page_content": str(getattr(doc, "page_content", "") or ""),
                    "metadata": dict(getattr(doc, "metadata", None) or {}),
                }
            )

        self._dirty = True

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        if not self._docs or self._index is None:
            return []

        limit = max(1, int(k))
        query_vector = np.asarray([self._embedding_function.embed_query(str(query or ""))], dtype="float32")
        if query_vector.ndim != 2 or query_vector.shape[1] != self._dim:
            raise RuntimeError("embedding_function.embed_query 返回了非法维度。")

        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError("缺少 faiss-cpu，无法执行 Faiss-HNSW 检索。") from e

        faiss.normalize_L2(query_vector)
        search_k = min(max(limit * 8, 64), len(self._docs))
        scores, indices = self._index.search(query_vector, search_k)

        out: List[Tuple[Any, float]] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self._docs):
                continue
            payload = self._docs[idx]
            metadata = dict(payload.get("metadata") or {})
            if filter:
                if any(metadata.get(key) != value for key, value in filter.items()):
                    continue
            # Faiss inner product 越大越近；统一转换成“越小越好”的距离分数。
            distance = float(max(0.0, 1.0 - float(score)))
            out.append((self._as_document(payload), distance))
            if len(out) >= limit:
                break
        return out

    def get_documents(self, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for _, payload in self._iter_filtered_docs(where):
            documents.append(str(payload.get("page_content") or ""))
            metadatas.append(dict(payload.get("metadata") or {}))
        return {"documents": documents, "metadatas": metadatas}

    def persist(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        docs_path = self._docs_path()
        docs_payload = "\n".join(
            json.dumps(doc, ensure_ascii=False) for doc in self._docs
        )
        docs_path.write_text(docs_payload + ("\n" if docs_payload else ""), encoding="utf-8")

        meta = {
            "backend": self.backend_name,
            "collection_name": self._collection_name,
            "dim": self._dim,
            "count": len(self._docs),
        }
        self._meta_path().write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        index_path = self._index_path()
        if self._index is not None:
            try:
                import faiss  # type: ignore
            except Exception as e:
                raise RuntimeError("缺少 faiss-cpu，无法持久化 Faiss-HNSW 索引。") from e
            faiss.write_index(self._index, str(index_path))
        elif index_path.exists():
            index_path.unlink()

        self._dirty = False

    def updated_at(self) -> float:
        ts = 0.0
        for path in [self._index_path(), self._docs_path(), self._meta_path()]:
            if path.exists():
                ts = max(ts, path.stat().st_mtime)
        return ts
