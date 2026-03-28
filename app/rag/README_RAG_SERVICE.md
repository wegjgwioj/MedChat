# RAGService（M1）

本模块在不破坏现有 `/v1/triage` 与 `/v1/chat` 的前提下，新增了一个可复用的 RAGService：

- 进程内实现：`app/rag/rag_core.py`
- 兼容层：`app/rag/retriever.py` 仍保留 `retrieve(query, top_k=5)` 的调用方式
- 独立 API：`/v1/rag/stats` 与 `/v1/rag/retrieve`

## 1. 做了什么

| 项目 | 旧方式（可能存在） | M1 方式（当前） | 收益 |
|---|---|---|---|
| Embedding 统一 | 入库与检索可能配置不一致 | 入库与检索共用同一套配置（默认 BCE embedding） | 结果稳定，可复现 |
| GPU 优先 | 可能未启用/不明确 | `RAG_DEVICE=auto` 时优先 cuda，否则 cpu | 加速入库与检索 |
| 两阶段检索 | 只做向量检索 top_k | Faiss-HNSW dense top_n + sparse 合并 -> rerank -> top_k | 召回更稳、排序更好 |
| 证据契约 | 字段不固定 | 固定字段（eid/text/source/chunk_id/score/rerank_score/metadata） | 便于评测与前端展示 |
| 语义缓存 | 无或仅本地缓存 | Redis 语义缓存（exact/semantic 命中） | 重复 query 延迟更低 |
| 独立接口 | 无 | 新增 `/v1/rag/*` | Agent/前端/评测可复用 |

## 2. 接口

### 2.1 GET /v1/rag/stats

返回当前向量库状态：

```bash
curl http://127.0.0.1:8000/v1/rag/stats
```

示例输出（字段示例）：

```json
{
  "backend": "faiss-hnsw",
  "collection": "medical_kb",
  "count": 801506,
  "persist_dir": ".../app/rag/kb_store",
  "device": "cuda",
  "embed_model": "maidalun1020/bce-embedding-base_v1",
  "rerank_model": "maidalun1020/bce-reranker-base_v1",
  "updated_at": "2025-12-23 10:23:21"
}
```

### 2.2 POST /v1/rag/retrieve

入参：

```json
{
  "query": "高血压能喝党参吗",
  "top_k": 3,
  "top_n": 30,
  "department": null,
  "use_rerank": null
}
```

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"高血压能喝党参吗\",\"top_k\":3}"
```

返回：
- `evidence` 为证据列表
- `retrieval_meta` 为最近一次检索的元信息（如 `cache_hit/cache_mode/cache_backend/hybrid_enabled/dense_hits/sparse_hits`）
- `stats` 为检索时刻的底座信息（不包含会话文本）

注意：服务端日志只会记录 query 的前 100 个字符或哈希，不会泄漏完整内容。

## 3. 证据契约（固定）

`retrieve()` 返回 `List[Dict]`，每条证据必须包含：

- `eid`: "E1"..."Ek"
- `text`: 证据文本（非空）
- `source`: 来源（至少稳定为文件名）
- `chunk_id`: 可追溯稳定 ID（`source:row:local_idx`）
- `score`: 向量阶段相似度/距离（数值）
- `rerank_score`: rerank 阶段分数（启用时为数值，否则为 null）
- `metadata`: 至少包含 `department/title/row/source_file`

## 4. 配置（.env）

推荐在项目根目录 `.env` 或系统环境变量中设置：

- `RAG_PROVIDER=bce`：默认
- `RAG_DEVICE=auto`：默认优先 cuda
- `RAG_EMBED_MODEL=maidalun1020/bce-embedding-base_v1`
- `RAG_RERANK_MODEL=maidalun1020/bce-reranker-base_v1`
- `RAG_USE_RERANKER=1`：默认开启
- `RAG_TOP_N=30`：默认
- `RAG_COLLECTION=medical_kb`
- `RAG_PERSIST_DIR=app/rag/kb_store`
- `RAG_CACHE_ENABLED=1`
- `RAG_REDIS_URL=redis://...`（或复用 `AGENT_REDIS_URL`）

兼容性：仍兼容旧变量（例如 `RAG_EMBEDDING_DEVICE`、`RAG_EMBEDDINGS_PROVIDER`、`RAG_BCE_MODEL`）。

## 5. 常见问题

### 5.1 GPU 未生效

- 检查 `RAG_DEVICE=auto` 是否设置
- `python -c "import torch; print(torch.cuda.is_available())"`（Windows 可能需先设置 `KMP_DUPLICATE_LIB_OK=TRUE`）
- 若强制 GPU：`RAG_DEVICE=cuda`，但若 torch 非 CUDA 版会直接报错提示如何处理

### 5.2 模型下载慢

- 首次使用会从 HuggingFace 下载模型
- 可提前在稳定网络下完成一次检索/入库，后续会命中缓存

### 5.3 索引目录缺失/找不到库

- 确认 `RAG_PERSIST_DIR` 指向正确目录（默认 `app/rag/kb_store`）
- 若目录不存在：先运行 `python app/rag/ingest_kb.py`
