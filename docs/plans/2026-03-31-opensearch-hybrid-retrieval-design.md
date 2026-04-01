# OpenSearch 统一混合召回设计

**日期：** 2026-03-31

**背景**

当前仓库的 RAG 主链路以 `Faiss-HNSW dense + 本地 token sparse + 应用层 rerank` 为主，已经具备可工作的召回能力，但与 README 中承诺的“BM25 + dense hybrid”仍不一致：

- sparse 侧不是标准 BM25 搜索后端
- dense 与 sparse 逻辑分散在应用层，工程形态不像生产检索平台
- 未来若需要迁移到云上托管检索，现有架构难以平滑演进

用户已明确本轮目标：

- 按 README 严格对齐混合召回能力
- 以 `OpenSearch` 作为目标后端
- 当前阶段优先本地单节点落地
- 不采用长期双写两套索引的生产形态

**设计目标**

将检索主后端统一到本地单节点 `OpenSearch`，同时承载：

- `BM25 sparse`
- `dense vector / kNN`
- `hybrid fusion`
- 现有应用层 rerank
- 现有 evidence / trace / API 契约

并满足以下约束：

- 不改变上层 `triage_service.py`、`api_server.py`、`agent/graph.py` 的调用接口
- 保持 `retrieve(query, top_k=5, ...) -> List[Dict]` 的 evidence 契约不变
- 保留现有 reranker 与证据阈值逻辑，减少一次性重构面
- 允许本地开发环境在缺失 OpenSearch 时给出明确错误，而不是静默回退成旧逻辑

**总体架构**

采用“单一主检索后端 + 应用层精排”的结构：

1. `app/rag/ingest_kb.py`
   负责把知识库 chunk 写入 OpenSearch 索引
2. `app/rag/opensearch_store.py`
   负责 OpenSearch 客户端、索引 mapping、bulk 写入、hybrid 查询
3. `app/rag/rag_core.py`
   保留统一检索入口，但底层改为调用 OpenSearch
4. `app/rag/retriever.py`
   继续作为兼容层，不改变外部调用方式
5. `app/rag/faiss_store.py`
   从主链路退出；本轮不立刻删除，但不再作为默认检索实现

检索链路变为：

`query -> OpenSearch(BM25 + kNN) -> hybrid 召回 -> 应用层 rerank -> threshold -> top_k evidence`

**索引设计**

新增单一 OpenSearch 索引，建议字段如下：

- `chunk_id`: keyword
- `text`: text
- `source`: keyword
- `source_file`: keyword
- `department`: keyword
- `title`: text / keyword 子字段
- `row`: integer
- `page`: integer
- `section`: keyword
- `domain`: keyword
- `source_kind`: keyword
- `session_id`: keyword
- `picked`: keyword
- `embedding`: `knn_vector`

设计原则：

- `text` 用于 BM25
- `embedding` 用于 dense kNN
- metadata 字段保持与现有 evidence 归一化逻辑兼容
- 优先使用固定维度向量与 HNSW 配置，保证本地单节点可运行

**入库设计**

`ingest_kb.py` 不再写入 Faiss store，而是：

1. 读取知识库源数据
2. 做现有文本清洗与语义切分
3. 通过现有 embedding 函数生成向量
4. 以 bulk 方式写入 OpenSearch

为避免一次 ingest 占用过大内存，bulk 采用分批写入：

- 每批固定文档数
- 每批生成向量后立即写入
- 写入失败时中止并返回清晰错误信息

**检索设计**

在 `rag_core.py` 中保留现有对外函数签名，但内部改成：

1. query 预处理与语义提纯逻辑保留
2. 根据 query 生成 embedding
3. 发起 OpenSearch hybrid 查询：
   - BM25：`match` / `multi_match`
   - dense：`knn` 查询
   - hybrid：优先使用 OpenSearch hybrid query / rank fusion；若当前本地版本或客户端能力受限，则采用应用层 RRF 作为兼容实现
4. 将返回结果归一化成现有 evidence 契约
5. 继续复用当前 rerank、threshold、cache、trace 输出

这里的关键策略是：

- `OpenSearch` 负责“标准稀疏 + 标准向量”召回
- `rag_core.py` 继续负责“项目定制的 rerank / 阈值 / evidence 格式 / trace”

**观测性设计**

保留并扩展当前 `get_last_retrieval_meta()` 输出，新增：

- `backend = opensearch`
- `hybrid_mode = opensearch_hybrid | app_rrf`
- `bm25_hits`
- `knn_hits`
- `fusion_hits`
- `opensearch_took_ms`
- `index_name`

`/v1/rag/stats` 也应改为返回 OpenSearch 相关状态，而不是 Faiss-only 状态。

**配置设计**

新增环境变量：

- `RAG_BACKEND=opensearch`
- `OPENSEARCH_URL`
- `OPENSEARCH_USERNAME`
- `OPENSEARCH_PASSWORD`
- `OPENSEARCH_INDEX`
- `OPENSEARCH_VERIFY_SSL`
- `OPENSEARCH_VECTOR_DIM`
- `OPENSEARCH_KNN_K`
- `OPENSEARCH_CANDIDATES`
- `RAG_HYBRID_MODE=opensearch|app_rrf`

原则：

- 本轮默认只支持 `OpenSearch` 作为主后端
- 缺配置时直接报错，避免“以为是 OpenSearch，实际还在走旧逻辑”

**测试策略**

本轮测试聚焦三层：

1. OpenSearch store 单元测试
   - mapping 构造正确
   - bulk payload 正确
   - hybrid 查询请求结构正确
2. rag_core 检索测试
   - BM25 + dense 结果能融合
   - rerank 与 evidence 归一化仍保持兼容
   - trace 输出新增 OpenSearch 字段
3. 回归测试
   - 现有 triage / chat / agent 安全熔断测试继续通过

测试不要求在 CI 中真的起一个 OpenSearch 节点；本轮优先用 mock/stub 验证查询构造和主链路行为。

**迁移与回滚**

迁移顺序：

1. 先引入 OpenSearch store 与配置
2. 再切入 ingest
3. 最后切换 retrieve 主链路

回滚策略：

- 代码层保留 `faiss_store.py`，但不作为默认路径
- 若 OpenSearch 主链路不稳定，可在单独分支回退 `rag_core.py` 调用层
- 本轮不做运行时自动回退，避免“静默切回旧实现”

**不做的事情**

本轮明确不做：

- AWS OpenSearch Service 托管部署
- 分布式集群运维脚本
- 大规模 benchmark 基准测试
- 删除全部 Faiss 历史代码
- 引入第二个专用向量数据库
