# OpenSearch 统一混合召回实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 将当前 RAG 主链路改造成以本地单节点 OpenSearch 为统一检索后端，落实 README 所述的标准 BM25 + dense hybrid 检索能力。

**架构：** 新增 OpenSearch store 负责索引、bulk 写入与 hybrid 查询；`ingest_kb.py` 改为写入 OpenSearch；`rag_core.py` 改为从 OpenSearch 获取 BM25 与 dense 召回结果，再复用现有 rerank、threshold 与 evidence 契约。

**技术栈：** Python 3.11、OpenSearch Python client、FastAPI、LangChain 兼容数据结构、pytest。

---

### 任务 1：补 OpenSearch store 的失败测试

**文件：**
- 新建：`tests/test_opensearch_store.py`
- 新建：`app/rag/opensearch_store.py`
- 修改：`requirements.txt`
- 修改：`.env.example`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- store 能生成包含 `text` 与 `embedding` 的索引 mapping
- bulk payload 会把 metadata 与向量一起写入
- hybrid 查询请求同时包含 BM25 和 kNN 所需参数

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_opensearch_store.py -v`
预期：FAIL，因为 `opensearch_store.py` 尚不存在

**步骤 3：编写最小实现**

实现 OpenSearch store：
- 客户端构造
- 索引 mapping 生成
- bulk 文档转换
- hybrid 查询请求构造

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_opensearch_store.py -v`
预期：PASS

### 任务 2：将 ingest 链路改为写入 OpenSearch

**文件：**
- 新建：`tests/test_ingest_opensearch.py`
- 修改：`app/rag/ingest_kb.py`
- 修改：`app/rag/utils/rag_shared.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- ingest 会按批生成 embedding 并写入 OpenSearch
- chunk metadata 保持与现有 evidence 契约兼容
- ingest 完成后不再依赖 Faiss 持久化

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_ingest_opensearch.py -v`
预期：FAIL，因为 ingest 仍在写 Faiss

**步骤 3：编写最小实现**

修改 ingest：
- 用 OpenSearch store 接收批量 chunk
- 每批先算 embedding 再 bulk 写入
- 保留原有清洗、切分与 metadata 生成逻辑

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_ingest_opensearch.py -v`
预期：PASS

### 任务 3：将 rag_core 检索主链路切到 OpenSearch hybrid

**文件：**
- 新建：`tests/test_rag_core_opensearch_hybrid.py`
- 修改：`app/rag/rag_core.py`
- 修改：`app/rag/retriever.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- `retrieve()` 会调用 OpenSearch hybrid 查询而不是旧的 Faiss + 本地 sparse
- 返回 evidence 契约与现有字段保持兼容
- retrieval meta 会写入 `backend=opensearch`、`bm25_hits`、`knn_hits`、`hybrid_mode`

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_rag_core_opensearch_hybrid.py -v`
预期：FAIL，因为主链路尚未切换

**步骤 3：编写最小实现**

修改 `rag_core.py`：
- 接入 OpenSearch store
- 删除主链路中的本地 BM25-like sparse 计算
- 将 hybrid 查询结果归一化到现有 evidence 契约
- 保留 query slimming、rerank、threshold 与 cache 逻辑

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_rag_core_opensearch_hybrid.py -v`
预期：PASS

### 任务 4：补 API / stats / 配置回归

**文件：**
- 新建：`tests/test_rag_stats_opensearch.py`
- 修改：`app/api_server.py`
- 修改：`app/rag/rag_core.py`
- 修改：`.env.example`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- `/v1/rag/stats` 返回 OpenSearch 后端状态
- `retrieval_meta` 包含 OpenSearch hybrid 关键信息
- 缺失 OpenSearch 配置时返回清晰错误而不是静默回退

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_rag_stats_opensearch.py -v`
预期：FAIL，因为 stats 仍以 Faiss 语义为主

**步骤 3：编写最小实现**

修改 stats 与配置：
- 新增 OpenSearch 环境变量
- stats 改为返回 OpenSearch backend/index/config 信息
- 缺配置时报可读错误

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_rag_stats_opensearch.py -v`
预期：PASS

### 任务 5：全量相关回归

**文件：**
- 修改：`tests/` 下新增 OpenSearch 检索测试
- 修改：`app/rag/` 与必要的 API 文件

**步骤 1：运行 OpenSearch 检索相关测试**

运行：
`pytest tests/test_opensearch_store.py tests/test_ingest_opensearch.py tests/test_rag_core_opensearch_hybrid.py tests/test_rag_stats_opensearch.py -v`

预期：PASS

**步骤 2：运行现有安全与主链路回归**

运行：
`pytest tests/test_safety_fuse.py tests/test_triage_safety_fuse.py tests/test_api_chat_safety_fuse.py tests/test_agent_chat_v2_safety_fuse.py -v`

预期：PASS，说明检索后端替换没有破坏安全熔断链路

**步骤 3：运行一次基础全量回归**

运行：`pytest -q`
预期：PASS

**步骤 4：提交**

运行：
```bash
git add app/rag app/api_server.py tests docs/plans .env.example requirements.txt
git commit -m "feat: add opensearch hybrid retrieval backend"
```
