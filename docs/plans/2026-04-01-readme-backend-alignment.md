# README 白名单后端对齐实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 将当前仓库后端从“README 目标态 + 历史遗留并存”升级为与 README 完全一致的单一后端系统，删除旧接口、旧状态源和静默 fallback。

**架构：** 以 `agent/chat_v2` 为唯一多轮问诊主入口，以 `Redis` 为唯一短期记忆后端，以 `OpenSearch` 为唯一检索后端；旧 `/v1/chat`、旧 `/v1/triage`、文件 session、SQLite fallback、Faiss 主路径与 README 外调试接口全部删除。

**技术栈：** Python 3.11、FastAPI、LangGraph、Redis、OpenSearch、pytest。

---

### 任务 1：建立 README 白名单测试与删除清单

**文件：**
- 新建：`tests/test_api_surface_readme_alignment.py`
- 新建：`tests/test_session_storage_readme_alignment.py`
- 修改：`app/api_server.py`
- 修改：`app/agent/storage.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- 公开 API 白名单只允许保留 `/health`、`/v1/agent/chat_v2`、`/v1/agent/chat_v2/stream`
- `build_session_store()` 在未配置 `AGENT_REDIS_URL` 时直接报错
- 不允许再自动回退到 SQLite 或文件会话

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_api_surface_readme_alignment.py tests/test_session_storage_readme_alignment.py -v`
预期：FAIL，因为当前仓库仍暴露旧接口且仍存在 SQLite fallback

**步骤 3：编写最小实现**

实现最小变更：
- 收紧 `build_session_store()` 为 Redis-only
- 暂不删除全部旧逻辑，但先让白名单测试显式指向待删除对象

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_api_surface_readme_alignment.py tests/test_session_storage_readme_alignment.py -v`
预期：PASS

### 任务 2：切换短期记忆为 Redis-only

**文件：**
- 修改：`app/agent/storage.py`
- 修改：`app/agent/storage_redis.py`
- 删除：`app/agent/storage_sqlite.py`
- 删除：`app/agent/storage_memory.py`
- 修改：`app/agent/graph.py`
- 修改：`app/api_server.py`
- 修改：`tests/test_agent_chat_v2_safety_fuse.py`

**步骤 1：先写失败测试**

新增或改写测试覆盖以下行为：
- `agent` 主链路只接受 Redis session store
- trace 中 `storage_meta.type` 必须为 `redis`
- 缺少 Redis 配置时 `agent/chat_v2` 初始化或调用直接失败

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_agent_chat_v2_safety_fuse.py tests/test_session_storage_readme_alignment.py -v`
预期：FAIL，因为当前测试和代码仍依赖 SQLite store

**步骤 3：编写最小实现**

实现要点：
- 删除 SQLite / memory store 代码文件
- 删除 `build_session_store()` fallback
- 将所有依赖 session store 的调用统一到 Redis-only 契约
- 将测试改为使用 fake Redis client / monkeypatch，而不是 SQLite 文件

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_agent_chat_v2_safety_fuse.py tests/test_session_storage_readme_alignment.py -v`
预期：PASS

### 任务 3：删除旧 `/v1/chat` 链路与文件会话持久化

**文件：**
- 修改：`app/api_server.py`
- 删除：旧 `/v1/chat` 相关 helper 与旧 chat graph 代码段
- 删除：`tests/test_api_chat_safety_fuse.py`
- 新建：`tests/test_legacy_chat_removed.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- `POST /v1/chat` 返回 404
- OpenAPI 中不再出现 `/v1/chat`
- 不再存在 `OUTPUT_DIR/sessions/*.json` 会话持久化路径依赖

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_legacy_chat_removed.py -v`
预期：FAIL，因为当前仍注册 `/v1/chat`

**步骤 3：编写最小实现**

实现要点：
- 删除旧 `ChatRequest` 主路径
- 删除 `_session_file_path`、`_load_or_create_session`、`_save_session`
- 删除旧 `_get_chat_graph()` 及其 inquiry/triage 分叉逻辑
- 同步移除依赖旧接口的测试

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_legacy_chat_removed.py -v`
预期：PASS

### 任务 4：删除 README 外调试与历史兼容接口

**文件：**
- 修改：`app/api_server.py`
- 新建：`tests/test_public_api_whitelist.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- `/v1/triage` 不再公开
- `/v1/rag/stats` 不再公开
- `/v1/rag/retrieve` 不再公开
- `/v1/ocr/ingest` 与 `/v1/ocr/status/{task_id}` 不再公开

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_public_api_whitelist.py -v`
预期：FAIL，因为这些接口当前仍然存在

**步骤 3：编写最小实现**

实现要点：
- 从 FastAPI app 中移除 README 外公共路由
- 若 OCR/triage/retrieval 有仍需复用的内部逻辑，则迁为内部函数，不再公开为 API

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_public_api_whitelist.py -v`
预期：PASS

### 任务 5：收口 agent 主链路到 README 的 Phase 0/1/2

**文件：**
- 修改：`app/agent/graph.py`
- 修改：`app/agent/router.py`
- 修改：`app/agent/state.py`
- 修改：`app/agent/prompts.py`
- 新建：`tests/test_agent_readme_alignment.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- Phase 0：OOD / 恶意请求会触发护栏拒答
- Phase 0：PII 脱敏信息进入 trace 或中间态，但不落原文
- Phase 1：信息不足时主动追问，而不是直接回答
- Phase 2：信息充分时生成结构化主诉并触发检索

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_agent_readme_alignment.py -v`
预期：FAIL，因为当前实现仍带有部分历史分叉与不完整约束

**步骤 3：编写最小实现**

实现要点：
- 统一 graph 节点职责，使其严格映射 README 的三阶段
- 清理历史兼容状态和重复分支
- 保留 SSE 语义，但统一底层结果来源

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_agent_readme_alignment.py -v`
预期：PASS

### 任务 6：清理非 OpenSearch 主检索路径

**文件：**
- 修改：`app/rag/rag_core.py`
- 修改：`app/rag/ingest_kb.py`
- 删除：`app/rag/faiss_store.py`
- 视情况删除或收缩：`app/rag/retriever.py`
- 修改：现有 OpenSearch 测试
- 新建：`tests/test_retrieval_readme_alignment.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- 检索主链路只允许 `OpenSearch`
- 缺失 OpenSearch 配置时显式失败
- 不存在 Faiss fallback 或本地 sparse 主路径

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_retrieval_readme_alignment.py tests/test_opensearch_store.py tests/test_rag_core_opensearch_hybrid.py tests/test_ingest_opensearch.py tests/test_rag_stats_opensearch.py -v`
预期：FAIL，因为当前仓库仍保留部分 Faiss 代码与 stats 语义

**步骤 3：编写最小实现**

实现要点：
- 从主链路删除 Faiss 相关代码路径
- 收紧 `rag_core` 和 ingest 为 OpenSearch-only
- 同步删除不再需要的文件和引用

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_retrieval_readme_alignment.py tests/test_opensearch_store.py tests/test_rag_core_opensearch_hybrid.py tests/test_ingest_opensearch.py tests/test_rag_stats_opensearch.py -v`
预期：PASS

### 任务 7：统一纵向档案与安全熔断状态源

**文件：**
- 修改：`app/agent/record_index.py`
- 修改：`app/safety/record_guard.py`
- 修改：`app/safety/safety_fuse.py`
- 修改：`app/agent/graph.py`
- 新建：`tests/test_longitudinal_record_readme_alignment.py`
- 修改：`tests/test_safety_fuse.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- 纵向档案只从 Redis-backed session 状态读取
- 安全熔断不依赖旧 `/v1/chat` 或 SQLite 残留状态
- agent 主链路输出的安全 trace 与档案来源统一

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_longitudinal_record_readme_alignment.py tests/test_safety_fuse.py tests/test_agent_chat_v2_safety_fuse.py -v`
预期：FAIL，因为当前仍存在旧状态源依赖

**步骤 3：编写最小实现**

实现要点：
- 清理所有旧状态读取入口
- 将待确认事实、confirmed records、安全约束统一到同一 session 数据源
- 保持安全 trace 契约稳定

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_longitudinal_record_readme_alignment.py tests/test_safety_fuse.py tests/test_agent_chat_v2_safety_fuse.py -v`
预期：PASS

### 任务 8：全量白名单回归与清理

**文件：**
- 修改：`tests/`
- 修改：`app/`
- 修改：`docs/plans/`

**步骤 1：运行 README 白名单测试**

运行：
`pytest tests/test_api_surface_readme_alignment.py tests/test_session_storage_readme_alignment.py tests/test_legacy_chat_removed.py tests/test_public_api_whitelist.py tests/test_agent_readme_alignment.py tests/test_retrieval_readme_alignment.py tests/test_longitudinal_record_readme_alignment.py -v`

预期：PASS

**步骤 2：运行主链路回归**

运行：
`pytest tests/test_agent_chat_v2_safety_fuse.py tests/test_opensearch_store.py tests/test_ingest_opensearch.py tests/test_rag_core_opensearch_hybrid.py tests/test_rag_stats_opensearch.py tests/test_safety_fuse.py -v`

预期：PASS

**步骤 3：运行全量回归**

运行：`pytest -q`
预期：PASS

**步骤 4：清理死代码与空文件引用**

实现要点：
- 删除已无引用文件
- 清理 imports、注释、无效测试
- 确认 `git grep` 不再出现 `/v1/chat`、`storage_sqlite`、`FaissHNSWStore` 等遗留关键字

**步骤 5：提交**

运行：
```bash
git add app tests docs/plans requirements.txt .env.example
git commit -m "feat: align backend implementation with README architecture"
```
