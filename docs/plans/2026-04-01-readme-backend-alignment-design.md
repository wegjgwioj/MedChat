# README 白名单后端对齐设计

**日期：** 2026-04-01

## 背景

当前仓库已经具备若干 README 目标能力的局部实现，包括：

- `OpenSearch` 混合召回主链路
- `agent/chat_v2` 多轮问诊编排
- 纵向健康档案与安全熔断
- Redis 语义缓存

但整体后端仍然处于“目标态与遗留态并存”的中间阶段，主要偏差包括：

- 多轮问诊同时存在新 `agent` 链路与旧 `/v1/chat` 链路
- 短期会话状态并非 README 所述的单一 `Redis`，而是 `Redis 优先 + SQLite fallback + 文件 session`
- 检索、问诊、档案与安全逻辑分散在 `api_server.py`、`agent/graph.py`、`triage_service.py` 等多处
- 仍保留多条 README 未定义的调试或历史兼容接口

用户已明确本轮升级原则：

- 以 `README.md` 为唯一真相
- 采用“严格白名单”执行
- README 未明确支持的后端接口、状态存储、fallback、兼容层都可删除
- 当前系统无前端与线上兼容负担，可直接进行架构收口

## 目标

将当前仓库升级为与 README 一致的**单一目标态后端**：

1. 只保留一条主动追问式多轮问诊主链路
2. 只保留一套 `Redis` 短期记忆
3. 只保留一套 `OpenSearch` 检索主链路
4. 只保留一套纵向档案与安全熔断状态源
5. 删除 README 外的旧接口、旧状态源和静默降级逻辑

本轮不是“补丁式修复”，而是一次有明确删除边界的整仓后端收口。

## 白名单原则

README 对齐后的后端只允许保留以下能力：

### 1. Evidence-Based Retrieval

- 语义切分与 Token 回退
- OCR 噪声鲁棒检索
- `BM25 + dense vector + rerank + 二次阈值过滤`
- Redis 语义缓存

### 2. Information-Seeking Agent

- `Phase 0` 意图护栏与 PII 脱敏
- `Phase 1` 主动追问与槽位补全
- `Phase 2` 结构化主诉生成
- 基于证据与安全约束的回答生成
- SSE 流式输出

### 3. Longitudinal Record & Safety

- 基于 `Redis` 的短期会话状态
- 纵向健康档案实体抽取与更新
- 上下文感知安全熔断

不在上述白名单内、且不是实现这些能力所必需的代码，均视为待删除对象。

## 目标架构

升级完成后的后端划分为五层：

### A. API Boundary

只保留 README 目标态所需的公共入口：

- `/health`
- 单一 agent 问诊入口
- 单一 agent SSE 流式入口

其余历史调试、兼容、旧问诊接口全部删除。

### B. Agent Orchestration

`app/agent/graph.py` 成为唯一问诊编排中心，完整承接：

- Guardrails & PII Masking
- Stepwise Reasoning & Slot-Filling
- Structured Complaint
- Retrieval
- Longitudinal Safety
- Persist

旧 `/v1/chat` 在 `api_server.py` 中的旧 graph、旧 inquiry/triage 状态机、旧 turn 持久化逻辑全部删除。

### C. Memory & Session

会话短期记忆统一为：

- `RedisSessionStore`

严格禁止：

- SQLite fallback
- 文件会话持久化
- 内存会话后备路径

`Redis` 缺失或不可用时直接失败，不再允许运行时静默降级。

### D. Retrieval

检索主后端统一为：

- `OpenSearch`
- `BM25 + dense vector + hybrid fusion`
- 应用层 rerank
- 二次阈值过滤

`Faiss` 与本地 sparse 逻辑从主链路移除，不再保留默认调用路径。

### E. Longitudinal Record & Safety

纵向档案、待确认事实、安全熔断均挂接在同一 agent 状态源上：

- 读取来源：Redis session
- 更新来源：agent / OCR / 记录抽取
- 判定来源：统一 safety 模块

不再允许：

- agent 一套状态
- 旧 `/v1/chat` 一套状态
- OCR 任务 SQLite 一套状态

这类多状态源并存形态。

## 公开接口收口

按 README 白名单，最终公共后端接口只保留：

- `GET /health`
- `POST /v1/agent/chat_v2`
- `POST /v1/agent/chat_v2/stream`

建议删除以下历史或调试接口：

- `POST /v1/chat`
- `POST /v1/triage`
- `GET /v1/rag/stats`
- `POST /v1/rag/retrieve`
- `POST /v1/ocr/ingest`
- `GET /v1/ocr/status/{task_id}`

说明：

- README 描述的是架构能力，而不是这些兼容或调试路由
- 当前系统无兼容负担，因此不需要为旧接口保留桥接层
- OCR 能力可以保留为内部库能力，但不再暴露为 README 外的公共接口

## 数据流

README 对齐后的唯一主数据流如下：

`请求进入 agent 入口`
-> `Phase 0 意图护栏与 PII 脱敏`
-> `从 Redis 读取短期会话`
-> `Phase 1 主动追问 / 槽位补全`
-> `Phase 2 结构化主诉生成`
-> `OpenSearch hybrid retrieval`
-> `rerank + 二次阈值过滤`
-> `纵向档案读取`
-> `安全熔断判定`
-> `生成回答 / 引文 / 追问`
-> `回写 Redis 会话与纵向状态`
-> `同步或 SSE 响应`

关键约束：

- 会话状态只能来自 Redis
- 检索证据只能来自 OpenSearch
- 安全判定只能依赖统一档案状态源

## 删除边界

本轮应删除或彻底重构的对象包括：

### 1. 旧问诊链路

- `/v1/chat`
- 旧 `ChatRequest` / `turns` 文件持久化
- 旧 `_get_chat_graph()` 与 inquiry/triage 流程
- 与旧 chat 专属契约绑定的 helper、测试与注释

### 2. 非 Redis 会话后端

- `storage_sqlite.py`
- `storage_memory.py`
- `build_session_store()` 中的 fallback 逻辑
- `outputs/sessions/*.json`

### 3. 非 OpenSearch 主检索路径

- `FaissHNSWStore` 主调用路径
- 旧本地 sparse 搜索主逻辑
- 任何运行时自动切回 Faiss 的兼容分支

### 4. README 外公共接口

- 旧 triage 调试接口
- RAG stats / retrieve 调试接口
- OCR 公开任务接口

## 失败策略

为保证 README 与代码一致，失败策略统一改为“显式失败”：

- `AGENT_REDIS_URL` 缺失或 Redis 不可用：Agent 主链路直接失败
- OpenSearch 配置缺失或后端不可用：检索主链路直接失败
- 安全模块不可用：不允许绕过安全检查直接输出回答
- 旧接口调用：返回 404，而不是保留兼容桥接

不允许再出现：

- Redis 不可用自动回退 SQLite
- 检索后端不可用自动切回 Faiss
- 旧 `/v1/chat` 与新 agent 并存

## 测试策略

测试也按白名单重写：

### 保留并增强

- `agent/chat_v2` 安全熔断测试
- OpenSearch 检索测试
- retrieval / safety 主链路回归

### 新增

- Redis-only 会话测试
- 缺 Redis / 缺 OpenSearch 的显式失败测试
- 旧接口已移除测试
- README 外 fallback 已移除测试
- API 白名单测试

### 删除

- 依赖 SQLite fallback 的测试
- 依赖 `/v1/chat` 的测试
- 依赖 Faiss 主路径的测试

## 分阶段实施策略

虽然目标是一次性 Big-Bang 收口，但实现顺序仍需控制风险：

1. 先建立 README 白名单测试与删除清单
2. 再切 Redis-only 会话层
3. 再移除旧 `/v1/chat` 与旧 API
4. 再清理非 OpenSearch 主检索路径
5. 最后统一安全、档案与主路由

这样可以确保每一步删除都有测试护栏，而不是最后一次性爆炸。

## 非目标

本轮明确不做：

- 修改 README 文案
- 为历史兼容保留桥接层
- 继续扩展新的调试接口
- 引入第二套会话或检索后端
- 在保留旧实现的前提下“先凑合共存”

## 预期结果

完成后，仓库会从“目标态与遗留态并存”变成“README 驱动的单一后端系统”：

- 单一问诊入口
- 单一 Redis 短期记忆
- 单一 OpenSearch 检索主链路
- 单一纵向档案与安全状态源
- 单一测试口径

README 与代码不再互相打架。
