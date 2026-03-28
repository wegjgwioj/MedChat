# MedCaht 后端系统说明书（System Spec）

版本：2026-03-28

本说明书仅覆盖后端系统（FastAPI / M1 RAG / M2 LangGraph Agent / 评测闭环），不包含前端。

---

## 1. 功能清单与边界

### 1.1 功能清单

- 健康检查：`GET /health`（见 [app/api_server.py](../app/api_server.py) 的 `health()`）。
- RAG 服务：
  - `GET /v1/rag/stats`（见 [app/api_server.py](../app/api_server.py) 的 `rag_stats()`）
  - `POST /v1/rag/retrieve`（见 [app/api_server.py](../app/api_server.py) 的 `rag_retrieve()`，内部调用 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `retrieve()`）
- 传统分诊：`POST /v1/triage`（见 [app/api_server.py](../app/api_server.py) 的 `triage()`，内部调用 [app/triage_service.py](../app/triage_service.py) 的 `triage_once()`）
- 多轮对话（旧接口）：`POST /v1/chat`（见 [app/api_server.py](../app/api_server.py) 的 `chat()`，使用 LangGraph 在 API 内做 deterministic 编排）
- LangGraph 多轮问诊 Agent v2：`POST /v1/agent/chat_v2`（见 [app/agent/router.py](../app/agent/router.py) 的 `agent_chat_v2()`，内部调用 [app/agent/graph.py](../app/agent/graph.py) 的 `run_chat_v2_turn()`）

### 1.2 明确边界

- 非诊断：系统输出为信息参考与就医建议，不给出确定诊断结论。
- 免责声明：M2 Answer 输出强制包含免责声明（见 [app/agent/graph.py](../app/agent/graph.py) 的 `_ensure_answer_contract()`）。
- 红旗分流：出现高风险症状时系统应进入 Escalate（建议急诊/线下就医），不继续追问或给出治疗方案（见 `_node_safety_gate()`）。
- Phase 0 护栏：越域请求与 prompt 攻击型输入应在进入槽位抽取前被拦截，并返回医疗问诊范围内的提示（见 `_classify_phase0_guardrail()` 与 `_node_safety_gate()`）。

---

## 2. 三种模式的状态机（Ask / Answer / Escalate）

本节定义 `/v1/agent/chat_v2` 的状态机，真实实现位于 [app/agent/graph.py](../app/agent/graph.py)。

### 2.1 状态定义

- Ask：信息不足，需要结构化追问（返回 `ask_text/questions/next_questions`）。
- Answer：信息足够或属于知识问答直达，进行 RAG 检索与回答生成（返回 `answer/citations`）。
- Escalate：命中红旗（高风险），给出就医升级提示（返回 `answer`，不返回追问/引用）。

### 2.2 状态转移条件（以真实代码为准）

- 安全门（SafetyGate）
  - 条件 1：`_classify_phase0_guardrail(user_message)` 判定为越域请求或 prompt 攻击
    - 动作：直接短路，返回固定拒答模板，并在 `trace.phase0_guardrail` 中记录 `blocked/label/reason`
  - 条件 2：`_looks_like_red_flag(user_message)` 命中关键词列表
    - 动作：`state["mode"] = "escalate"`，并填充 `ESCALATE_ANSWER_TEMPLATE`
  - 证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_node_safety_gate()`。

- 规划器（TriagePlanner）
  - 知识问答直达（kb_qa）：当 `_looks_like_kb_question(user_message)` 为真时
    - 动作：`mode=answer`，并清空追问字段，`trace.planner_strategy="kb_qa"`
    - 证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_node_triage_planner()`；单测见 [tests/test_agent_kb_qa_bypass.py](../tests/test_agent_kb_qa_bypass.py)。
  - 槽位不足：`_slots_sufficient_for_answer(slots)` 为假时
    - 动作：`mode=ask`，构造结构化追问 `questions` 与兼容字段 `next_questions`
    - 证据：见 `_node_triage_planner()`。
  - 槽位充足：`_slots_sufficient_for_answer(slots)` 为真时
    - 动作：`mode=answer`，进入 RAG 节点
    - 证据：见 `_node_triage_planner()` 与 `_route_after_planner()`。

### 2.3 状态机结构（节点与边）

- Entry：`SafetyGate`
- Edges：
  - `SafetyGate -> PersistState`（Escalate）
  - `SafetyGate -> MemoryUpdate`（非 Escalate）
  - `MemoryUpdate -> TriagePlanner`
  - `TriagePlanner -> PersistState`（Ask / Escalate）
  - `TriagePlanner -> RAGRetrieve`（Answer）
  - `RAGRetrieve -> AnswerCompose -> PersistState -> END`

证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_get_graph()`、`_route_after_safety()`、`_route_after_planner()`。

---

## 3. 数据流与控制流说明

### 3.1 `/v1/agent/chat_v2` 数据流

1) 输入：`session_id?` + `user_message` + RAG 参数（`top_k/top_n/use_rerank`）

2) 会话加载：

- 从 Redis 读取 `AgentSessionState`（messages/slots/summary/longitudinal_records）。
- 连接由 `AGENT_REDIS_URL` 决定。

证据：见 [app/agent/storage.py](../app/agent/storage.py) 与 [app/agent/storage_redis.py](../app/agent/storage_redis.py)。

3) 槽位与摘要：

- `MemoryUpdate` 将用户消息追加到 `messages`（只保留最近 20 轮），并抽取槽位后合并。
- `summary` 使用规则生成（避免额外 LLM 带来的不确定性）。
- 稳定病史会写入 `longitudinal_records`，再聚合为 `record_summary` 参与后续检索与安全校验。

证据：见 [app/agent/state.py](../app/agent/state.py) 的 `AgentSessionState.append_message()`、`trim_messages()`、`build_summary_from_slots()`，以及 [app/agent/record_index.py](../app/agent/record_index.py)。

4) RAG：

- `trace.planner_strategy == "kb_qa"` 时，构造 `rag_query = "问题：" + user_message`，直接走知识问答检索。
- 常规问诊路径会先根据 slots 生成结构化 `chief_complaint`，再构造 `rag_query = record_summary + "；主诉：" + chief_complaint`；若主诉为空才退回 `summary`。
- 调用 [app/rag/retriever.py](../app/rag/retriever.py) 兼容层最终落到 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `retrieve()`。

证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_node_rag_retrieve()`。

5) 回答生成：

- evidence 质量不足：不调用 LLM，走低证据回答模板，并继续执行记录冲突检测。
- evidence 非空：调用 LLM 生成回答，并强制加入引用行与免责声明。

证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_node_answer_compose()` 与 `_ensure_answer_contract()`。

6) 持久化：

- 将更新后的 session 写回 Redis。
- trace 中返回 `storage.type/key_prefix/redis_url`。

证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_node_persist_state()`。

### 3.2 `/v1/rag/retrieve` 控制流

- `api_server.rag_retrieve()` 负责鉴权、日志安全（query 前 100 字或 hash）、并调用 `rag_core.retrieve()`。
- 响应返回 `evidence` 与 `stats`（backend/collection/count/device/embed_model/rerank_model）。

证据：见 [app/api_server.py](../app/api_server.py) 的 `rag_retrieve()`。

---

## 4. 日志与隐私脱敏策略

### 4.1 日志策略

- 统一 trace_id：middleware 为每个请求生成 trace_id 并写入日志 record（`trace_id` 字段）。
  - 证据：见 [app/api_server.py](../app/api_server.py) 的 `add_trace_id_middleware`、`_TraceIdFilter`。
- 避免明文用户隐私：
  - `/v1/rag/retrieve` 日志打印 query 的安全摘要（最多 100 字，超长则附 sha256）。
    - 证据：见 [app/api_server.py](../app/api_server.py) 的 `_safe_query_for_log()`。
  - `/v1/agent/chat_v2` 日志同样最多 100 字或 hash。
    - 证据：见 [app/agent/router.py](../app/agent/router.py) 的 `_safe_for_log()`。

### 4.2 会话落盘策略

- `/v1/chat` 默认仅保存 meta（present/length/sha256），除非 `ALLOW_SAVE_SESSION_RAW_TEXT=1`。
  - 证据：见 [app/api_server.py](../app/api_server.py) 的 `chat()`、`_text_meta()`。
- `/v1/agent/chat_v2` 持久化到 Redis：保存最近 20 轮 message（最长 4000 字）。
  - 证据：见 [app/agent/state.py](../app/agent/state.py) 的 `AgentSessionState.trim_messages()`。

---

## 5. 错误处理与可观测 trace 字段定义

### 5.1 HTTP 层错误格式

- 400：请求参数不合法（RequestValidationError / ValueError）
- 401：鉴权失败（`code=UNAUTHORIZED`，附 trace_id）
- 500：内部错误（`code=INTERNAL_ERROR`，附 trace_id）

证据：见 [app/api_server.py](../app/api_server.py) 的 exception handlers。

### 5.2 `/v1/agent/chat_v2` trace 字段

- `trace.node_order: string[]`
  - 节点执行顺序（包含：SafetyGate/MemoryUpdate/TriagePlanner/RAGRetrieve/AnswerCompose/PersistState）。
- `trace.timings_ms: { [node: string]: int }`
  - 每节点耗时（毫秒）。
- `trace.phase0_guardrail?: { blocked: bool, label?: string, reason?: string }`
  - 链路最前端护栏结果。
- `trace.record_admission?: { admitted?: int, merged?: int, dropped?: int, total_records?: int }`
  - 纵向档案准入结果。
- `trace.rag_stats: { hits?: int, latency_ms?: int, device?: string, collection?: string, count?: int, backend?: string, cache_hit?: bool, cache_mode?: string, cache_backend?: string, dense_hits?: int, sparse_hits?: int }`
  - RAG 节点的统计信息。
- `trace.planner_strategy?: string`
  - 规划器策略（如 kb_qa）。
- `trace.chief_complaint?: string`
  - 常规问诊路径下生成的结构化主诉，用于检索和前端调试面板展示。
- `trace.storage?: { type: string, redis_url?: string, key_prefix?: string }`
  - 存储后端信息。

证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_trace_start/_trace_end/_node_rag_retrieve/_node_persist_state`，以及单测 [tests/test_agent_graph_trace.py](../tests/test_agent_graph_trace.py)。
