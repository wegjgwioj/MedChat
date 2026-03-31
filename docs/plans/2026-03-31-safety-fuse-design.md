# 安全熔断统一接入设计

**日期：** 2026-03-31

**背景**

当前仓库已经具备三条回答出口：
- `/v1/triage`：单轮分诊 JSON
- `/v1/chat`：旧版多轮问诊 JSON
- `/v1/agent/chat_v2` 与 `/v1/agent/chat_v2/stream`：LangGraph 多轮问诊

README 将“上下文感知安全熔断”描述为核心能力，但现状存在两类问题：
- 主链路上的记录冲突判定没有真正统一接入
- 已有安全能力分散在 `confirmed_constraints`、`medication_safety_guard`、`conflict_judge`、`record_guard` 等模块，行为不一致

**设计目标**

将“确认后的档案冲突熔断”统一接入所有回答出口，满足以下约束：
- 只依据“已确认”的档案事实触发拦截，未确认信息不能直接否决回答
- 以确定性拦截为主，受约束重写为辅
- 三条链路的安全行为和 trace 字段保持一致
- 不改变现有问诊、RAG、OCR 主流程的职责边界

**核心方案**

新增一个统一的安全熔断模块，负责执行四步：
1. 收集已确认约束
2. 检测回答中的冲突候选
3. 删除冲突建议并插入安全提醒
4. 在必要时对过滤后的回答执行受约束重写

**接入方式**

- `triage_service.py`
  对 `answer_json` 的 `immediate_actions`、`what_not_to_do`、`reasoning` 进行统一熔断，并把安全元数据写回 `meta.trace`
- `api_server.py`
  对旧版 `/v1/chat` 最终 `doctor_reply` 接入同一套熔断逻辑，保证旧链路也能阻断冲突药物建议
- `agent/graph.py`
  将现有 `_apply_confirmed_medication_safety()` 升级为调用统一熔断入口，保留其在 agent 回答阶段的上下文能力
- `agent/router.py`
  stream 不做独立安全逻辑，直接复用 `run_chat_v2_turn()` 的最终结果

**数据来源**

- 已确认长期档案：`AgentSessionState.longitudinal_records`
- OCR 进入会话后的待确认事实：只有在用户确认后才能转入拦截来源
- 外部 `clinical_record_path`：仅当后续链路具备确认机制时才能进入强约束；本轮不放开未确认外部资料的否决权

**输出约定**

统一产出安全 trace，至少包含：
- `constraint_count`
- `candidate_count`
- `blocked_count`
- `warning_count`
- `model_judge_used`
- `rewrite_used`
- `blocked_items`

**测试策略**

本轮用 TDD 覆盖三类行为：
- 核心熔断函数：确认约束下会删除冲突药物，并保留安全提醒
- `triage` / `chat`：最终用户可见回答不再包含冲突药物建议
- `agent_chat_v2`：普通返回与 stream 返回共享同一安全结果
