# M2：AgentOrchestration（LangGraph 多轮可追问问诊编排）

本模块在不破坏既有 `/v1/triage`、`/v1/chat` 的前提下，新增一个更适合前后端分离的多轮问诊入口：`/v1/agent/chat_v2`。

## 1. 文字版架构图

```text
Client(前端)
  |
  | POST /v1/agent/chat_v2 {session_id?, user_message, top_k, top_n, use_rerank}
  v
FastAPI Router (app/agent/router.py)
  |
  v
LangGraph Orchestration (app/agent/graph.py)
  |
  +-- N1 SafetyGate      ：Phase 0 护栏 + 红旗症状识别
  +-- N2 MemoryUpdate    ：槽位抽取/更新 session_state + longitudinal record admission
  +-- N3 TriagePlanner   ：判断缺口 -> ask / answer
  +-- N4 RAGRetrieve     ：调用 M1 RAG（Faiss-HNSW + dual-path hybrid + Redis cache）
  +-- N5 AnswerCompose   ：LLM 生成回答（带引用 [E1][E2]）
  +-- N6 PersistState    ：Redis 持久化
```

## 2. 会话状态（Session State）

- 使用 Redis 持久化：由 `AGENT_REDIS_URL` 指定连接
- 仅保存最近 20 轮 messages，避免上下文爆炸
- 额外维护 `longitudinal_records` 与 `record_summary`

## 3. API

### 3.1 POST /v1/agent/chat_v2

入参：

```json
{
  "session_id": "可选，不传则后端生成",
  "user_message": "string",
  "top_k": 5,
  "top_n": 30,
  "use_rerank": true
}
```

出参（固定结构）：

```json
{
  "session_id": "...",
  "mode": "ask"|"answer"|"escalate",
  "next_questions": ["..."],
  "answer": "...",
  "citations": [{"eid":"E1","score":0.1,"department":"内科","title":"...","snippet":"...","source":"...","chunk_id":"...","rerank_score":null}],
  "slots": {"age":30, "sex":"男", "symptoms":["咳嗽"], "duration":"3天", "fever":"unknown", "department_guess":"内科", "red_flags":[]},
  "summary": "...",
  "trace": {"node_order":["SafetyGate"],"timings_ms":{"SafetyGate":1},"phase0_guardrail":{"blocked":false},"rag_stats":{"backend":"faiss-hnsw","hits":3}}
}
```

## 4. 示例（Windows PowerShell）

1）启动服务：

```powershell
uvicorn app.api_server:app --reload
```

2）首轮（不传 session_id，缺槽位会返回追问）：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/agent/chat_v2" `
  -H "Content-Type: application/json" `
  -d '{"user_message":"我咳嗽发热"}'
```

预期：`mode=ask`，并返回 `session_id` 与 1~3 个 `next_questions`。

3）第二轮（携带 session_id 补充信息，进入回答并带 citations）：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/agent/chat_v2" `
  -H "Content-Type: application/json" `
  -d '{"session_id":"上一步返回的session_id","user_message":"我30岁男，咳嗽3天，体温38.5度"}'
```

预期：`mode=answer`，`answer` 非空，`citations` 非空（或 RAG 无命中时为空但 answer 提示“未检索到可靠资料”）。

4）红旗症状（直接 escalate）：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/agent/chat_v2" `
  -H "Content-Type: application/json" `
  -d '{"user_message":"我胸痛并呼吸困难"}'
```

预期：`mode=escalate`，直接给就医建议。

## 5. 常见问题

### 5.1 如何清理 session

最简单方式：用 Redis 客户端删除对应 key，或自行写脚本调用 `RedisSessionStore.delete_session(session_id)`。

### 5.2 为什么日志里看不到完整 user_message

为了避免泄漏隐私，日志最多打印前 100 个字符或 hash。

### 5.3 LLM 不可用怎么办

- 主链路按严格模式执行，不做静默降级。
- 若要定位问题，请检查 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 与模型可用性。
