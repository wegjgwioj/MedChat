# MedCaht 后端 API 文档（仅后端）

版本：2025-12-23

说明：

- 本文档根据仓库真实代码生成，接口以 FastAPI 实现为准。
- 鉴权：当环境变量 `TRIAGE_API_KEY` 设置后，部分接口需要请求头 `X-API-Key` 匹配（见 [app/api_server.py](../app/api_server.py) 的 `_auth_guard()`）。
- `/v1/agent/chat_v2` 当前不走 `_auth_guard`（见 [app/agent/router.py](../app/agent/router.py)），如需公网暴露建议补齐。

---

## 0. 统一约定

### 0.1 Base URL

- 本地默认：`http://127.0.0.1:8000`

### 0.2 通用错误响应

当触发错误处理器时，响应形如：

- `400 BAD_REQUEST`
- `401 UNAUTHORIZED`
- `500 INTERNAL_ERROR`

响应结构（示例）：

~~~json
{
  "code": "BAD_REQUEST",
  "message": "请求参数不合法",
  "trace_id": "<uuid>"
}
~~~

证据：见 [app/api_server.py](../app/api_server.py) 的 exception handlers。

### 0.3 记录感知安全护栏

- `/v1/agent/chat_v2` 会在会话内维护 `record_summary`，优先保留年龄、既往史、用药和过敏史等稳定信息。
- `/v1/triage` 可传 `clinical_record_path`，后端会读取该文本并对回答中的高风险药物建议做记录感知校验。
- 当前第一阶段仅覆盖“明确过敏史 -> 明确风险药物名”冲突拦截；命中后会：
  - 在自由文本回答里追加风险提醒
  - 在 triage JSON 中移除对应 `immediate_actions`
  - 在 `trace` 中记录 `record.safety`

---

## 1) POST /v1/agent/chat_v2（LangGraph Agent v2）

实现位置：

- 路由：`agent_chat_v2()`，见 [app/agent/router.py](../app/agent/router.py)
- 业务：`run_chat_v2_turn()`，见 [app/agent/graph.py](../app/agent/graph.py)

### 1.1 请求（JSON）

Schema（来自 Pydantic 模型 `AgentChatV2Request`）：

- `session_id`：string，可选；不传则后端生成
- `user_message`：string，必填；用户输入
- `top_k`：int，默认 5，范围 [1,20]；最终返回证据条数
- `top_n`：int，默认 30，范围 [1,200]；第一阶段召回条数
- `use_rerank`：bool，默认 true；是否启用 rerank

注意：

- `user_message` 不能为空；否则 `run_chat_v2_turn()` 会抛 ValueError（见 [app/agent/graph.py](../app/agent/graph.py)）。
- 进入 LLM 前会对常见 PII 做脱敏，目前包括手机号、身份证号、地址片段；原始输入不会原样拼进 prompt。

请求示例：

Windows PowerShell：

~~~powershell
$body = @{ session_id = "demo_s1"; user_message = "我头疼两天了"; top_k = 5; top_n = 30; use_rerank = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/agent/chat_v2" -ContentType "application/json" -Body $body
~~~

bash：

~~~bash
curl -sS "http://127.0.0.1:8000/v1/agent/chat_v2" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo_s1","user_message":"我头疼两天了","top_k":5,"top_n":30,"use_rerank":true}'
~~~

### 1.2 响应（JSON）

Schema（来自 Pydantic 模型 `AgentChatV2Response`）：

- `session_id`：string，必有
- `mode`：string，必有；取值为 `ask | answer | escalate`
- `ask_text`：string；Ask 模式下为非空自然追问开场，否则可能为空字符串
- `questions`：array[object]；Ask 模式下为非空，Answer/Escalate 为空数组
- `next_questions`：array[string]；Ask 模式下为非空，Answer/Escalate 为空数组
- `answer`：string；Answer/Escalate 模式下为非空，Ask 模式下为空字符串
- `citations`：array[object]；Answer 且 evidence>0 时通常非空；否则为空数组
- `slots`：object；结构化槽位快照（见 [app/agent/state.py](../app/agent/state.py) 的 `Slots`）
- `summary`：string；由槽位规则生成的短摘要
- `trace`：object；可观测字段（node_order/timings_ms/rag_stats 等）

补充说明：

- `trace.record_conflicts`：若回答命中过敏史冲突，会返回命中的 `matched_term/record_term/message` 列表；无冲突时为空列表或缺省。
- `trace.rag_stats`：当前会补充 `cache_hit/cache_mode/hybrid_enabled/search_query/evidence_quality`，便于观察检索是否命中缓存以及是否开启 hybrid。

字段契约（何时为空）：

- `mode=ask`：
  - `ask_text` 非空
  - `questions` 非空
  - `next_questions` 非空
  - `answer` 为空字符串
  - `citations` 为空数组
- `mode=answer`：
  - `answer` 非空
  - `ask_text` 为空字符串
  - `next_questions` 为空数组
  - `citations`：若 RAG hits>0 则通常非空；若 RAG 失败/无证据则为空
- `mode=escalate`：
  - `answer` 非空（红旗分流模板）
  - `questions/next_questions/citations` 为空

证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_node_safety_gate/_node_triage_planner/_node_rag_retrieve/_node_answer_compose`。

响应示例（精简示意）：

~~~json
{
  "session_id": "demo_s1",
  "mode": "ask",
  "ask_text": "为了更准确判断，我想再确认几个关键信息。",
  "questions": [{"slot":"age","question":"请问你大概多大年龄？","type":"text"}],
  "next_questions": ["请问你大概多大年龄？"],
  "answer": "",
  "citations": [],
  "slots": {"age": null, "sex": "", "symptoms": [], "red_flags": []},
  "summary": "",
  "trace": {"node_order": ["SafetyGate","MemoryUpdate","TriagePlanner","PersistState"], "timings_ms": {"SafetyGate": 0}}
}
~~~

---

## 2) GET /v1/rag/stats（RAG 底座状态）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `rag_stats()`，内部调用 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `get_stats()`。

### 2.1 请求

- 无请求体

Windows PowerShell：

~~~powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/v1/rag/stats"
~~~

bash：

~~~bash
curl -sS "http://127.0.0.1:8000/v1/rag/stats"
~~~

### 2.2 响应

- `collection`：string
- `count`：int
- `persist_dir`：string
- `device`：string（例如 cpu/cuda/cuda:0）
- `embed_model`：string
- `rerank_model`：string|null
- `updated_at`：string

---

## 3) POST /v1/rag/retrieve（独立 RAG 检索）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `rag_retrieve()`，内部调用 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `retrieve()`。

### 3.1 鉴权（可选）

- 如果设置了 `TRIAGE_API_KEY`：必须带 `X-API-Key`。

### 3.2 请求（JSON）

来自 `RagRetrieveRequest`（见 [app/api_server.py](../app/api_server.py)）：

- `query`：string，必填
- `top_k`：int，默认 5，范围 [1,20]
- `top_n`：int，默认 30，范围 [1,200]
- `department`：string|null，可选；科室过滤（严格等值匹配）
- `use_rerank`：bool|null，可选；为 null 时按环境变量 `RAG_USE_RERANKER` 决定

相关环境变量（影响最终 evidence 返回）：

- `RAG_RERANK_MIN_SCORE`：number，可选；启用 rerank 时，仅保留 `rerank_score >= 阈值` 的证据
- `RAG_VECTOR_MAX_SCORE`：number，可选；无论是否启用 rerank，都先过滤 `score > 阈值` 的证据
- `RAG_HYBRID_ENABLED`：bool，可选；默认开启，对 dense 候选执行 sparse+dense 混合排序
- `RAG_HYBRID_ALPHA`：number，可选；hybrid 中 dense 权重，范围 `[0,1]`
- `RAG_CACHE_ENABLED`：bool，可选；开启进程内语义缓存
- `RAG_CACHE_TTL_SECONDS`：int，可选；缓存 TTL
- `RAG_CACHE_MAX_ENTRIES`：int，可选；缓存最大条目数
- `RAG_CACHE_SIM_THRESHOLD`：number，可选；近似 query 复用缓存的相似度阈值

Windows PowerShell（带鉴权示例）：

~~~powershell
$headers = @{ "X-API-Key" = "<your_key>" }
$body = @{ query = "咳嗽发热怎么办"; top_k = 5; top_n = 30; department = $null; use_rerank = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/rag/retrieve" -Headers $headers -ContentType "application/json" -Body $body
~~~

bash：

~~~bash
curl -sS "http://127.0.0.1:8000/v1/rag/retrieve" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your_key>" \
  -d '{"query":"咳嗽发热怎么办","top_k":5,"top_n":30,"department":null,"use_rerank":true}'
~~~

### 3.3 响应（JSON）

来自 `rag_retrieve()` 的返回字典（见 [app/api_server.py](../app/api_server.py)）：

- `query`：string
- `top_k`：int
- `top_n`：int
- `use_rerank`：bool（若请求未显式传，则由环境变量推导）
- `evidence`：array[object]，证据列表
- `evidence_quality`：object（`level/reason/count/...`，与后端统一证据质量口径一致）
- `retrieval_meta`：object，最近一次检索的轻量元信息；当前可能包含 `search_query/cache_hit/cache_mode/hybrid_enabled`
- `stats`：object（collection/count/device/embed_model/rerank_model）

`evidence` 单条字段契约（由单测保障）：

- `eid`：string（E1..Ek 连续）
- `text`：string（非空）
- `source`：string（非空）
- `score`：number
- `rerank_score`：number|null
- `chunk_id`：string
- `metadata`：object（至少包含 department/title/row/source_file）

说明：

- evidence 会先经过分数阈值过滤，再执行 `top_k` 截断与 `eid` 重排。
- 因此当阈值设置较严时，返回条数可能小于请求的 `top_k`，甚至为空数组。

证据：见 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `_normalize_evidence_item()` 与 `retrieve()`；单测见 [tests/test_rag_retrieve_contract.py](../tests/test_rag_retrieve_contract.py)。

--- 

## 3.5 POST /v1/triage（单次分诊）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `triage()`，内部调用 [app/triage_service.py](../app/triage_service.py) 的 `triage_once()`。

### 3.5.1 请求（JSON）

- `user_text`：string，必填
- `top_k`：int，默认 5
- `mode`：string，`fast|safe`
- `clinical_record_path`：string|null，可选；本地病历/记录摘要文本路径

补充说明：

- 当传入 `clinical_record_path` 且文本中包含明确过敏史时，后端会对结构化分诊结果做记录感知安全校验。

### 3.5.2 响应补充字段

- `answer.record_conflicts`：array[object]，仅在命中记录安全冲突时出现
- `answer.uncertainty`：会追加 `record_conflict`
- `meta.trace`：会追加 `record.safety`
- `meta.trace` 中的 `rag.retrieve` 步骤会追加 `cache_hit/cache_mode/hybrid_enabled/search_query`

示例（精简）：

~~~json
{
  "answer": {
    "immediate_actions": [],
    "uncertainty": "record_conflict",
    "record_conflicts": [
      {
        "category": "drug_allergy",
        "record_term": "青霉素过敏",
        "matched_term": "阿莫西林",
        "message": "既往记录提示青霉素过敏，应避免阿莫西林等青霉素类药物。"
      }
    ]
  },
  "meta": {
    "trace": [
      {"step": "record.safety", "status": "conflict", "count": 1}
    ]
  }
}
~~~

---

## 4) OCR 接口（MinerU）

实现位置：

- 路由：见 [app/api_server.py](../app/api_server.py) 的 `ocr_ingest()` 与 `ocr_status()`
- 客户端：见 [app/ocr/mineru_client.py](../app/ocr/mineru_client.py)
- 幂等状态落库：见 [app/agent/storage_sqlite.py](../app/agent/storage_sqlite.py) 的 `ocr_tasks` 表

### 4.1 POST /v1/ocr/ingest

用途：

- 为远程 URL 创建 OCR 任务
- 为本地文件创建 OCR 任务并上传到 MinerU 预签名地址

支持两种请求形式，二选一：

1. JSON（URL 模式）

~~~json
{
  "session_id": "demo_s1",
  "file_url": "https://example.com/report.pdf"
}
~~~

2. multipart/form-data（文件上传模式）

~~~text
session_id=demo_s1
file=<binary>
~~~

注意：

- 本地文件上传不是直接把 multipart 发给 MinerU 解析接口。
- 后端会先调用 MinerU 的 `/api/v4/file-urls/batch` 获取预签名上传地址，再把文件 PUT 上去。

响应字段：

- `session_id`：会话 ID
- `task_id`：MinerU batch/task ID
- `status`：固定为 `pending`
- `trace_id`：MinerU trace_id（若有）
- `source_url`：URL 模式为原始 URL；文件模式为文件名
- `source_kind`：`url | upload`

### 4.2 GET /v1/ocr/status/{task_id}

用途：

- 查询 MinerU OCR 状态
- 首次完成时自动下载 `full_zip_url`，提取文本并写入 Chroma
- 后续重复查询同一 `task_id` 不会重复入库

查询参数：

- `session_id`：可选；通常第一次 ingest 后无需再传
- `source_url`：可选；仅兼容补传

响应字段：

- `task_id`
- `status`
- `done`
- `ingested`
- `session_id`
- `trace_id`
- `picked`：实际选中的结果文件（如 `result.md`）
- `message`：失败或未入库原因

状态说明：

- `done=false`：仍在处理
- `done=true, ingested=true`：已完成并已入库
- `done=true, ingested=false`：任务完成，但未找到有效文本或结果包异常

---

## 4) 兼容/旧接口

### 4.1 POST /v1/triage（单次分诊）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `triage()`，内部调用 `triage_once()`（见 [app/triage_service.py](../app/triage_service.py)）。

说明：

- `user_text` 在进入 LLM 生成分诊 JSON 前会先做 PII 脱敏（手机号、身份证号、地址片段）。

请求（`TriageRequest`）：

- `user_text`：string，必填
- `top_k`：int，默认 5
- `mode`：`fast|safe`，默认 fast，但 API 层默认强制为 safe（除非 localhost 且 `ALLOW_FAST_MODE=1`）。见 [app/api_server.py](../app/api_server.py) 的 `_apply_mode_policy()`。
- `clinical_record_path`：string|null，可选

响应：遵循统一分诊协议（见 [app/triage_protocol.py](../app/triage_protocol.py) 的 `build_triage_payload()`）：

- `answer`：object
- `evidence`：array
- `rag_query`：string
- `meta`：object（mode/created_at；可能额外包含 forced_safe）

### 4.2 POST /v1/chat（多轮对话旧接口）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `chat()`。

请求（`ChatRequest`）：

- `session_id`：string|null，可选
- `patient_message`：string，必填
- `top_k`：int，默认 5
- `mode`：`fast|safe`，默认 fast，但可能被 API 层强制改写为 safe

响应（真实返回字段）：

- `session_id`：string
- `act`：string（INQUIRY/DIAGNOSIS/EXPLANATION/RECOMMENDATION 等，见 [app/api_server.py](../app/api_server.py) 的 `_select_chat_act()`）
- `doctor_reply`：string
- `intake_slots`：object
- `triage`：object|null（为分诊协议 payload 或 null）
- `trace_id`：string
- `debug.trace`：array（LangGraph 编排 trace）

证据：见 [app/api_server.py](../app/api_server.py) 的 `chat()` return 结构。
