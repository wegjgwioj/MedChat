# MedCaht 后端评测闭环（模板 + 复现方法）

版本：2026-03-28

约束：

- 本文档只提供“评测方法/模板/复现步骤”。除非你已经跑出报告文件，否则不要在文档里填任何具体数值（禁止虚构）。
- 脚本以仓库现有实现为准，且已对齐真实 API 契约：
  - `/v1/agent/chat_v2` 请求字段为 `user_message`（见 [app/agent/router.py](../app/agent/router.py)）
  - `/v1/rag/retrieve` 响应证据键为 `evidence`（见 [app/api_server.py](../app/api_server.py)）

---

## 1. 评测对象与范围

### 1.1 Agent v2（端到端多轮回放）

- 入口：`POST /v1/agent/chat_v2`（见 [app/agent/router.py](../app/agent/router.py)）
- 核心逻辑：`run_chat_v2_turn()`（见 [app/agent/graph.py](../app/agent/graph.py)）
- 目标：评估 Ask/Answer/Escalate 的分支质量、RAG 引用率、对话轮次与稳定性。

### 1.2 RAG（检索质量）

- 入口：`POST /v1/rag/retrieve`（见 [app/api_server.py](../app/api_server.py)）
- 核心逻辑：`retrieve()`（见 [app/rag/rag_core.py](../app/rag/rag_core.py)）
- 目标：评估召回与 rerank 的证据质量、命中率与可解释性。

### 1.3 性能（并发与延迟）

- 对象：Agent v2 与 RAG 两个端点
- 目标：在 1/5/10（或自定义）并发下的 P50/P95、错误率、超时、吞吐等。

---

## 2. 数据集与自检要求（MedDG）

数据位置（pickle）：

- [app/MedDG_UTF8](../app/MedDG_UTF8)

说明：

- `eval_meddg_e2e.py` 与 `eval_rag_quality.py` 默认会解析 `app/MedDG_UTF8/<split>.pk`。
- 若默认路径不存在，可显式传 `--meddg_path <absolute_or_relative_path>`。
- 若两者都不可用，脚本会直接报出缺失文件路径，不再等到 pickle 打开阶段才失败。

硬性要求（已在脚本里实现）：

- 所有评测脚本运行时必须先打印：
  - dialogs 样本条数
  - turn 的关键字段名（keys）

对应实现文件：

- [scripts/eval_meddg_e2e.py](../scripts/eval_meddg_e2e.py)
- [scripts/eval_rag_quality.py](../scripts/eval_rag_quality.py)
- [scripts/eval_perf.py](../scripts/eval_perf.py)

---

## 3. 复现前置条件

### 3.1 环境变量

- DeepSeek/OpenAI 兼容：
  - `DEEPSEEK_API_KEY`（必需）
  - `DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`（可选）
  - 实现见 [app/config_llm.py](../app/config_llm.py)

- 鉴权（可选）：
  - `TRIAGE_API_KEY`：设置后 `/v1/rag/retrieve` 等需要 `X-API-Key`
  - 见 [app/api_server.py](../app/api_server.py) `_auth_guard()`

- RAG：
  - `RAG_PERSIST_DIR`（默认在 app/rag/kb_store）
  - `RAG_DEVICE`/`RAG_EMBEDDING_DEVICE`（cpu/cuda）
  - `RAG_USE_RERANKER`（true/false）
  - `AGENT_REDIS_URL` / `RAG_REDIS_URL`（开启会话存储与 Redis 语义缓存时必需）
  - 见 [app/rag/utils/rag_shared.py](../app/rag/utils/rag_shared.py)

### 3.2 启动服务

PowerShell：

~~~powershell
python -m uvicorn app.api_server:app --host 0.0.0.0 --port 8000
~~~

bash：

~~~bash
python -m uvicorn app.api_server:app --host 0.0.0.0 --port 8000
~~~

健康检查：

- `GET http://127.0.0.1:8000/health`

---

## 3.3 统一入口（推荐）

脚本：

- [scripts/eval_run_all.py](../scripts/eval_run_all.py)

用途：

- 顺序执行 Agent v2、RAG、Perf 三类评测脚本
- 汇总 `meddg_eval_summary.json`、`rag_eval_summary.json`、`perf_eval.json`
- 生成单一入口产物 `reports/eval_suite_summary.json`

PowerShell：

~~~powershell
python scripts/eval_run_all.py --meddg_path app/MedDG_UTF8/test.pk --base_url http://127.0.0.1:8000 --out_dir reports
~~~

可选跳过：

~~~powershell
python scripts/eval_run_all.py --meddg_path app/MedDG_UTF8/test.pk --base_url http://127.0.0.1:8000 --skip_perf
~~~

---

## 4. 评测一：端到端（Agent v2，多轮回放）

脚本：

- [scripts/eval_meddg_e2e.py](../scripts/eval_meddg_e2e.py)

### 4.1 运行

PowerShell：

~~~powershell
python scripts/eval_meddg_e2e.py --base_url http://127.0.0.1:8000 --split test --limit 50 --top_k 5 --top_n 30 --use_rerank 1
~~~

bash：

~~~bash
python scripts/eval_meddg_e2e.py --base_url http://127.0.0.1:8000 --split test --limit 50 --top_k 5 --top_n 30 --use_rerank 1
~~~

### 4.2 产物

- `reports/meddg_eval_cases.csv`
- `reports/cases.csv`（别名）
- `reports/meddg_eval_summary.json`

新增关注字段：

- case CSV：`evidence_quality_level`、`evidence_quality_reason`
- summary JSON：`evidence_quality_counts`、`low_evidence_rate`

### 4.3 指标模板（自行填充）

- 覆盖率：
  - 样本数：
  - 成功返回率：
  - 失败类型分布（401/400/500/timeout）：

- 对话行为：
  - Ask 比例：
  - Answer 比例：
  - Escalate 比例：
  - 平均 Ask 轮数（每 session）：

- 可解释性：
  - citations 非空比例（Answer 且 evidence>0）：
  - 平均 citations 条数：

- Trace 完整性（建议抽样检查）：
  - `trace.node_order` 为空比例：
  - `trace.timings_ms` 是否含关键节点：SafetyGate/Planner/RAG/Compose/Persist
  - `trace.phase0_guardrail` 是否按预期记录 blocked/label
  - `trace.record_admission` 是否记录 admitted/merged/dropped

证据字段定义：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_trace_start/_trace_end`。

---

## 5. 评测二：RAG 质量（检索/证据）

脚本：

- [scripts/eval_rag_quality.py](../scripts/eval_rag_quality.py)

### 5.1 运行

PowerShell：

~~~powershell
python scripts/eval_rag_quality.py --base_url http://127.0.0.1:8000 --split test --limit 200 --top_k 5 --top_n 30 --use_rerank 1
~~~

bash：

~~~bash
python scripts/eval_rag_quality.py --base_url http://127.0.0.1:8000 --split test --limit 200 --top_k 5 --top_n 30 --use_rerank 1
~~~

### 5.2 产物

- `reports/rag_eval_details.csv`
- `reports/details.csv`（别名）
- `reports/rag_eval_summary.json`

新增关注字段：

- details CSV：`evidence_quality_level`、`evidence_quality_reason`
- summary JSON：`evidence_quality_counts`、`low_evidence_rate`

### 5.3 指标模板（自行填充）

- 命中与质量：
  - 有 evidence 的比例：
  - 平均 evidence 条数：
  - rerank 启用/禁用对比（两次运行对照）：
  - 阈值过滤对比（建议至少对照一组 `RAG_RERANK_MIN_SCORE` / `RAG_VECTOR_MAX_SCORE`）：
  - 低证据比例（返回 evidence 条数 < `RAG_MIN_EVIDENCE`）：
  - `dense_hits/sparse_hits` 分布（验证双路召回是否工作）：
  - `cache_hit/cache_mode/cache_backend` 分布（验证 Redis 语义缓存是否工作）：

- 契约一致性：
  - evidence item 必填字段缺失次数（eid/text/source/chunk_id）：
  - evidence 被阈值过滤后仍返回的比例：

契约证据：见单测 [tests/test_rag_retrieve_contract.py](../tests/test_rag_retrieve_contract.py)。

---

## 6. 评测三：性能/并发

脚本：

- [scripts/eval_perf.py](../scripts/eval_perf.py)

### 6.1 运行

PowerShell：

~~~powershell
python scripts/eval_perf.py --base_url http://127.0.0.1:8000 --limit 100 --concurrency 1 5 10
~~~

bash：

~~~bash
python scripts/eval_perf.py --base_url http://127.0.0.1:8000 --limit 100 --concurrency 1 5 10
~~~

### 6.2 指标模板（自行填充）

- Agent v2：
  - P50/P95 延迟：
  - 错误率：
  - 超时率：

- RAG：
  - P50/P95 延迟：
  - 错误率：

建议将结果保存到 `reports/` 下并在 [reports/README.md](../reports/README.md) 里记录复现命令与机器信息。

---

## 7. 常见失败与排查

- `401 UNAUTHORIZED`：检查 `TRIAGE_API_KEY` 与请求头 `X-API-Key`
- `500 INTERNAL_ERROR`：
  - 检查 LLM key 是否设置（见 [app/config_llm.py](../app/config_llm.py)）
  - 检查 RAG 底座是否已 ingest（见 [app/rag/ingest_kb.py](../app/rag/ingest_kb.py)）
- evidence 为空：可能是 KB 未入库、department 过滤过严、或 top_n 过小
