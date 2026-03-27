# MedCaht 后端部署与运行手册（Runbook）

版本：2025-12-23

范围：仅后端（FastAPI + LLM + RAG + LangGraph Agent）。

---

## 1. 运行形态

- 服务入口：FastAPI app 在 [app/api_server.py](../app/api_server.py)
- Agent v2：路由在 [app/agent/router.py](../app/agent/router.py)，核心在 [app/agent/graph.py](../app/agent/graph.py)
- RAG：核心在 [app/rag/rag_core.py](../app/rag/rag_core.py)，KB ingest 在 [app/rag/ingest_kb.py](../app/rag/ingest_kb.py)

---

## 2. 环境准备

### 2.1 Python 依赖

优先使用仓库提供的 `environment.yml` 或 `requirements.txt`：

- [environment.yml](../environment.yml)
- [requirements.txt](../requirements.txt)

conda（示例）：

~~~bash
conda env create -f environment.yml
conda activate medchat
~~~

pip（示例）：

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
~~~

### 2.2 LLM 配置（必需）

实现：见 [app/config_llm.py](../app/config_llm.py)。

必须：

- `DEEPSEEK_API_KEY`

可选：

- `DEEPSEEK_BASE_URL`（OpenAI-compatible base url）
- `DEEPSEEK_MODEL`

Windows PowerShell（示例）：

~~~powershell
$env:DEEPSEEK_API_KEY = "<your_key>"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL = "deepseek-chat"
~~~

bash：

~~~bash
export DEEPSEEK_API_KEY="<your_key>"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
~~~

隐私说明：

- 进入 LLM 的用户文本会先做轻量 PII 脱敏，当前覆盖手机号、身份证号、地址片段。
- 脱敏仅作用于 prompt 入模，不改动原始请求体、RAG 检索文本或会话持久化策略。
- 若使用 `/v1/triage` 的 `clinical_record_path`，请确保该文件仅保存在受控主机上；后端只读取文本并做本地规则校验，不会把原始路径回传给前端。

---

## 3. RAG 知识库入库（Chroma）

### 3.1 数据位置

KB CSV 数据：

- [app/rag/kb_docs/dataset-v2/合并数据-CSV格式](../app/rag/kb_docs/dataset-v2/%E5%90%88%E5%B9%B6%E6%95%B0%E6%8D%AE-CSV%E6%A0%BC%E5%BC%8F)

Chroma persist：

- 默认目录：见 [app/rag/kb_store](../app/rag/kb_store)
- 可用环境变量 `RAG_PERSIST_DIR` 覆盖（见 [app/rag/utils/rag_shared.py](../app/rag/utils/rag_shared.py)）

### 3.2 运行 ingest

PowerShell：

~~~powershell
python -m app.rag.ingest_kb --dataset_dir app/rag/kb_docs/dataset-v2/合并数据-CSV格式
~~~

bash：

~~~bash
python -m app.rag.ingest_kb --dataset_dir app/rag/kb_docs/dataset-v2/合并数据-CSV格式
~~~

进度文件：

- `app/rag/kb_store/ingest_progress.json`

### 3.3 GPU/CPU 切换

设备解析实现：见 [app/rag/utils/rag_shared.py](../app/rag/utils/rag_shared.py)。

常用环境变量：

- `RAG_DEVICE`：cpu/cuda/cuda:0
- `RAG_EMBEDDING_DEVICE`：同上
- `RAG_RERANK_MIN_SCORE`：可选；启用 rerank 时仅保留 `rerank_score >= 阈值`
- `RAG_VECTOR_MAX_SCORE`：可选；检索层先过滤 `score > 阈值` 的证据

验证（服务启动后）：

- `GET /v1/rag/stats` 查看 `device/embed_model/rerank_model/count`（实现见 [app/api_server.py](../app/api_server.py)）

---

## 4. 启动服务

### 4.1 本地开发

PowerShell：

~~~powershell
python -m uvicorn app.api_server:app --host 0.0.0.0 --port 8000
~~~

bash：

~~~bash
python -m uvicorn app.api_server:app --host 0.0.0.0 --port 8000
~~~

本地 smoke 推荐增加：

~~~powershell
$env:AGENT_SLOT_EXTRACTOR = "rules"
$env:CHAT_SLOT_EXTRACTOR = "rules"
python -m uvicorn app.api_server:app --host 127.0.0.1 --port 8000
~~~

### 4.2 健康检查

- `GET http://127.0.0.1:8000/health`

---

## 5. 鉴权（可选）

实现：见 [app/api_server.py](../app/api_server.py) 的 `_auth_guard()`。

- 设置 `TRIAGE_API_KEY` 后：
  - `/v1/triage`、`/v1/chat`、`/v1/rag/retrieve` 需要请求头 `X-API-Key`
  - `/v1/agent/chat_v2` 当前不校验（见 [app/agent/router.py](../app/agent/router.py)）

---

## 6. 常见问题排障

- 服务启动即报错：
  - 检查 `DEEPSEEK_API_KEY` 是否缺失（缺失会在 LLM 初始化时报 RuntimeError，见 [app/config_llm.py](../app/config_llm.py)）

- `/v1/rag/retrieve` evidence 为空：
  - KB 是否已 ingest
  - `department` 是否过滤过严
  - `top_n` 是否过小
  - `RAG_RERANK_MIN_SCORE` / `RAG_VECTOR_MAX_SCORE` 是否设置过严

- 评测脚本跑不通：
  - 先检查 `/health` 与 `/v1/rag/stats`
  - 如启用了鉴权，检查 `X-API-Key`
  - 脚本会先打印 MedDG 样本条数与字段 keys，确认数据格式是否如预期

- `/v1/triage` 返回 `record_conflict`：
  - 检查 `clinical_record_path` 对应文本里是否包含明确过敏史
  - 命中后后端会移除相关高风险 `immediate_actions`，并在 `meta.trace` 写入 `record.safety`
  - 第一阶段只覆盖明确药物过敏与明确药名冲突，不做模糊医学推断

- 资源占用高：
  - 关闭 rerank：设置 `RAG_USE_RERANKER=false`
  - 降低 `top_n`
  - 如需控制噪声而非扩大召回，可优先调小 `RAG_RERANK_MIN_SCORE` / 调大 `RAG_VECTOR_MAX_SCORE`，避免直接把阈值设得过严导致 evidence 清空
