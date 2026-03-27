# healthcare agent

本仓库实现一个“风险分层 + 就医建议”的医疗对话系统，包含后端服务、RAG 证据检索与前端 UI。

## 技术栈

- 后端：FastAPI + Uvicorn
- Agent 编排：LangGraph（将对话流程编排为状态机）
- LLM 接入：LangChain + OpenAI 兼容接口（例如 DeepSeek）
- RAG：Chroma（本地持久化向量库）+ SentenceTransformers（Embedding）
- 前端：React + Vite

## 系统能力（概览）

- 多轮问诊：收集关键信息，决定是否进入分诊
- 分诊：检索证据、生成结构化结果（风险等级/红旗症状/就医建议等）
- 纵向档案摘要：从年龄/既往史/用药/过敏史中生成 `record_summary`，用于后续检索与安全约束
- 记录感知安全护栏：当回答命中既往过敏药物时自动追加警示，并在 triage JSON 中移除高风险动作
- 混合检索：在 dense 候选上叠加本地 sparse BM25 风格评分，提升短症状词/短问句的召回稳定性
- 语义缓存：重复或高度相似的 query 可直接命中进程内缓存，降低重复检索延迟
- OCR 入库：支持远程 URL 解析和本地文件上传，完成后自动入库到向量库
- 引用强制：回答中的 `[E1]` 等引用必须能定位到证据块
- 安全审查：`mode=safe` 会启用更严格的安全审查链
- 可观测：返回 `trace`（`INQUIRY → RETRIEVE → ASSESS → SAFETY`）及耗时，便于调试与问题定位

## HTTP API

- `GET /health`：健康检查
- `POST /v1/chat`：多轮对话入口（前端使用）
- `POST /v1/triage`：单次分诊入口（便于独立调试，可选传 `clinical_record_path` 参与记录安全校验）
- `POST /v1/agent/chat_v2`：LangGraph 医患问诊 Agent（多轮 + 结构化追问 + trace）
- `POST /v1/ocr/ingest`：创建 OCR 任务（URL 或本地文件）
- `GET /v1/ocr/status/{task_id}`：查询 OCR 状态，完成后自动入库

---

## 项目结构与模块边界

### 后端（目录：app/）

- `app/api_server.py`
	- **职责**：HTTP 服务入口；实现 `/health`、`/v1/chat`、`/v1/triage`；把“问诊→检索→评估→安全审查”编排为 LangGraph 状态机。
	- **边界**：只做编排与 I/O（请求/响应 schema、session 落盘、trace 汇总），不在这里实现检索细节/业务规则。

- `app/triage_service.py`
	- **职责**：分诊引擎的主实现（RAG 检索、证据引用校验、结构化输出、guardrails、安全审查链等）。
	- **边界**：对外提供可复用步骤（供 LangGraph 节点调用），并保证单次分诊的接口行为稳定。

- `app/triage_protocol.py`
	- **职责**：分诊协议/输出 schema（例如风险等级、红旗症状、就医建议等字段的定义与约束）。
	- **边界**：只定义 schema 与规则，不处理外部依赖（LLM、向量库）。

- `app/config_llm.py`
	- **职责**：LLM 配置与实例化（从 `.env` 读取 key/base_url/model 等）。
	- **边界**：只负责“如何获得可用的 LLM 客户端”，不包含业务流程与分诊策略。

### RAG（目录：app/rag/）

- `app/rag/retriever.py`
	- **职责**：从本地 Chroma 向量库中检索证据块。
	- **边界**：只负责“给定 query → 返回 evidence”，不生成最终回答。

- `app/rag/ingest_kb.py`
	- **职责**：从 `app/rag/kb_docs/` 构建/更新向量库到 `app/rag/kb_store/`。
	- **边界**：这是离线数据工程脚本；运行成本较高（模型下载与 embedding 计算），与在线服务解耦。

- `app/rag/kb_store/`
	- **职责**：持久化向量库（Chroma）。
	- **边界**：可直接拷贝到新电脑使用，避免重建。

### 前端（React + Vite，目录：frontend/）

- `frontend/src/App.tsx`
	- **职责**：对话 UI；右侧面板展示 `rag_query / evidence / trace`。
	- **边界**：只消费后端返回字段并展示，不做任何分诊推理。

### 测试（目录：tests/）

- `tests/test_api_auth.py`
	- **职责**：接口鉴权与基础 API 行为的回归测试。

---

## 启动方式

前置条件：
- Miniconda/Anaconda
- Node.js（建议 18+）

### 1) 创建 Python 环境

```powershell
conda env create -f environment.yml
conda activate healthcare-agent
```

如果不使用 conda，也可以直接用 pip：

```powershell
python -m pip install -r requirements.txt
```

推荐优先使用项目自带虚拟环境执行验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### 2) 配置 .env

```powershell
copy .env.example .env
```

然后编辑 `.env`，至少填写：
- `DEEPSEEK_API_KEY=...`

### 3) 启动后端

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
uvicorn app.api_server:app --host 127.0.0.1 --port 8000
```

如需启用本地 RAG 证据（向量库不随仓库提交）：
- 把你的 `kb_store`（Chroma 持久化目录）放到本机云盘同步路径
- 在 `.env` 里设置 `RAG_PERSIST_DIR` 为该目录的绝对路径

健康检查：打开 `http://127.0.0.1:8000/health`，应返回 `{"status":"ok"}`。

### 4) 启动前端

```powershell
cd frontend
npm ci
npm run dev
```

打开：`http://127.0.0.1:5173/`。

---

## 环境变量（常用）

- `DEEPSEEK_API_KEY`：必填
- `TRIAGE_API_KEY`：可选，开启接口鉴权（请求需带 `X-API-Key`）
- `OUTPUT_DIR`：默认 `outputs`，保存 `/v1/chat` 的 session 轨迹
- `ALLOW_SAVE_SESSION_RAW_TEXT=1`：可选，落盘保存原文（不推荐，注意隐私）
- `CHAT_SLOT_EXTRACTOR=rules`：可选，强制不用 LLM 抽槽（用于离线测试/稳定性）
- `AGENT_SLOT_EXTRACTOR=rules`：可选，强制 `/v1/agent/chat_v2` 不用 LLM 抽槽（用于离线测试/CI）
- `RAG_HYBRID_ENABLED=1`：可选，开启 dense+sparse 混合排序
- `RAG_HYBRID_ALPHA=0.60`：可选，控制 hybrid 中 dense 权重
- `RAG_CACHE_ENABLED=0|1`：可选，开启进程内语义缓存
- `RAG_CACHE_TTL_SECONDS=300`：可选，缓存 TTL
- `RAG_CACHE_MAX_ENTRIES=128`：可选，缓存最大条目数
- `RAG_CACHE_SIM_THRESHOLD=0.85`：可选，语义缓存相似度阈值
- `clinical_record_path`：`/v1/triage` 可选字段；传入本地病历/摘要文本路径后，会启用基于过敏史的记录感知安全护栏

Windows 兼容性：
- 若出现 OpenMP 冲突（`libiomp5md.dll already initialized`），可设置 `KMP_DUPLICATE_LIB_OK=TRUE` 再启动后端。

本地 smoke 建议：

```powershell
$env:AGENT_SLOT_EXTRACTOR = "rules"
$env:CHAT_SLOT_EXTRACTOR = "rules"
python -m uvicorn app.api_server:app --host 127.0.0.1 --port 8000
```

---

## 验收要点（Agent v2）

- 多轮同 session：前端会自动保存 `session_id`，连续发送多条消息应保持不变。
- 追问交互：当后端返回 `mode=ask` 时，追问面板的按钮/选项点击后发送的是“答案文本”，不会把追问问题句子当成用户输入发回去。
- 防复读护栏：如果手动把追问问题原样复制到输入框并发送，前端会拦截并提示“请直接回答追问内容”。

## 清理与安全建议（重要）

- 不要把 API Key 写进代码或提交到仓库；推荐放在本地 `.env`。
- `outputs/`、`__pycache__/`、`.pytest_cache/` 都是生成物/缓存（已在 `.gitignore` 中忽略）。
- `app/rag/kb_store/`（Chroma 持久化向量库）体积可能很大，默认不提交；克隆后可运行入库脚本重新生成。
- `app/rag/kb_docs/dataset-v2/`、`app/MedDG_UTF8/` 属于数据集/评测数据，默认不提交；需要评测时请自行准备数据后再运行评测脚本。
