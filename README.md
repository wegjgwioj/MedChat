# MedChat

面向患者的智能医疗伴诊与问答系统。当前仓库已经具备多轮问诊 Agent、混合 RAG、OCR 入库、SSE 流式接口、长期记录与基础安全约束等核心能力，但整体定位仍然是“高完成度 MVP + 部分增强能力”，不是“全部目标能力都已稳定收口的最终版”。

## 当前状态

### 已实现

- 多轮 Agent 状态机：`Ask / Answer / Escalate`
- 结构化追问与槽位补全
- 混合检索：dense + sparse + rerank + 阈值过滤 + 缓存
- OCR 任务创建、状态查询、文本提取、自动入库
- 已确认病史/过敏史驱动的用药安全过滤
- `/v1/agent/chat_v2/stream` SSE 流式接口
- 基础自动化测试闭环

### 部分完成

- PII 脱敏与日志最小化
- 会话记忆与长期记录边界
- 安全裁决的规则层 / 模型层可解释性
- 评测脚本与报告聚合

### 仍未形成稳定能力

- 仓库默认不附带正式 MedDG 评测数据
- 完整医疗 NER 管线
- 独立部署的小模型安全裁判体系
- 带正式指标证据的评测闭环

## 仓库结构

```text
app/                    后端主代码（Agent / RAG / OCR / Safety / API）
frontend/               前端界面（React + Vite）
tests/                  pytest 测试
scripts/                评测与自测脚本
docs/plans/             分阶段设计与实施计划
reports/                评测产物输出目录（默认可以为空）
重构方案.md             目标架构与当前落地状态说明
差距表.md               目标能力 vs 当前证据的核对表
```

## 快速开始

### 1. 安装依赖

后端：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

前端：

```powershell
cd frontend
npm install
```

### 2. 配置环境变量

参考 [`.env.example`](.env.example) 新建或修改 `.env`。最少需要：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

如果需要 API Key 鉴权，可设置：

- `TRIAGE_API_KEY`

### 3. 启动后端

```powershell
uvicorn app.api_server:app --host 127.0.0.1 --port 8000 --reload
```

### 4. 启动前端

```powershell
cd frontend
npm run dev
```

默认前端开发地址为 `http://127.0.0.1:5173` 或 `http://localhost:5173`。

## 关键接口

- `GET /health`：健康检查
- `POST /v1/triage`：单轮分诊
- `POST /v1/agent/chat_v2`：多轮 Agent 问诊
- `POST /v1/agent/chat_v2/stream`：SSE 流式返回
- `POST /v1/rag/retrieve`：独立 RAG 检索
- `POST /v1/ocr/ingest`：创建 OCR 任务
- `GET /v1/ocr/status/{task_id}`：查询 OCR 并自动入库

## 测试与验证

后端全量测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

前端构建验证：

```powershell
cd frontend
npm run build
```

## 评测说明

评测脚本位于 `scripts/`，包括：

- `eval_meddg_e2e.py`
- `eval_rag_quality.py`
- `eval_perf.py`
- `eval_run_all.py`

注意：

- 仓库默认只保留 `app/MedDG_UTF8/.gitkeep` 作为占位目录，不附带正式 MedDG 数据文件
- 运行端到端或 RAG 评测时，请通过 `--meddg_path` 提供你本地的 `test.pk/train.pk/dev.pk`
- 评测输出会写入 `reports/`

示例：

```powershell
.\.venv\Scripts\python.exe scripts/eval_run_all.py --base_url http://127.0.0.1:8000 --meddg_path D:\datasets\MedDG\test.pk
```

更多说明见 [app/eval/README_EVAL.md](app/eval/README_EVAL.md)。

## 文档口径

对答辩、汇报或简历，请遵循以下表述：

- 可以说“项目目标是……”
- 可以说“当前已经实现……”
- 不要把目标能力直接写成当前稳定事实
- 不要写没有可复现实验支撑的最终指标

建议结合以下文档一起使用：

- [重构方案.md](重构方案.md)
- [差距表.md](差距表.md)
- [docs/plans/2026-03-30-refactor-gap-closure-plan.md](docs/plans/2026-03-30-refactor-gap-closure-plan.md)
