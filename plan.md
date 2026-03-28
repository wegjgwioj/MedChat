# 重构项目计划（依据重构方案 + 当前代码）

## 已确认决策
- OCR 走 **Mineru 精准解析**（需 `MINERU_TOKEN`）。
- `/v1/ocr/ingest` **统一走 Mineru**（不再直接接受文本）。
- 入库字段先 **仅原始文本/markdown**（不做结构化字段抽取）。
- `/v1/ocr/ingest` 支持 **URL + 文件上传**。
- 通过 `/v1/ocr/status/{task_id}` 查询时 **完成即自动入库**。

## 现状对齐（关键代码）
- 现有 OCR 入库接口：[/app/api_server.py:1124-1171](app/api_server.py#L1124-L1171)
- RAG 检索含 BM25 + 向量 + rerank：[/app/rag/rag_core.py:556-634](app/rag/rag_core.py#L556-L634)
- 会话状态含纵向档案结构：[/app/agent/state.py:98-137](app/agent/state.py#L98-L137)
- Agent 编排与安全分流：[/app/agent/graph.py:1-260](app/agent/graph.py#L1-L260)
