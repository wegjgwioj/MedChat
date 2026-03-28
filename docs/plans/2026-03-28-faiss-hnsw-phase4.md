# Faiss-HNSW 检索底座切换（Phase 4）实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 把当前 RAG 主索引从 Chroma 直接切换为 Faiss-HNSW，并同步收敛检索接口、入库产物、测试与后端文档，使实现与 `重构方案.md` 的主干目标一致。

**架构：** 新增唯一的 `FaissHNSWStore` 持久化后端，入库阶段直接写入 `index.faiss + docs.jsonl + meta.json`，检索阶段直接走 HNSW ANN 检索，不再保留 Chroma 主链路。`rag_core.py` 仅依赖统一的 store 接口完成 count / dense search / sparse doc 读取 / 持久化状态读取。

**技术栈：** Python 3.11、faiss-cpu、sentence-transformers、FastAPI、pytest。

---

### 任务 1：先补 Faiss-HNSW store 纯接口测试

**文件：**
- 新建：`tests/test_rag_faiss_store.py`
- 预期修改：`app/rag/faiss_store.py`

**步骤 1：先写失败测试**

覆盖以下行为：

```python
def test_faiss_store_roundtrip_add_search_and_reload(tmp_path):
    ...

def test_faiss_store_search_respects_department_filter(tmp_path):
    ...

def test_faiss_store_get_documents_returns_metadata_rows(tmp_path):
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`py -3.11 -m pytest '.worktrees/project-remediation/tests/test_rag_faiss_store.py' -q`

预期：FAIL，原因是 store 尚未实现。

**步骤 3：编写最小实现**

实现内容：
- 新建 `app/rag/faiss_store.py`
- 提供唯一后端 `FaissHNSWStore`
- 提供：
  - `add_documents(documents)`
  - `similarity_search_with_score(query, k, filter=None)`
  - `get_documents(where=None)`
  - `count()`
  - `persist()`
  - `updated_at()`
- 持久化文件固定为：
  - `app/rag/kb_store/index.faiss`
  - `app/rag/kb_store/docs.jsonl`
  - `app/rag/kb_store/meta.json`

**步骤 4：再次运行测试，确认通过**

运行：`py -3.11 -m pytest '.worktrees/project-remediation/tests/test_rag_faiss_store.py' -q`

预期：PASS

### 任务 2：切换 RAG 检索与入库主链路

**文件：**
- 修改：`app/rag/rag_core.py`
- 修改：`app/rag/ingest_kb.py`
- 修改：`app/rag/utils/rag_shared.py`
- 修改：`requirements.txt`
- 修改：`environment.yml`

**步骤 1：先写失败测试**

补充或调整以下行为：

```python
def test_retrieve_returns_empty_when_store_is_empty(monkeypatch):
    ...

def test_hybrid_search_reads_sparse_docs_from_store(monkeypatch):
    ...

def test_get_stats_reports_faiss_backend(monkeypatch):
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`py -3.11 -m pytest '.worktrees/project-remediation/tests/test_rag_empty_store.py' '.worktrees/project-remediation/tests/test_rag_hybrid.py' -q`

预期：FAIL，失败点集中在旧的 Chroma `_collection` 假设。

**步骤 3：编写最小实现**

实现内容：
- `rag_core.py` 改为只依赖 `FaissHNSWStore`
- 清理 `_collection.count()`、`collection.get()` 这种旧接口依赖
- `ingest_kb.py` 直接写入 Faiss-HNSW 索引，不再构建 Chroma
- `get_stats()` 返回真实持久化状态与索引更新时间

实现约束：
- 不保留双后端切换逻辑
- 不引入兼容分支
- evidence 契约保持不变

**步骤 4：再次运行测试，确认通过**

运行：`py -3.11 -m pytest '.worktrees/project-remediation/tests/test_rag_empty_store.py' '.worktrees/project-remediation/tests/test_rag_hybrid.py' '.worktrees/project-remediation/tests/test_rag_threshold_filter.py' '.worktrees/project-remediation/tests/test_rag_cache.py' -q`

预期：PASS

### 任务 3：同步文档、报告与全量验证

**文件：**
- 修改：`docs/BACKEND_PROJECT_REPORT.md`
- 修改：`docs/BACKEND_FILE_INDEX.md`
- 修改：`docs/BACKEND_API.md`
- 修改：`docs/BACKEND_EVAL.md`
- 修改：`README.md`

**步骤 1：对齐文档**

更新以下事实：
- RAG 主索引为 `Faiss-HNSW`
- 会话存储为 Redis
- phase0 guardrail / record admission / Redis semantic cache / dense+sparse dual-path 已上线
- 不写任何未实现能力

**步骤 2：运行全量测试**

运行：`py -3.11 -m pytest '.worktrees/project-remediation/tests' -q`

预期：PASS

**步骤 3：本地逻辑检查**

检查：
- 入库后 `index.faiss/docs.jsonl/meta.json` 是否生成
- `/v1/rag/stats` 是否返回真实 count / updated_at
- `/v1/agent/chat_v2` trace 中是否仍保留 `rag_stats / phase0_guardrail / record_admission`

**步骤 4：准备后续集成**

运行：

```bash
git -C .worktrees/project-remediation status --short
git -C .worktrees/project-remediation log --oneline -n 5
```

确认当前批次仅包含本轮代码与文档改动，不把 `.env` 与本地演示产物混进去。
