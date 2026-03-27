# 混合检索与语义缓存（Phase 2）实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 为当前 RAG 检索链路补上“本地 sparse+dense 混合召回 + 进程内语义缓存”，在不改变现有 evidence 契约的前提下提升短问句检索质量与重复请求延迟。

**架构：** 保持现有 Chroma dense 检索、query slimming、rerank、阈值过滤主干不变；在 dense 候选集上增加 BM25 风格 sparse 评分融合，并在 `retrieve()` 外层增加 TTL/LRU 语义缓存。先不引入 Redis 和离线倒排索引，后续再演进到更重的基础设施方案。

**技术栈：** Python 3.11、FastAPI、Chroma、LangChain、pytest；沿用现有环境变量配置方式，按 TDD 执行。

---

### 任务 1：先补混合检索纯函数测试

**文件：**
- 新建：`tests/test_rag_hybrid.py`
- 预期修改：`app/rag/rag_core.py`

**步骤 1：先写失败测试**

覆盖以下行为：

```python
def test_hybrid_merge_promotes_sparse_match_when_dense_ties():
    ...

def test_hybrid_merge_can_be_disabled_by_env(monkeypatch):
    ...

def test_hybrid_merge_keeps_evidence_contract_fields():
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_rag_hybrid.py -q`

预期：
- FAIL
- 失败原因应为混合检索相关函数或行为尚未实现

**步骤 3：编写最小实现**

实现内容：
- 在 `app/rag/rag_core.py` 增加：
  - token 规范化函数
  - sparse BM25 风格评分函数
  - dense/sparse 融合排序函数
- 让 `retrieve()` 在 dense 候选基础上执行融合，再进入 rerank/threshold

实现约束：
- 不新增对外 API 参数
- 不修改 evidence 字段结构
- `RAG_HYBRID_ENABLED=0` 时完全退回旧路径

**步骤 4：再次运行测试，确认通过**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_rag_hybrid.py -q`

预期：PASS

**步骤 5：提交**

```bash
git add app/rag/rag_core.py tests/test_rag_hybrid.py
git commit -m "feat: add hybrid retrieval ranking"
```

### 任务 2：先补语义缓存测试

**文件：**
- 新建：`tests/test_rag_cache.py`
- 预期修改：`app/rag/rag_core.py`

**步骤 1：先写失败测试**

覆盖以下行为：

```python
def test_retrieve_cache_hits_on_same_request(monkeypatch):
    ...

def test_retrieve_cache_bypasses_when_disabled(monkeypatch):
    ...

def test_retrieve_cache_isolated_by_department(monkeypatch):
    ...

def test_retrieve_cache_supports_semantic_overlap_hit(monkeypatch):
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_rag_cache.py -q`

预期：FAIL

**步骤 3：编写最小实现**

实现内容：
- 在 `app/rag/rag_core.py` 增加进程内缓存对象与 TTL/LRU 管理
- 为 `retrieve()` 增加：
  - cache key 生成
  - exact hit
  - semantic overlap hit
  - bypass 条件
- 保存最近一次检索 meta，供后续 API/trace 读取

实现约束：
- 缓存值只保存最终 evidence 与必要 meta
- 不在日志中输出完整 query
- 不命中时仍走完整检索逻辑

**步骤 4：再次运行测试，确认通过**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_rag_cache.py -q`

预期：PASS

**步骤 5：提交**

```bash
git add app/rag/rag_core.py tests/test_rag_cache.py
git commit -m "feat: add semantic cache for rag retrieval"
```

### 任务 3：接入外部调用面并补文档

**文件：**
- 修改：`app/rag/retriever.py`
- 修改：`app/triage_service.py`
- 修改：`app/agent/graph.py`
- 修改：`docs/BACKEND_API.md`
- 修改：`README.md`

**步骤 1：先写失败测试**

在已有 RAG/Agent 测试中补充以下行为：

```python
def test_agent_trace_exposes_rag_cache_meta(...):
    ...

def test_triage_retrieve_meta_keeps_contract(...):
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests -k "rag or agent" -q`

预期：FAIL，且失败点集中在新 meta 未暴露或契约不匹配

**步骤 3：编写最小实现**

实现内容：
- `app/rag/retriever.py` 保持兼容，只在内部附带可读 meta 访问能力
- `app/triage_service.py` 与 `app/agent/graph.py` 在 trace 中透出 cache/hybrid 命中信息
- 文档补充新增环境变量与返回字段说明

实现约束：
- 不破坏现有 `retrieve()` 返回的 evidence 列表
- meta 缺失时链路仍可正常工作

**步骤 4：再次运行测试，确认通过**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests -k "rag or agent" -q`

预期：PASS

**步骤 5：提交**

```bash
git add app/rag/retriever.py app/triage_service.py app/agent/graph.py docs/BACKEND_API.md README.md
git commit -m "feat: expose rag hybrid and cache metadata"
```

### 任务 4：总体验证与合并准备

**文件：**
- 仅在验证暴露问题时回修对应文件

**步骤 1：运行定向验证**

运行：

```bash
..\..\.venv\Scripts\python.exe -m pytest tests\test_rag_hybrid.py tests\test_rag_cache.py -q
```

预期：PASS

**步骤 2：运行全量测试**

运行：`..\..\.venv\Scripts\python.exe -m pytest -q`

预期：PASS

**步骤 3：本地审阅**

检查：
- `RAG_HYBRID_ENABLED=0` 是否可完全回退
- cache 是否按 `department/top_k/use_rerank` 做隔离
- retrieve 契约字段是否保持稳定
- query 日志是否仍只输出截断版或哈希

**步骤 4：合并与推送**

运行：

```bash
git status --short
git log --oneline --decorate -n 5
```

确认：
- 当前批次验证通过
- 可把 `project-remediation` 合并回 `master`
- 合并动作在独立工作树完成，避免污染主工作区
