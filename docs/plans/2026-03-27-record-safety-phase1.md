# 纵向档案与记录感知安全护栏（Phase 1）实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 为现有医疗 Agent 增加“纵向档案摘要 + 基于档案的安全冲突拦截”能力，优先覆盖过敏史与禁忌药建议冲突。

**架构：** 在现有 SQLite 会话与槽位体系上新增稳定的 `record_summary`，不引入 Redis/NLI 等重型依赖；同时抽离一个纯函数安全模块，负责从历史记录中提取过敏约束、检测回答中的风险药物，并在 Agent 回答与 triage JSON 两条链路上做最小可解释改写与 trace 记录。

**技术栈：** Python 3.11、FastAPI、Pydantic v2、LangGraph、pytest；保持 ASCII 代码风格，按 TDD 执行。

---

### 任务 1：先补纯函数与状态层测试

**文件：**
- 新建：`tests/test_record_safety.py`
- 修改：`app/agent/state.py`
- 预期新建：`app/safety/record_guard.py`

**步骤 1：先写失败测试**

覆盖以下行为：

```python
def test_build_record_summary_keeps_longitudinal_fields():
    ...

def test_detect_record_conflicts_hits_penicillin_allergy():
    ...

def test_apply_record_conflicts_to_answer_text_appends_warning_once():
    ...

def test_apply_record_conflicts_to_triage_json_rewrites_actions_and_uncertainty():
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_record_safety.py -q`

预期：
- FAIL
- 失败原因应为缺少 `app.safety.record_guard` 或相关函数未实现

**步骤 3：编写最小实现**

实现内容：
- 在 `app/safety/record_guard.py` 新增纯函数：
  - `build_record_summary_from_slots(slots)`
  - `extract_record_constraints(record_text)`
  - `detect_record_conflicts(text, record_text)`
  - `apply_record_conflicts_to_answer_text(answer, conflicts)`
  - `apply_record_conflicts_to_triage_json(answer_json, conflicts)`
- 在 `app/agent/state.py` 为 `AgentSessionState` 增加 `record_summary`

实现约束：
- 第一阶段只做“明确过敏史 -> 明确风险药物名”冲突拦截
- 不做模糊医学推断，不做 NLI
- 改写必须可解释，且不能删除原有免责声明

**步骤 4：再次运行测试，确认通过**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_record_safety.py -q`

预期：PASS

**步骤 5：提交**

```bash
git add app/agent/state.py app/safety/record_guard.py tests/test_record_safety.py
git commit -m "feat: add record-aware safety helpers"
```

### 任务 2：接入 Agent v2 链路

**文件：**
- 修改：`app/agent/graph.py`
- 新建：`tests/test_agent_record_safety.py`

**步骤 1：先写失败测试**

覆盖以下行为：

```python
def test_memory_update_builds_record_summary():
    ...

def test_answer_compose_applies_record_conflict_warning(monkeypatch):
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_agent_record_safety.py -q`

预期：
- FAIL
- 失败原因应为 `record_summary` 未更新，或回答链路未应用记录安全护栏

**步骤 3：编写最小实现**

实现内容：
- `MemoryUpdate` 节点更新 `sess.record_summary`
- `RAGRetrieve` 组装 `rag_query` 时优先拼接 `record_summary`
- `AnswerCompose` 在 LLM/模板回答后调用记录冲突护栏
- 在 `trace` 中增加 `record_conflicts`

实现约束：
- 不改变现有 `mode=ask|answer|escalate` 契约
- 无冲突时不影响回答文本

**步骤 4：再次运行测试，确认通过**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_agent_record_safety.py -q`

预期：PASS

**步骤 5：提交**

```bash
git add app/agent/graph.py tests/test_agent_record_safety.py
git commit -m "feat: wire record-aware safety into agent graph"
```

### 任务 3：接入 triage 服务链路并补文档

**文件：**
- 修改：`app/triage_service.py`
- 新建：`tests/test_triage_record_safety.py`
- 修改：`README.md`
- 修改：`docs/BACKEND_API.md`
- 修改：`docs/BACKEND_DEPLOY_RUNBOOK.md`

**步骤 1：先写失败测试**

覆盖以下行为：

```python
def test_triage_step_build_payload_marks_record_conflict():
    ...

def test_triage_once_with_clinical_record_filters_unsafe_drug_action(monkeypatch, tmp_path):
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_triage_record_safety.py -q`

预期：FAIL

**步骤 3：编写最小实现**

实现内容：
- triage JSON 在安全链后追加记录冲突护栏
- `meta.trace` 或等价可观测字段记录冲突命中
- 文档补充：
  - `record_summary`/记录安全护栏的作用
  - 使用 `clinical_record_path` 触发安全检查的方式
  - 本地 smoke 推荐配置

**步骤 4：再次运行测试，确认通过**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_triage_record_safety.py -q`

预期：PASS

**步骤 5：提交**

```bash
git add app/triage_service.py tests/test_triage_record_safety.py README.md docs/BACKEND_API.md docs/BACKEND_DEPLOY_RUNBOOK.md
git commit -m "feat: add record-aware safety guard to triage service"
```

### 任务 4：总体验证与合并准备

**文件：**
- 修改：`scripts/selftest_agent_v2.py`（仅当新能力需要自测覆盖）
- 修改：相关测试文件（仅在验证暴露缺口时）

**步骤 1：运行定向验证**

运行：

```bash
..\..\.venv\Scripts\python.exe -m pytest tests\test_record_safety.py tests\test_agent_record_safety.py tests\test_triage_record_safety.py -q
..\..\.venv\Scripts\python.exe scripts\selftest_agent_v2.py
```

预期：
- 所有新增测试 PASS
- 自测脚本 PASS

**步骤 2：运行全量测试**

运行：`..\..\.venv\Scripts\python.exe -m pytest -q`

预期：PASS

**步骤 3：做本地代码审阅**

检查：
- 记录冲突逻辑只做显式规则，不做过度医疗推断
- 无冲突时回答不发生额外改写
- trace 中不泄露完整敏感原文

**步骤 4：合并与推送**

运行：

```bash
git status --short
git log --oneline --decorate -n 5
```

确认：
- worktree 分支验证通过
- 准备把 `project-remediation` 合并到 `master`
- 在独立工作树完成 merge，避免污染用户当前 dirty workspace

