# 重构差距收口实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 把当前系统从“高完成度 MVP + 部分增强能力”收口到“可答辩、可验收、可持续迭代”的稳定版本，优先补齐安全与隐私、事实权限边界、安全裁决分层、评测证据与文档口径。

**架构：** 采用“先收口高风险主链路，再补可复现证据，最后统一对外口径”的顺序推进。保留现有 `Agent -> RAG -> OCR -> Safety` 主骨架不推翻，只在现有模块上加强边界、测试和证据产物，避免把目标架构和当前实现再次混写。

**技术栈：** Python 3.11、FastAPI、Pydantic v2、LangGraph、pytest、SQLite/Redis 会话存储、现有评测脚本与 `reports/` 产物目录。

---

## 优先级总览

### P0：必须先收口

1. 全链路隐私脱敏与日志最小化
2. 已确认事实 / 待确认事实 / 无权事实的边界固化
3. 规则安全与模型安全裁决的职责分层

### P1：P0 完成后推进

4. 可复现评测闭环与报告产物
5. 文档、README、答辩口径同步

---

### 任务 1：收口 PII 脱敏范围与日志最小化

**文件：**
- 修改：`app/privacy/pii.py`
- 修改：`app/api_server.py`
- 修改：`app/agent/graph.py`
- 修改：`app/agent/router.py`
- 修改：`app/triage_service.py`
- 测试：`tests/test_pii_redaction.py`

**步骤 1：先写失败测试**

```python
def test_redact_pii_for_llm_masks_phone_id_address_and_contact_name():
    from app.privacy.pii import redact_pii_for_llm

    text = "我叫张三，电话13812345678，身份证110101199001011234，住址：北京市朝阳区望京街道88号。"
    redacted = redact_pii_for_llm(text)

    assert "13812345678" not in redacted
    assert "110101199001011234" not in redacted
    assert "望京街道88号" not in redacted
    assert "<PHONE>" in redacted
    assert "<IDCARD>" in redacted
    assert "<ADDRESS>" in redacted
```

再补一条日志侧测试：

```python
def test_safe_log_helpers_do_not_emit_raw_phone_number():
    from app.agent.router import _safe_for_log

    masked = _safe_for_log("联系电话 13812345678，头痛两天")
    assert "13812345678" not in masked
```

**步骤 2：运行测试，确认它按预期失败**

运行：`.\.venv\Scripts\python.exe -m pytest tests/test_pii_redaction.py -v`

预期：
- FAIL
- 现有规则无法覆盖联系人姓名或更多住址变体

**步骤 3：编写最小实现**

在 `app/privacy/pii.py` 中扩展规则，但只补这次确实需要的范围：

```python
_PHONE_RE = ...
_IDCARD_RE = ...
_ADDRESS_RE = ...
_CONTACT_RE = re.compile(r"(?:(?:我叫|姓名|联系人)\s*[:：]?\s*)([^\s，。；;]{2,12})")

def redact_pii_for_llm(text: str) -> str:
    s = str(text or "")
    ...
    s = _CONTACT_RE.sub(lambda m: m.group(0).replace(m.group(1), "<NAME>"), s)
    return s
```

同时统一以下行为：

- `app/api_server.py`、`app/agent/router.py`、`app/agent/graph.py` 的日志辅助函数先做 `redact_pii_for_llm()`，再截断
- `app/triage_service.py` 中任何进入 LLM 的原始用户文本都必须先过脱敏函数
- 不新增“完整原文 debug 日志”开关

**步骤 4：再次运行测试，确认通过**

运行：
- `.\.venv\Scripts\python.exe -m pytest tests/test_pii_redaction.py -v`
- `.\.venv\Scripts\python.exe -m pytest tests/test_api_auth.py tests/test_agent_graph_trace.py -v`

预期：
- PASS
- 既有 API/trace 测试不回归

**步骤 5：提交**

```bash
git add app/privacy/pii.py app/api_server.py app/agent/graph.py app/agent/router.py app/triage_service.py tests/test_pii_redaction.py
git commit -m "feat: tighten pii redaction and log minimization"
```

---

### 任务 2：固化“已确认事实 / 待确认事实 / 无权事实”的写入边界

**文件：**
- 修改：`app/agent/state.py`
- 修改：`app/agent/graph.py`
- 修改：`app/api_server.py`
- 修改：`app/agent/record_index.py`
- 测试：`tests/test_agent_pending_record_confirmation.py`
- 测试：`tests/test_agent_longitudinal_record.py`
- 测试：`tests/test_ocr_api.py`

**步骤 1：先写失败测试**

```python
def test_unconfirmed_ocr_fact_does_not_enter_record_summary(monkeypatch):
    from app.agent.graph import run_chat_v2_turn

    # 准备一个带 pending_record_facts 的 session
    ...

    out = run_chat_v2_turn(session_id="pending-only", user_message="我先不确认这个过敏史")

    assert "青霉素过敏" not in out["summary"]
```

再补一条升级测试：

```python
def test_confirmed_pending_fact_is_promoted_into_longitudinal_records(...):
    ...
    assert any(item.value == "青霉素过敏" for item in session.longitudinal_records)
    assert all(item.status != "pending" for item in session.pending_record_facts if item.value == "青霉素过敏")
```

**步骤 2：运行测试，确认它按预期失败**

运行：
- `.\.venv\Scripts\python.exe -m pytest tests/test_agent_pending_record_confirmation.py -v`
- `.\.venv\Scripts\python.exe -m pytest tests/test_agent_longitudinal_record.py -v`

预期：
- FAIL
- 出现 pending 事实进入摘要或状态迁移不完整的问题

**步骤 3：编写最小实现**

收口规则，只做这三层：

```python
class PendingRecordFact(BaseModel):
    fact_id: str
    category: str
    label: str = ""
    value: str = ""
    source_kind: str = "ocr"
    source_excerpt: str = ""
    status: Literal["pending", "confirmed", "rejected"] = "pending"
```

实现要求：

- `app/api_server.py`
  - OCR 抽取出的事实只写入 `pending_record_facts`
  - `pending` 事实不能直接写入 `record_summary`
- `app/agent/graph.py`
  - `MemoryUpdate` 只在用户明确确认时把 `pending` 升级为 `longitudinal_records`
  - 用户拒绝后标记为 `rejected`
  - `rejected` 事实不能重复追问
- `app/agent/record_index.py`
  - 只消费已确认事实，不读取 `pending_record_facts`

**步骤 4：再次运行测试，确认通过**

运行：
- `.\.venv\Scripts\python.exe -m pytest tests/test_agent_pending_record_confirmation.py tests/test_agent_longitudinal_record.py tests/test_ocr_api.py -v`

预期：
- PASS
- OCR、Agent、长期记录三块一起通过

**步骤 5：提交**

```bash
git add app/agent/state.py app/agent/graph.py app/api_server.py app/agent/record_index.py tests/test_agent_pending_record_confirmation.py tests/test_agent_longitudinal_record.py tests/test_ocr_api.py
git commit -m "feat: enforce confirmed-vs-pending record boundaries"
```

---

### 任务 3：把安全判断拆成“规则层”和“模型裁决层”

**文件：**
- 修改：`app/safety/conflict_judge.py`
- 修改：`app/safety/record_guard.py`
- 修改：`app/safety/medication_safety_guard.py`
- 修改：`app/triage_service.py`
- 测试：`tests/test_conflict_judge.py`
- 测试：`tests/test_record_safety.py`
- 测试：`tests/test_triage_record_safety.py`
- 测试：`tests/test_medication_safety_guard.py`

**步骤 1：先写失败测试**

```python
def test_rule_safety_can_block_without_model_loading(monkeypatch):
    import app.safety.conflict_judge as judge
    monkeypatch.setattr(judge, "_get_judge_model", lambda: (_ for _ in ()).throw(RuntimeError("should not load")))

    # 规则已知的药物过敏冲突，不应该依赖模型才能拦截
    ...
    assert result["blocked_medications"][0]["name"] == "阿莫西林"
```

再补一条 trace 测试：

```python
def test_triage_trace_marks_rule_and_model_safety_paths(...):
    ...
    assert trace["safety"]["rule_checks"] >= 1
    assert "model_judge_used" in trace["safety"]
```

**步骤 2：运行测试，确认它按预期失败**

运行：
- `.\.venv\Scripts\python.exe -m pytest tests/test_conflict_judge.py tests/test_record_safety.py tests/test_triage_record_safety.py tests/test_medication_safety_guard.py -v`

预期：
- FAIL
- 当前实现的职责边界和 trace 字段还不够清晰

**步骤 3：编写最小实现**

目标不是引入新模型，而是把职责分开：

```python
def judge_json_conflicts(answer_json, conflicts):
    # 只做“模型是否确认语义冲突”
    ...

def guard_medication_candidates(candidates, constraints):
    # 只做确定性规则拦截
    ...
```

强约束：

- `medication_safety_guard.py` 负责确定性规则否决
- `conflict_judge.py` 只在“规则不够表达、但需要语义确认”的地方启用
- `triage_service.py` 和 `record_guard.py` 必须在 trace 中区分：
  - `rule_checks`
  - `rule_blocked`
  - `model_judge_used`
  - `model_confirmed`

不做的事：

- 不引入新的 DeBERTa 服务化部署
- 不新增“模型失败时偷偷放行”的隐藏回退

**步骤 4：再次运行测试，确认通过**

运行：
- `.\.venv\Scripts\python.exe -m pytest tests/test_conflict_judge.py tests/test_record_safety.py tests/test_triage_record_safety.py tests/test_medication_safety_guard.py -v`

预期：
- PASS
- trace 中能看出规则层与模型层分别做了什么

**步骤 5：提交**

```bash
git add app/safety/conflict_judge.py app/safety/record_guard.py app/safety/medication_safety_guard.py app/triage_service.py tests/test_conflict_judge.py tests/test_record_safety.py tests/test_triage_record_safety.py tests/test_medication_safety_guard.py
git commit -m "refactor: split rule safety from model safety judge"
```

---

### 任务 4：补齐可复现评测闭环与报告产物

**文件：**
- 修改：`app/eval/README_EVAL.md`
- 修改：`scripts/eval_meddg_e2e.py`
- 修改：`scripts/eval_rag_quality.py`
- 修改：`scripts/eval_perf.py`
- 修改：`scripts/eval_run_all.py`
- 修改：`reports/README.md`
- 新建：`reports/.gitkeep`（如果需要保留空目录）
- 测试：`tests/test_eval_cli_compat.py`
- 测试：`tests/test_eval_run_all.py`
- 测试：`tests/test_eval_evidence_quality.py`

**步骤 1：先写失败测试**

```python
def test_eval_run_all_writes_suite_summary_when_reports_exist(tmp_path):
    ...
    assert (tmp_path / "eval_suite_summary.json").exists()
```

再补一条数据缺失提示测试：

```python
def test_eval_readme_examples_match_current_repo_state():
    # 没有 MedDG 数据时，不应暗示“仓库自带完整 test.pk”
    ...
```

**步骤 2：运行测试，确认它按预期失败**

运行：
- `.\.venv\Scripts\python.exe -m pytest tests/test_eval_cli_compat.py tests/test_eval_run_all.py tests/test_eval_evidence_quality.py -v`

预期：
- FAIL 或语义不一致
- 文档对数据路径和产物的描述过于理想化

**步骤 3：编写最小实现**

实现要求：

- `app/eval/README_EVAL.md`
  - 明确区分“仓库自带脚本”和“外部自行提供 MedDG 数据”
  - 写清楚当前仓库 `app/MedDG_UTF8` 仅为占位目录
- `scripts/eval_run_all.py`
  - 保持现有 CLI 契约
  - 缺少数据时给出清晰错误，不生成误导性成功结论
- `reports/README.md`
  - 固定列出预期产物：
    - `meddg_eval_summary.json`
    - `rag_eval_summary.json`
    - `perf_eval.json`
    - `eval_suite_summary.json`

**步骤 4：再次运行测试，确认通过**

运行：
- `.\.venv\Scripts\python.exe -m pytest tests/test_eval_cli_compat.py tests/test_eval_run_all.py tests/test_eval_evidence_quality.py -v`

如果本机具备评测数据，再补一次：

- `.\.venv\Scripts\python.exe scripts/eval_run_all.py --base_url http://127.0.0.1:8000 --meddg_path <你的实际数据路径>`

预期：
- 测试 PASS
- 有数据时产生真实报告；无数据时明确失败，不伪造指标

**步骤 5：提交**

```bash
git add app/eval/README_EVAL.md scripts/eval_meddg_e2e.py scripts/eval_rag_quality.py scripts/eval_perf.py scripts/eval_run_all.py reports/README.md tests/test_eval_cli_compat.py tests/test_eval_run_all.py tests/test_eval_evidence_quality.py
git commit -m "docs: make evaluation pipeline reproducible and evidence-based"
```

---

### 任务 5：统一 README、重构方案与差距表口径

**文件：**
- 修改：`README.md`
- 修改：`重构方案.md`
- 修改：`差距表.md`

**步骤 1：先做口径核对**

人工检查以下问题：

- 是否把“目标能力”写成“当前稳定事实”
- 是否还保留无证据的最终指标
- 是否明确说明当前测试通过，但评测数据与上线级合规仍未收口

**步骤 2：补充最小事实表**

在 `README.md` 和 `差距表.md` 中统一成三栏：

```md
| 能力 | 当前状态 | 证据 |
| --- | --- | --- |
| Agent Ask/Answer/Escalate | 已实现 | tests/test_agent_graph_trace.py |
| 混合检索与 rerank | 已实现 | tests/test_rag_hybrid.py |
| OCR 自动入库 | 已实现 | tests/test_ocr_api.py |
| 强合规 PII | 部分完成 | app/privacy/pii.py |
| 完整评测闭环 | 未完成 | app/MedDG_UTF8 仅占位 |
```

**步骤 3：明确答辩表述**

`重构方案.md` 只保留三类句式：

- “项目目标是……”
- “当前已实现……”
- “当前尚未形成稳定能力……”

删除或改写：

- 没有复现实验支撑的指标
- 容易被问出代码证据缺失的绝对化表述

**步骤 4：人工复核**

逐项对照以下命令的结果再落文档：

- `.\.venv\Scripts\python.exe -m pytest -q`
- `Get-ChildItem .\app\MedDG_UTF8 -Force`
- `Get-ChildItem .\reports -Recurse -File`

预期：
- 文档内容与仓库现状一致

**步骤 5：提交**

```bash
git add README.md 重构方案.md 差距表.md
git commit -m "docs: align README and refactor status with repository evidence"
```

---

## 最终验收顺序

1. `.\.venv\Scripts\python.exe -m pytest tests/test_pii_redaction.py tests/test_agent_pending_record_confirmation.py tests/test_agent_longitudinal_record.py tests/test_conflict_judge.py tests/test_record_safety.py tests/test_triage_record_safety.py tests/test_eval_cli_compat.py tests/test_eval_run_all.py -v`
2. `.\.venv\Scripts\python.exe -m pytest -q`
3. 如有本机评测数据：`.\.venv\Scripts\python.exe scripts/eval_run_all.py --base_url http://127.0.0.1:8000 --meddg_path <实际数据路径>`
4. 人工核对 `README.md`、`重构方案.md`、`差距表.md` 与仓库现状一致

## 完成定义

满足以下条件才算这轮差距收口完成：

- 用户原始敏感信息不再直接进入主日志与 LLM 输入
- 未确认 OCR/病历事实不再拥有否决权
- 规则安全与模型安全职责分离，trace 可解释
- 评测脚本对“有数据/无数据”两种状态给出诚实结果
- README / 方案文档 / 差距表三份口径一致

计划已完成，并保存到 `docs/plans/2026-03-30-refactor-gap-closure-plan.md`。有两个执行选项：

**1. 当前会话执行** - 我在这个会话里按任务顺序实现并逐步校验

**2. 新会话执行** - 你开一个新会话，按 `executing-plans` 批量推进并在检查点回报

**你选择哪一种？**
