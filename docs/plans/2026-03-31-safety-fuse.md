# 安全熔断统一接入实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 将确认后的档案冲突熔断统一接入 `/v1/triage`、`/v1/chat`、`/v1/agent/chat_v2` 与其 stream 结果。

**架构：** 新增一个统一安全熔断模块，负责收集确认约束、检测冲突、执行确定性删除和必要时的受约束重写；三个入口共享同一套后处理语义和 trace 输出。

**技术栈：** Python 3.11、FastAPI、LangGraph、Pydantic v2、pytest。

---

### 任务 1：补核心安全熔断单元测试

**文件：**
- 新建：`tests/test_safety_fuse.py`
- 修改：`app/safety/__init__.py`
- 修改：`app/safety/confirmed_constraints.py`
- 修改：`app/safety/medication_safety_guard.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- 已确认青霉素过敏时，回答中的阿莫西林建议会被删除
- 删除后会追加明确的安全提醒
- 无确认约束时，不会误删正常回答

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_safety_fuse.py -v`
预期：FAIL，因为统一熔断入口尚不存在

**步骤 3：编写最小实现**

新增统一熔断模块，先实现纯函数级别的约束收集、候选检测、删除与警告拼接。

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_safety_fuse.py -v`
预期：PASS

### 任务 2：将统一熔断接入 triage JSON 链路

**文件：**
- 新建：`tests/test_triage_safety_fuse.py`
- 修改：`app/triage_service.py`
- 修改：`app/safety/record_guard.py`
- 修改：`app/safety/conflict_judge.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- triage 在已确认过敏约束下不会返回冲突药物行动项
- `record_conflicts` 和安全 trace 会写回响应
- 未确认外部病历不会直接触发否决

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_triage_safety_fuse.py -v`
预期：FAIL，因为 triage 主链路尚未调用统一熔断

**步骤 3：编写最小实现**

将 `triage_service.py` 中目前禁用的记录安全后处理改为调用统一熔断入口，并保留“未确认资料不直接否决”的规则。

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_triage_safety_fuse.py -v`
预期：PASS

### 任务 3：将统一熔断接入 `/v1/chat`

**文件：**
- 新建：`tests/test_api_chat_safety_fuse.py`
- 修改：`app/api_server.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- 会话中已有已确认过敏事实时，`/v1/chat` 的最终 `doctor_reply` 不包含冲突药物建议
- trace 中出现统一安全字段

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_api_chat_safety_fuse.py -v`
预期：FAIL，因为旧版 chat 尚未接入统一熔断

**步骤 3：编写最小实现**

在 `/v1/chat` 结果出站前调用统一熔断入口，并保持已有 session 持久化结构兼容。

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_api_chat_safety_fuse.py -v`
预期：PASS

### 任务 4：将统一熔断接入 agent chat v2 与 stream

**文件：**
- 新建：`tests/test_agent_chat_v2_safety_fuse.py`
- 修改：`app/agent/graph.py`
- 修改：`app/agent/router.py`

**步骤 1：先写失败测试**

新增测试覆盖以下行为：
- agent 普通返回会屏蔽冲突药物建议
- stream 最终事件与普通返回共享同一安全结果
- 现有 agent trace 继续保留，并追加统一安全字段

**步骤 2：运行测试，确认它按预期失败**

运行：`pytest tests/test_agent_chat_v2_safety_fuse.py -v`
预期：FAIL，因为 agent 仍使用局部安全实现

**步骤 3：编写最小实现**

把 `graph.py` 中现有的已确认用药安全逻辑收敛到统一熔断入口；`router.py` 只复用最终结果，不新增第二套安全逻辑。

**步骤 4：再次运行测试，确认通过**

运行：`pytest tests/test_agent_chat_v2_safety_fuse.py -v`
预期：PASS

### 任务 5：回归验证

**文件：**
- 修改：`tests/` 下新增测试文件
- 修改：必要的安全模块与入口文件

**步骤 1：运行全量相关测试**

运行：`pytest tests/test_safety_fuse.py tests/test_triage_safety_fuse.py tests/test_api_chat_safety_fuse.py tests/test_agent_chat_v2_safety_fuse.py -v`
预期：PASS

**步骤 2：运行一次基础回归**

运行：`pytest -q`
预期：至少新建测试可被正常收集并通过，不再出现“0 items”的状态

**步骤 3：整理实现**

检查三个入口的安全 trace 命名是否一致，删除重复逻辑，补必要注释。

**步骤 4：提交**

运行：
```bash
git add app tests docs/plans
git commit -m "feat: unify confirmed-record safety fuse"
```
