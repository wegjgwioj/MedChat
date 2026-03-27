# Agent Chat SSE（Phase 3）实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 为 `/v1/agent/chat_v2` 增加可落地的 SSE 流式接口，并让前端优先走流式链路，在不破坏现有 JSON 契约的前提下降低首包等待时间。

**架构：** 保留现有同步 JSON 路由，新增 `POST /v1/agent/chat_v2/stream` 作为 `text/event-stream` 输出；前端新增 `fetch + ReadableStream` 的 SSE 解析器，消费 `ack/stage/final/error` 四类事件。第一版只做协议级流式，不做 token 级 LLM 流。

**技术栈：** Python 3.11、FastAPI、StreamingResponse、React 19、TypeScript、Vite、pytest；按 TDD 执行。

---

### 任务 1：先补后端 SSE 路由测试

**文件：**
- 修改：`tests/test_api_auth.py`
- 预期修改：`app/agent/router.py`

**步骤 1：先写失败测试**

覆盖以下行为：

```python
def test_agent_chat_v2_stream_returns_sse_events(monkeypatch):
    ...

def test_agent_chat_v2_stream_emits_error_event_when_turn_fails(monkeypatch):
    ...
```

**步骤 2：运行测试，确认按预期失败**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_api_auth.py -k "chat_v2_stream" -q`

预期：
- FAIL
- 失败原因应为路由不存在或返回格式不是 `text/event-stream`

**步骤 3：编写最小实现**

实现内容：
- 在 `app/agent/router.py` 新增 `POST /v1/agent/chat_v2/stream`
- 用 `StreamingResponse` 输出 `ack/stage/final/error`
- `final` 事件 payload 复用 `run_chat_v2_turn()` 的完整结果

**步骤 4：再次运行测试，确认通过**

运行：`..\..\.venv\Scripts\python.exe -m pytest tests\test_api_auth.py -k "chat_v2_stream" -q`

预期：PASS

**步骤 5：提交**

```bash
git add app/agent/router.py tests/test_api_auth.py
git commit -m "feat: add sse route for agent chat v2"
```

### 任务 2：先补前端 SSE 客户端与页面接入验证

**文件：**
- 修改：`frontend/src/lib/api.ts`
- 修改：`frontend/src/types/agent.ts`
- 修改：`frontend/src/pages/Chat.tsx`

**步骤 1：先写失败测试或编译约束**

本仓库当前前端无单测框架，因此先用类型约束驱动：

- 增加 `AgentChatV2StreamEvent` 等类型
- 让 `Chat.tsx` 调用新 API 编译通过

**步骤 2：运行构建，确认当前失败**

运行：`npm run build`
目录：`frontend/`

预期：
- 若尚未实现新类型/函数，应出现 TypeScript 编译失败

**步骤 3：编写最小实现**

实现内容：
- `api.ts` 新增 `chatV2Stream()`
- 解析 `text/event-stream`
- `Chat.tsx` 优先调用流式 API
- 无法读流时回退 `chatV2()`

**步骤 4：再次运行构建，确认通过**

运行：`npm run build`
目录：`frontend/`

预期：PASS

**步骤 5：提交**

```bash
git add frontend/src/lib/api.ts frontend/src/types/agent.ts frontend/src/pages/Chat.tsx
git commit -m "feat: wire frontend to agent sse stream"
```

### 任务 3：补 API 文档与最小 smoke 说明

**文件：**
- 修改：`docs/BACKEND_API.md`
- 修改：`README.md`

**步骤 1：先补失败断言或待验证项**

在已有 API 回归基础上，确保文档与真实接口一致，特别是：
- 新增路由
- 事件类型
- 前端回退行为

**步骤 2：编写最小实现**

实现内容：
- 文档补充 `/v1/agent/chat_v2/stream`
- 给出 `curl` / PowerShell 示例
- 说明第一版是协议级流式，不是 token 流

**步骤 3：运行验证**

运行：

```bash
..\..\.venv\Scripts\python.exe -m pytest tests\test_api_auth.py -k "chat_v2_stream" -q
```

预期：PASS

**步骤 4：提交**

```bash
git add docs/BACKEND_API.md README.md
git commit -m "docs: document agent chat sse endpoint"
```

### 任务 4：总体验证

**文件：**
- 若验证暴露问题，再回修对应文件

**步骤 1：运行后端定向测试**

运行：

```bash
..\..\.venv\Scripts\python.exe -m pytest tests\test_api_auth.py tests\test_selftest_agent_v2.py -q
```

预期：PASS

**步骤 2：运行前端构建**

运行：`npm run build`
目录：`frontend/`

预期：PASS

**步骤 3：运行全量测试**

运行：`..\..\.venv\Scripts\python.exe -m pytest -q`

预期：PASS

**步骤 4：本地 smoke**

运行本地服务后，确认：
- 流式端点能快速收到 `ack`
- 最终仍能收到完整 `final`
- 前端聊天主流程不退化

**步骤 5：合并与推送**

在独立工作树完成 merge/push，避免污染主工作区。
