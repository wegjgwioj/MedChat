# Agent Chat SSE 设计说明

**目标：** 为 `/v1/agent/chat_v2` 增加流式返回能力，先把首字节响应时间降下来，并保持现有 JSON 契约和 UI 主流程不被打穿。

**现状：**
- 当前前端通过 `frontend/src/lib/api.ts` 的 `chatV2()` 走一次性 `fetch -> json()`。
- 后端 `app/agent/router.py` 只有同步 JSON 路由 `/v1/agent/chat_v2`。
- 现有 `run_chat_v2_turn()` 返回的是完整结果，没有 token 级别生成流。

## 方案对比

### 方案 A：新增 POST SSE 端点，前端用 fetch 读 event-stream

做法：
- 新增 `POST /v1/agent/chat_v2/stream`
- 请求体沿用当前 `AgentChatV2Request`
- 响应为 `text/event-stream`
- 前端通过 `fetch` + `ReadableStream` 解析 `event:` / `data:`，最终仍落成一个 `AgentChatV2Response`

优点：
- 不需要把 POST 改成 GET
- 不需要把复杂请求参数塞进 querystring
- 可与现有 UI 和类型系统平滑兼容

缺点：
- 浏览器不能直接用原生 `EventSource`
- 需要自己写一层 SSE 解析器

### 方案 B：改成 GET + EventSource

做法：
- 新增 GET 流式端点
- 请求参数全部转 querystring
- 前端用原生 `EventSource`

优点：
- 客户端实现简单

缺点：
- 当前请求体有 `session_id/top_k/top_n/use_rerank`，改成 GET 会让参数组装和兼容逻辑变脆
- 后续如果要带更复杂输入，不适合继续扩展

### 方案 C：维持 JSON，同步返回，前端只做“假流式”加载态

优点：
- 实现最小

缺点：
- 不满足 `重构方案.md` 中对 SSE 的目标
- 只能改善感知，不是协议级流式

## 采用方案

本批次采用 **方案 A**。

原因：
- 对现有调用链侵入最小
- 真正使用 SSE 协议
- 方便后续把中间事件扩展到更细粒度，而不是推翻现有接口

## 事件协议

先定义 4 类事件：

1. `ack`
- 请求已接收
- 尽快返回，降低 TTFB

2. `stage`
- 当前阶段信息
- 第一版只发少量固定阶段，如 `received`、`running`

3. `final`
- 完整 `AgentChatV2Response`
- 前端拿到它后按现有逻辑渲染

4. `error`
- 标准化错误信息

示例：

```text
event: ack
data: {"request_id":"...","session_id":"..."}

event: stage
data: {"phase":"running"}

event: final
data: {"session_id":"...","mode":"answer",...}
```

## 后端设计

### 路由

新增：
- `POST /v1/agent/chat_v2/stream`

保留：
- `POST /v1/agent/chat_v2`

这样前端和外部调用方可以逐步迁移，不会影响现有脚本和测试。

### 实现方式

- 在 `app/agent/router.py` 内新增流式路由
- 用 `StreamingResponse` 返回 `text/event-stream`
- 先同步执行 `run_chat_v2_turn()`，但在真正计算前立即 `yield ack`
- 计算完成后 `yield final`
- 出错时 `yield error`

说明：
- 这一版不是 token 级生成流
- 但已经实现协议级流式和更早的首包返回
- 后续如需更细粒度，可把 graph/LLM 内部阶段逐步暴露为更多 `stage` 事件

## 前端设计

### API 层

在 `frontend/src/lib/api.ts` 新增：
- `chatV2Stream()`

职责：
- 发起 POST 请求
- 校验 `content-type`
- 按 SSE 协议增量解析
- 通过回调把 `ack/stage/final/error` 发给页面

### UI 层

`frontend/src/pages/Chat.tsx` 改造策略：
- 默认改走 `chatV2Stream()`
- `ack` 到达时只更新一个临时“处理中”状态
- `final` 到达后仍使用当前 `UiMessage` 组装逻辑
- 若浏览器/网络不支持流读取，则回退到现有 `chatV2()`

## 测试策略

### 后端

新增或扩展测试覆盖：
- SSE 路由返回 `text/event-stream`
- 至少包含 `ack` 与 `final`
- 出错时输出 `error`
- 鉴权与 JSON 路由保持一致

### 前端

由于当前前端没有单元测试框架，本批次优先做：
- TypeScript 编译验证
- `vite build`

## 并行拆分

可以并行的三块：

1. 后端流式路由与事件格式
2. 前端 API/页面接入
3. 文档与 API 测试

依赖关系：
- 事件协议先由主线定死
- 然后三块可并行推进

## 风险与控制

风险：
- SSE 解析器若处理不稳，会导致前端卡住或重复消息
- 流式端点与旧 JSON 端点若行为漂移，会造成双维护问题

控制：
- `final` 事件 payload 完全复用 `AgentChatV2Response`
- 旧 JSON 路由保留不动
- 测试先锁定 `ack/final/error` 的最小事件集合
