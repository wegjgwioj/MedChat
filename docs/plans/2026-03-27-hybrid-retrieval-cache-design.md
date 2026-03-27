# 混合检索与语义缓存设计说明

**目标：** 在不引入 Redis 和 Faiss 新基础设施的前提下，为当前 RAG 链路补上可验证的混合召回与语义缓存能力，优先提升医疗短问句场景下的召回质量与重复问题响应延迟。

**现状：**
- 当前 `app/rag/rag_core.py` 只有 Chroma 向量召回 + 可选 rerank，没有稀疏召回分支。
- 当前没有缓存层，重复问法仍会完整走瘦身、检索、重排链路。
- `triage_service` 与 `agent.graph` 都直接依赖 `app.rag.retriever.retrieve()`，因此改造必须保持现有 evidence 契约稳定。

## 方案对比

### 方案 A：本地轻量混合检索 + 进程内语义缓存

做法：
- 继续保留现有 Chroma dense 检索。
- 在检索结果候选集上增加一个本地 sparse 评分分支，按 query-token 与文档 token 的 BM25 风格分数做融合。
- 新增进程内 TTL + LRU 的语义缓存，缓存最终 evidence 列表与检索元信息。

优点：
- 不改部署架构，不引入外部服务。
- 可以在现有测试体系里快速落地并验证。
- 失败面小，适合当前项目的增量重构。

缺点：
- sparse 分支不是完整离线倒排索引，规模继续扩大时上限一般。
- 语义缓存只在单进程内生效，不能跨实例共享。

### 方案 B：直接接入 `rank_bm25` + Redis 语义缓存

做法：
- 增加第三方 BM25 库建立本地稀疏索引。
- 缓存层改为 Redis，支持跨进程复用。

优点：
- 更贴近 `重构方案.md` 的最终目标形态。
- 缓存与检索层更容易演进到线上多实例。

缺点：
- 改造面大，需要补依赖、初始化逻辑、部署文档与运维配置。
- 当前项目验证成本显著上升，不适合作为这一批最先落地的改造。

### 方案 C：只做缓存，不动检索

做法：
- 保持现有 dense + rerank，只增加语义缓存。

优点：
- 改造最小，容易上线。

缺点：
- 无法解决 `重构方案.md` 中最核心的“语义鸿沟”问题。
- 对短问句、症状词这类输入的检索质量提升有限。

## 采用方案

本批次采用 **方案 A**。

原因：
- 它能在当前代码结构里最直接补齐 `2.1` 和 `4` 的核心缺口。
- 保持 `retrieve()` 对外契约不变，能同时兼容 triage 与 agent 两条链路。
- 后续若要升级到 Redis / 真正 BM25 倒排索引，也可以在这一层继续替换实现，而不是推翻接口。

## 详细设计

### 1. 混合检索

新增一层候选融合，流程保持：

1. 原始 query 进入现有 query slimming。
2. Dense 分支仍通过 Chroma `similarity_search_with_score()` 拉取 `top_n` 候选。
3. Sparse 分支不单独查库，而是在 dense 候选集上计算 token-based BM25 风格分数。
4. 通过加权融合分数重新排序，再进入可选 rerank。
5. 保持现有阈值过滤与 evidence 契约。

采用这个结构的原因：
- 不需要新增索引构建流程。
- 能在候选有限的前提下纠正“症状词和医学术语部分重合但向量距离不稳定”的问题。
- 仍然兼容已有 rerank 与 score threshold 逻辑。

配置项规划：
- `RAG_HYBRID_ENABLED`：默认开启。
- `RAG_HYBRID_ALPHA`：控制 dense / sparse 融合权重。
- `RAG_HYBRID_TOKEN_MIN_LEN`：控制参与 sparse 评分的最小 token 长度。

### 2. 语义缓存

缓存层放在 `retrieve()` 外层，命中后直接返回最终 evidence 列表。

缓存 key 组成：
- 规范化后的 query
- `top_k`
- `top_n`
- `department`
- `use_rerank`
- 当前检索关键配置摘要

缓存值：
- `items`
- `meta`，包括 `cache_hit`、`cached_at`、`search_query`

命中策略：
- 先尝试精确 key 命中。
- 若未命中，再在有限缓存条目上比较 query token overlap，相似度达到阈值则命中。
- 对空 query、不合法参数、显式关闭缓存的请求直接 bypass。

配置项规划：
- `RAG_CACHE_ENABLED`
- `RAG_CACHE_TTL_SECONDS`
- `RAG_CACHE_MAX_ENTRIES`
- `RAG_CACHE_SIM_THRESHOLD`

### 3. 可观测性与兼容

新增轻量可观测字段，但不破坏已有返回结构：
- 在内部保存最近一次检索 meta，供 API 与 trace 读取。
- `/v1/rag/retrieve` 可追加 `meta.cache_hit`、`meta.search_query`、`meta.hybrid_enabled`。
- `agent.graph` 与 `triage_service` 可以逐步接入这些 meta，但本批次先保证不影响现有调用契约。

### 4. 风险与控制

风险：
- sparse 评分过强会把 dense 召回顺序打乱，影响已有 rerank 结果。
- 语义缓存若 key 设计过粗，可能把不同科室或不同参数请求错命中。

控制：
- 所有融合和缓存能力都走显式环境变量，可快速关闭。
- 先用测试锁定以下行为：混合排序、缓存命中、缓存绕过、department 隔离、契约稳定。
- 默认只缓存最终证据，不缓存用户完整原文到日志。
