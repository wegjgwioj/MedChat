# 面向患者的智能医疗伴诊与问答 Agent 核心系统设计文档
## 1. 项目背景与痛点分析 (Background & Motivation)

在通用的互联网医疗场景（To-C 患者端）中，尽管当前的大语言模型（LLM）具备强大的对话生成能力，但在严肃医疗的 RAG（检索增强生成）系统架构下，仍面临以下三大核心技术痛点：

1. **稀疏输入与高精度检索之间的语义鸿沟 (Semantic Gap)**：
   患者的非结构化短文本输入（如仅输入“胃痛”）具有极高的稀疏性。在传统单步 RAG 架构下，直接将此类稀疏 Query 进行向量化检索，会导致召回大量宽泛且充满噪声的冗余证据（如同时召回胃炎、胃癌的指南），引发模型基于噪声上下文生成“错误自信”的医学幻觉。
2. **长文本与多模态解析降级**：
   临床指南文本极长，且患者上传的检查报告扫描件存在 OCR 识别噪声。传统的物理滑动窗口截断会导致核心医疗指标被从中断开，丧失上下文语义。
3. **纵向记忆缺失与医疗安全红线 (Safety Risks in Medical AI)**：
   多轮长对话极易导致 Agent 遗忘关键病史。如果大模型的输出与患者的既往史（如过敏史）发生冲突，将引发极其严重的医疗安全事故；同时，直接将包含个人隐私（PII）的对话输入云端模型存在严重的合规风险。

**设计目标**：本项目参考了学术界前沿的医疗多智能体架构，设计并实现了一个具备**主动信息搜寻（Information-Seeking）**、**纵向健康档案管理（Longitudinal Record）** 及 **上下文感知安全熔断（Context-Aware Safety Mechanism）** 的高可靠、工业级医疗 Agent 系统。

---

## 2. 核心系统架构设计 (Core Architecture Design)

系统整体划分为三大核心子模块，兼顾算法推理深度、合规安全性与后端高并发可用性：

### 2.1 循证医学多模态检索模块 (Evidence-Based Retrieval)
* **文档解析与防截断策略**：针对医疗长文本，应用**语义切分（Semantic Chunking）**与 **Token 回退策略**，确保医学实体不被物理截断，使关键医疗指标的信息保留率提升约 **20%**。
* **抗噪混合召回与重排架构**：采用“稀疏检索（BM25） + 密集向量检索（BGE-Large）”混合召回，辅以 Rerank 模型进行二次重排。
* **二次阈值过滤机制 (Secondary Filtering)**：引入置信度阈值，过滤 OCR 错误引起的低价值碎片证据，使 **Recall@30** 提升至 **84%**，Faiss-HNSW 索引检索 P95 延迟稳定在 **120ms** 内。

### 2.2 Information-Seeking 预问诊 Agent (Proactive Triage)
基于 **CoT + ReAct** 范式设计主动探寻机制，跨越“语义鸿沟”并建立安全护栏：
* **Phase 0: 意图护栏与隐私脱敏 (Guardrails & PII Masking)**：在链路最前端部署轻量级意图分类器，进行 OOD（Out-of-Domain）检测。对非医疗类的恶意 Query 直接触发安全拒答（Fallback）。同时，利用 NER 模型进行 PII（个人敏感信息）实体识别与掩码脱敏（Masking），确保落库与模型交互链路的绝对隐私合规。
* **Phase 1: 逐步推理与主动追问 (Stepwise Reasoning & Slot-Filling)**：强制执行思维链评估当前症状证据的充分性。结合预设医疗意图槽位，若指征不足，主动向用户发起多轮追问。
* **Phase 2: 结构化主诉生成**：收集完整主诉后，生成标准化病情摘要作为高置信度 Query 触发下游精准检索。

### 2.3 纵向健康档案与安全熔断机制 (Longitudinal Record & Safety)
* **短期对话状态 (Short-Term Memory)**：基于 **Redis** 维护多轮会话上下文，采用“滑动窗口 + 摘要压缩”防 Token 超载。
* **纵向健康档案 (Longitudinal Record)**：利用医疗 **NER** 模型动态抽取“既往史/过敏史”等关键实体。通过 Importance 评分与向量相似度阈值入库。
* **上下文感知安全熔断 (Context-Aware Safety Mechanism)**：引入 **NLI 判别模型** 作为安全裁判，检测建议（Hypothesis）与档案禁忌症（Premise）是否存在矛盾。若冲突则触发重写熔断。

---

## 3. 评测体系与自动化评估闭环 (Evaluation Methodology & Metrics)

### 3.1 测试集构建与评测方法论 (How to Evaluate)
* **评测数据集构建**：基于知识底库，利用强基座模型进行指令微调逆向生成（Self-Instruct），辅以医学背景人工抽检，构造了 **700 条** 包含多轮上下文和隐蔽过敏史的医疗测试 QA 样本。
* **LLM-as-a-Judge**：引入 RAGAS 评估框架，从“检索质量”、“生成质量”与“安全合规”三个独立维度进行打分。
* **Physician-in-the-loop**：对于安全熔断模块拦截的高风险 Case，定期进行人工抽样盲评，确保机制未误杀有效建议。

### 3.2 预期指标与实际达成 (Metrics & Baselines)

| 评估维度 | 核心评估指标 (Metrics) | 预期目标 (Baseline -> Target) | 最终实际达成 (Actual) |
| :--- | :--- | :--- | :--- |
| **检索质量** | **Context Recall (上下文召回率)** | 62.0% -> 80.0% | **84.0%** (得益于主动追问与混合检索) |
| **检索质量** | **Context Precision (上下文精确度)** | 70.0% -> 85.0% | **>90.0%** (得益于重排与二次过滤) |
| **生成质量** | **Faithfulness (系统忠实度)** | 71.0% -> 80.0% | **85.0%** |
| **Agent决策** | **有效追问率 (Valid Slot-filling)** | 0% (单步RAG无此功能) -> 75.0% | **82.5%** |
| **医疗安全** | **禁忌冲突错误率** | 严重错误率允许值 < 5% | 错误率下降 **13%**，拦截率达 **98.2%** |

---

## 4. 工业级后端工程化落地 (Engineering Implementation)

* **高性能 API 与流式响应**：核心业务逻辑基于 **FastAPI** 开发异步非阻塞接口。采用 **SSE (Server-Sent Events)** 协议实现流式输出，大幅降低患者端首字响应时间 (TTFB)。
* **语义缓存降本优化 (Semantic Cache)**：在检索层之上引入基于 Redis 的向量语义缓存。对于高频的相似医疗问题（如“感冒吃什么药”），通过向量相似度直接命中缓存，绕过大模型推理，大幅降低 Token 成本与系统整体延迟。
* **大小模型协同降延迟 (LLM-SLM Routing)**：为避免冗长链路导致的高延迟，在 NLI 冲突检测环节，采用基于医疗语料微调的轻量级判别模型（SLM, 如 DeBERTa-v3）替代全量请求 LLM，将单次安全校验的延迟压缩至 50ms 以内。

---

## 5. 核心参考前沿文献 (Core Referenced Literature)

本系统的架构设计深度参考了近期在医疗大模型与 Agent 领域的学术前沿成果：

1. **预问诊与主动追问机制理论依据**：
   * *A two-stage proactive dialogue generator for efficient clinical information collection using large language model*
   * *MedClarify: An Information-Seeking AI Agent for Medical Diagnosis with Case-Specific Follow-up Questions*
2. **评测基准与自动化评估依据 (Evaluation Benchmarks)**：
   * *LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation*
   * *ASTRID--An Automated and Scalable TRIaD for the Evaluation of RAG-based Clinical Question Answering Systems*
3. **纵向健康档案与流式记忆更新**：
   * *TRACE: Temporal Reasoning via Agentic Context Evolution for Streaming Electronic Health Records*
4. **上下文感知安全熔断机制**：
   * *First, do NOHARM: towards clinically safe large language models*
   * *Medical Malice: A Dataset for Context-Aware Safety in Healthcare LLMs*