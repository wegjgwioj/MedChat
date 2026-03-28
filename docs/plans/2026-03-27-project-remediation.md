# 项目卡点整改实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 先修复会阻塞开发、测试和演示的基础问题，再补齐 OCR Phase 1 闭环，最后统一依赖与文档口径。

**架构：** 采用“先运行基线、再核心链路、后范围收口”的顺序推进。所有任务优先补自动化验证，避免继续在不稳定基础上叠加功能。

**技术栈：** FastAPI、Pydantic v2、LangChain、Chroma、sentence-transformers、React 19、Vite、TypeScript、pytest、Windows 本地开发环境

---

## 优先级总览

### P0：必须先解决

1. 固定 Python/pytest 的可验证运行基线，消除 Windows 临时目录权限导致的伪失败。
2. 让 RAG 在“空库 / 离线 / 模型未缓存”场景下快速失败或快速返回，不能长时间卡死。
3. 修正 `openai` SDK 版本与代码调用方式不一致的问题，避免查询瘦身逻辑静默失效。

### P1：在 P0 完成后推进

4. 补齐 OCR URL 链路的幂等与任务持久化，避免重复入库。
5. 明确并实现 OCR 文件上传方案；若 Mineru 不支持上传，则调整 Phase 1 范围，不要盲写。
6. 补前端 OCR 交互和构建验证，让前后端都能稳定联调。

### P2：收口与答辩口径

7. 更新 README、`.env.example`、重构文档，统一“设计目标”和“当前实现”。
