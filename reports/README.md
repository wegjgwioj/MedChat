# reports 目录说明

本目录用于保存评测产物与阶段性报告。

## 当前约定

- 仓库默认不附带现成评测结果
- 在未运行评测前，本目录可以为空
- 运行评测脚本后，产物默认写入本目录

## 可能出现的文件

- `meddg_eval_summary.json`：MedDG 端到端摘要
- `meddg_eval_cases.csv`：MedDG 逐轮明细
- `rag_eval_summary.json`：RAG 离线摘要
- `rag_eval_details.csv`：RAG 逐条明细
- `perf_eval.json`：性能压测摘要
- `eval_suite_summary.json`：`eval_run_all.py` 聚合摘要

## 注意

- 这些文件只有在本地提供数据并实际执行脚本后才会生成
- 不要把空目录或占位目录理解为“评测已经完成”
