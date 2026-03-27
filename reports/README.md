# Reports（评测产物目录）

本目录存放可复现的评测产物（JSON/CSV），用于企业课程设计验收审计。

## 产物列表

- `eval_suite_summary.json`：统一评测入口汇总（串联 C1/C2/C3）
- `meddg_eval_summary.json`：C1 端到端多轮回放汇总
- `meddg_eval_cases.csv`：C1 case 级明细（每段对话/每轮）
- `rag_eval_summary.json`：C2 RAG 离线质量汇总
- `rag_eval_details.csv`：C2 query 级明细
- `perf_eval.json`：C3 性能评测汇总

## 推荐入口

优先使用统一入口，一次生成整套 summary：

```powershell
python scripts/eval_run_all.py --meddg_path app/MedDG_UTF8/test.pk --base_url http://127.0.0.1:8000 --out_dir reports
```

如需跳过某一类评测：

```powershell
python scripts/eval_run_all.py --meddg_path app/MedDG_UTF8/test.pk --base_url http://127.0.0.1:8000 --skip_perf
```

## 复现命令（Windows PowerShell）

先确认测试集（本仓库内置，用于评测）：

- `app/MedDG_UTF8/test.pk`

先启动后端：

```powershell
uvicorn app.api_server:app --host 127.0.0.1 --port 8000
```

然后执行：

```powershell
python scripts/eval_run_all.py --meddg_path app/MedDG_UTF8/test.pk --base_url http://127.0.0.1:8000 --out_dir reports
python scripts/eval_meddg_e2e.py --meddg_path app/MedDG_UTF8/test.pk --n 100 --base_url http://127.0.0.1:8000 --top_k 5 --top_n 30 --use_rerank 1
python scripts/eval_rag_quality.py --meddg_path app/MedDG_UTF8/test.pk --n 200 --top_k 5 --top_n 30 --use_rerank 1 --base_url http://127.0.0.1:8000
python scripts/eval_perf.py --base_url http://127.0.0.1:8000 --concurrency 1,5,10 --requests 20 --meddg_path app/MedDG_UTF8/test.pk
```

提示：如果你想离线运行（不启动后端），可以后续把脚本扩展为“进程内调用 app/rag/retriever.py”。目前验收以“黑盒 HTTP 复现”为主。

## 建议的验收检查点

- C1（E2E）：`meddg_eval_summary.json` 里 `error_rate` 应接近 0；并记录 `avg_latency_ms/p95_latency_ms`。
- C2（RAG）：`rag_eval_summary.json` 里 `coverage_rate` 应 > 0（至少能返回 evidences）；`hit_rate` 用于横向对比参数（top_k/重排/embedding）。
- C3（Perf）：`perf_eval.json` 关注并发 1/5/10 下两条链路的 `rps/p95_ms/error_rate`。
