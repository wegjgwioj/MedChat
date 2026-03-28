# RAG GPU 入库与验证

本项目的 RAG 索引使用 Faiss-HNSW 持久化到 `app/rag/kb_store/`（默认产物为 `index.faiss/docs.jsonl/meta.json`）。

## 1) 目标

- 默认使用 **BCEmbedding (`maidalun1020/bce-embedding-base_v1`)** 进行 embedding
- `RAG_EMBEDDING_DEVICE=auto` 时：有 GPU 就用 `cuda`，否则用 `cpu`
- 可强制指定：`RAG_EMBEDDING_DEVICE=cuda` 或 `cuda:0`

## 2) 数据隔离（防止评测/总结污染KB）

入库脚本只允许读取以下目录（或其子目录）中的 CSV：

- `app/rag/kb_docs/dataset-v2/合并数据-CSV格式/`

`MedDG_UTF8`、总结文件等不会被扫描入库。

> 如需抽样/分批，可使用 `RAG_CSV_MAX_ROWS` 或把 CSV 放到允许目录的子目录中，并通过 `RAG_KB_DIR` 指向该子目录。

## 3) 安装 GPU 版 torch（示例）

### Windows / Linux

请按你的 CUDA 版本，在 PyTorch 官网选择对应命令安装（不同机器/驱动会不同）：

- https://pytorch.org/get-started/locally/

安装完成后验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

> Windows 备注：如果你遇到 `OMP: Error #15: Initializing libiomp5md.dll...`，说明 OpenMP 运行时冲突。
> 入库/检索脚本已在代码里自动设置 `KMP_DUPLICATE_LIB_OK=TRUE` 以规避；
> 但如果你要单独跑上面的 `python -c "import torch"`，请先在终端设置：
>
> PowerShell：`$env:KMP_DUPLICATE_LIB_OK="TRUE"`
>
> bash：`export KMP_DUPLICATE_LIB_OK=TRUE`

## 4) 运行入库

### Windows PowerShell

```powershell
$env:RAG_RESET="1"
$env:RAG_EMBEDDINGS_PROVIDER="bce"
$env:RAG_EMBEDDING_DEVICE="auto"   # 或 cuda / cuda:0
$env:RAG_COLLECTION="medical_kb"
$env:RAG_INGEST_BATCH_SIZE="256"
$env:RAG_PERSIST_EVERY_N_BATCHES="10"
# 可选：把 allowed 目录的子目录作为 kb_dir
# $env:RAG_KB_DIR="子目录名"

python app/rag/ingest_kb.py
```

### Linux bash

```bash
export RAG_RESET=1
export RAG_EMBEDDINGS_PROVIDER=bce
export RAG_EMBEDDING_DEVICE=auto   # 或 cuda / cuda:0
export RAG_COLLECTION=medical_kb
export RAG_INGEST_BATCH_SIZE=256
export RAG_PERSIST_EVERY_N_BATCHES=10
# 可选：export RAG_KB_DIR=子目录名

python app/rag/ingest_kb.py
```

## 5) GPU 验证（入库期间）

- 终端 1：开始入库
- 终端 2：观察 GPU 占用

```bash
nvidia-smi -l 1
```

同时也可用 Python 验证：

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available())"
```

## 6) 最小自测（检索）

```bash
python -c "from app.rag.retriever import retrieve; print(retrieve('咳嗽发热怎么办',5)[0])"
```

如果能返回包含 `eid/source/text/score` 的字典，说明检索链路可用。

---

## 7) 与旧项目相比：RAG 质量提升点（便于做 PPT）

| 维度 | 旧实现（可能存在） | 当前实现（M1/M0 结合） | 可验证方式 |
|---|---|---|---|
| Embedding 一致性 | 入库/检索可能配置不一致 | 入库与检索共用统一配置（默认 BCE embedding） | 入库与检索日志同时打印 provider/model/device |
| GPU 优先 | 可能未启用或不透明 | `RAG_DEVICE=auto` 自动优先 cuda，否则 cpu | `RAG_DEBUG=1` 输出 device；入库期间 `nvidia-smi -l 1` |
| 两阶段检索 | 只做向量 top_k | Faiss-HNSW dense top_n + sparse 合并 + 可选 BCE rerank | 调整 `RAG_TOP_N`/`RAG_USE_RERANKER` 对比结果 |
| 证据契约 | 字段可能不固定 | 固定字段：eid/text/source/chunk_id/score/rerank_score/metadata | 运行测试 `tests/test_rag_retrieve_contract.py` |
| 可观测性 | 打印信息不统一 | `GET /v1/rag/stats` 返回 backend/count/device/model | `GET /v1/rag/stats` |
