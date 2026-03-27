# 前端说明

## 启动

```powershell
npm ci
npm run dev
```

默认访问：`http://127.0.0.1:5173`

## 构建验证

```powershell
npm ci
npm run build
```

## 当前已接入能力

- `/v1/agent/chat_v2`：多轮问诊
- `/v1/ocr/ingest`：支持 URL 解析和本地文件上传
- `/v1/ocr/status/{task_id}`：轮询 OCR 结果，完成后自动入库

## 说明

- 本地文件上传由后端转发到 MinerU 预签名上传地址，不需要前端自行处理对象存储。
- OCR 完成后文本会自动写入后端向量库；前端只负责创建任务和轮询状态。
