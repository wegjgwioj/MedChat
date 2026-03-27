# -*- coding: utf-8 -*-
"""selftest_agent_v2.py

命令行自测 M2：/v1/agent/chat_v2

用法：
  python scripts/selftest_agent_v2.py

说明：
- 该脚本通过 HTTP 调用 FastAPI 服务，因此需要你先启动：
  uvicorn app.api_server:app --reload
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import requests


BASE = "http://127.0.0.1:8000"
_SLOT_REPLY = {
    "age": "我24岁。",
    "sex": "我是男。",
    "symptoms": "主要症状是头痛。",
    "duration": "已经持续两天了。",
    "severity": "严重程度6/10。",
    "fever": "没有发烧。",
    "location": "主要在头部。",
    "meds": "目前没有用药。",
    "allergy": "无药物过敏史。",
    "history": "既往无特殊病史。",
    "pregnancy": "不适用。",
}


def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{BASE}/v1/agent/chat_v2", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def _build_reply_from_questions(questions: Any) -> str:
    parts = []
    seen = set()
    if isinstance(questions, list):
        for q in questions:
            if not isinstance(q, dict):
                continue
            slot = str(q.get("slot") or "").strip()
            if not slot or slot in seen:
                continue
            seen.add(slot)
            reply = _SLOT_REPLY.get(slot)
            if reply:
                parts.append(reply)
    if not parts:
        parts.append("主要症状是头痛，已经两天了，严重程度6/10，没有发烧，目前没有用药，无药物过敏史。")
    return " ".join(parts)


def _is_valid_escalate_trace(trace: Any) -> bool:
    if not isinstance(trace, dict):
        return False
    order = trace.get("node_order")
    if not isinstance(order, list):
        return False
    required = {"SafetyGate", "PersistState"}
    return required.issubset(set(str(x) for x in order))


def main() -> int:
    print("[M2] Step1: 第1轮（new session），头痛两天 -> 应 ask（结构化追问）")
    out1 = _post({"user_message": "我头疼两天了怎么办"})
    print(json.dumps(out1, ensure_ascii=False, indent=2))
    sid = out1.get("session_id")
    if not sid:
        print("[M2] FAIL: 未返回 session_id")
        return 1
    if out1.get("mode") != "ask":
        print("[M2] FAIL: Step1 期望 mode=ask")
        return 1
    if not (out1.get("ask_text") or "").strip():
        print("[M2] FAIL: Step1 期望 ask_text 非空")
        return 1
    if not (out1.get("questions") or out1.get("next_questions")):
        print("[M2] FAIL: Step1 期望 questions 或 next_questions 非空")
        return 1
    tr1 = out1.get("trace") or {}
    if not (tr1.get("node_order") or []):
        print("[M2] FAIL: Step1 期望 trace.node_order 非空")
        return 1
    if not isinstance(tr1.get("timings_ms"), dict):
        print("[M2] FAIL: Step1 期望 trace.timings_ms 为 dict")
        return 1

    print("\n[M2] Step2-StepN: 根据 questions 自动补槽位，直到进入 answer")
    current = out1
    out_answer: Dict[str, Any] = {}
    for step in range(2, 7):
        if current.get("mode") == "answer":
            out_answer = current
            break
        questions = current.get("questions") or []
        reply = _build_reply_from_questions(questions)
        current = _post({"session_id": sid, "user_message": reply, "top_k": 3, "top_n": 30, "use_rerank": True})
        print(json.dumps(current, ensure_ascii=False, indent=2))
        if current.get("session_id") != sid:
            print(f"[M2] FAIL: Step{step} session_id 未延续")
            return 1
        if current.get("mode") == "answer":
            out_answer = current
            break

    if not out_answer:
        print("[M2] FAIL: 在限定轮次内未进入 answer")
        return 1

    out3 = out_answer

    tr3 = out3.get("trace") or {}
    if not (tr3.get("node_order") or []):
        print("[M2] FAIL: Step3 期望 trace.node_order 非空")
        return 1
    if not isinstance(tr3.get("timings_ms"), dict):
        print("[M2] FAIL: Step3 期望 trace.timings_ms 为 dict")
        return 1

    rag_stats = tr3.get("rag_stats") or {}
    if isinstance(rag_stats, dict):
        hits = int(rag_stats.get("hits") or 0)
        citations = out3.get("citations") or []
        if hits > 0 and not citations:
            print("[M2] FAIL: Step3 期望 hits>0 时 citations 非空")
            return 1

    print("\n[M2] Step4: 第4轮（same session），最剧烈头痛 + 呕吐 -> 应 escalate")
    out4 = _post({"session_id": sid, "user_message": "这是我经历过最剧烈的头痛，而且一直在呕吐"})
    print(json.dumps(out4, ensure_ascii=False, indent=2))
    if out4.get("mode") != "escalate" or not (out4.get("answer") or "").strip():
        print("[M2] FAIL: Step4 期望 mode=escalate 且 answer 非空")
        return 1

    tr4 = out4.get("trace") or {}
    order4 = tr4.get("node_order") or []
    if not _is_valid_escalate_trace(tr4):
        print(f"[M2] FAIL: Step4 trace.node_order 未体现 SafetyGate -> PersistState 短路，实际={order4}")
        return 1

    print("\n[M2] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
