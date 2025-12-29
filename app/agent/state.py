# -*- coding: utf-8 -*-
"""state.py

会话状态与槽位定义（Pydantic v2）。

注意：
- 该模块仅定义数据结构与小型纯函数，不做 I/O。
- 为避免上下文爆炸，只保留最近 20 轮 messages。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TriState = Literal["yes", "no", "unknown"]
Role = Literal["user", "assistant", "system"]


def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class Message(BaseModel):
    role: Role
    content: str = Field(default="", description="消息内容（仅保存最近 20 轮）")
    ts: str = Field(default_factory=utc_now_iso, description="UTC 时间戳")


class Slots(BaseModel):
    """结构化槽位（可追问）。

    约定：
    - 未知一律使用空字符串/空列表/unknown，不要塞入长段文本。
    - red_flags 允许存关键词列表。
    """

    age: Optional[int] = Field(default=None, description="年龄（岁）")
    sex: str = Field(default="", description="性别：男/女/其他/未知")

    symptoms: List[str] = Field(default_factory=list, description="症状列表")
    duration: str = Field(default="", description="时长：如 3天/2小时/不确定")
    severity: str = Field(default="", description="严重程度：如 7/10 或 轻/中/重")
    fever: TriState = Field(default="unknown", description="是否发热")
    location: str = Field(default="", description="部位：如 胸/腹/咽喉")

    pregnancy: TriState = Field(default="unknown", description="是否妊娠（适用时）")
    meds: str = Field(default="", description="用药")
    allergy: str = Field(default="", description="过敏史")
    history: str = Field(default="", description="既往史")

    red_flags: List[str] = Field(default_factory=list, description="红旗症状命中")
    department_guess: Optional[str] = Field(default=None, description="科室猜测，用于RAG过滤")

    def merge_from_partial(self, patch: "Slots") -> "Slots":
        """只用 patch 中的“非空/非unknown”字段补齐当前槽位。"""

        out = self.model_copy(deep=True)

        def _nonempty_str(s: str) -> bool:
            return isinstance(s, str) and s.strip() != "" and s.strip().lower() not in {"unknown", "不确定", "不知道"}

        if patch.age is not None and out.age is None:
            out.age = int(patch.age)

        if _nonempty_str(patch.sex) and not _nonempty_str(out.sex):
            out.sex = patch.sex.strip()[:16]

        if patch.symptoms and not out.symptoms:
            out.symptoms = [str(x).strip() for x in patch.symptoms if str(x).strip()][:12]

        for k in ["duration", "severity", "location", "meds", "allergy", "history"]:
            pv = getattr(patch, k)
            if _nonempty_str(pv) and not _nonempty_str(getattr(out, k)):
                setattr(out, k, str(pv).strip()[:200])

        if patch.fever in {"yes", "no"} and out.fever == "unknown":
            out.fever = patch.fever

        if patch.pregnancy in {"yes", "no"} and out.pregnancy == "unknown":
            out.pregnancy = patch.pregnancy

        if patch.red_flags and not out.red_flags:
            out.red_flags = [str(x).strip() for x in patch.red_flags if str(x).strip()][:10]

        if patch.department_guess and not out.department_guess:
            out.department_guess = str(patch.department_guess).strip()[:40]

        return out


class AgentSessionState(BaseModel):
    session_id: str
    messages: List[Message] = Field(default_factory=list, description="最近 N 轮对话")
    slots: Slots = Field(default_factory=Slots)
    summary: str = Field(default="", description="短摘要，用于构建 rag_query 与 LLM 上下文")

    # 为动态追问与防重复服务：
    # - asked_slots：历史已追问过的槽位集合（只存字段名）
    # - last_questions_hash：上一轮追问问题的 hash，用于降低重复刷屏
    # - slot_ask_counts：每个槽位的追问次数，用于 anti-repeat（批次1新增）
    asked_slots: List[str] = Field(default_factory=list, description="已追问过的槽位字段名")
    last_questions_hash: str = Field(default="", description="上一轮追问问题集合的 hash")
    slot_ask_counts: Dict[str, int] = Field(default_factory=dict, description="每个槽位的追问次数，用于 anti-repeat")

    # 为安全分流服务：
    # - safety_level：none/warn/critical
    safety_level: str = Field(default="none", description="安全等级：none/warn/critical")
    last_update_ts: str = Field(default_factory=utc_now_iso)

    def append_message(self, role: Role, content: str) -> None:
        msg = Message(role=role, content=(content or "").strip()[:4000])
        self.messages.append(msg)
        self.trim_messages(max_turns=20)
        self.last_update_ts = utc_now_iso()

    def trim_messages(self, max_turns: int = 20) -> None:
        if max_turns <= 0:
            self.messages = []
            return
        if len(self.messages) > max_turns:
            self.messages = self.messages[-max_turns:]


def build_summary_from_slots(slots: Slots) -> str:
    """用规则构建稳定的 summary，避免额外 LLM 调用带来的不确定性。"""

    parts: List[str] = []

    if slots.age is not None:
        parts.append(f"年龄{int(slots.age)}岁")

    sex = (slots.sex or "").strip()
    if sex:
        parts.append(f"性别{sex}")

    if slots.symptoms:
        parts.append("症状：" + "、".join([s for s in slots.symptoms if s][:6]))

    if (slots.location or "").strip():
        parts.append(f"部位：{slots.location.strip()}")

    if (slots.duration or "").strip():
        parts.append(f"时长：{slots.duration.strip()}")

    if (slots.severity or "").strip():
        parts.append(f"严重程度：{slots.severity.strip()}")

    if slots.fever != "unknown":
        parts.append(f"发热：{slots.fever}")

    # 妊娠只在女性/可疑场景下也可以记录，这里保持简洁
    if slots.pregnancy != "unknown":
        parts.append(f"妊娠：{slots.pregnancy}")

    if (slots.history or "").strip():
        parts.append(f"既往史：{slots.history.strip()}")

    if (slots.meds or "").strip():
        parts.append(f"用药：{slots.meds.strip()}")

    if (slots.allergy or "").strip():
        parts.append(f"过敏：{slots.allergy.strip()}")

    if slots.red_flags:
        parts.append("红旗：" + "、".join([x for x in slots.red_flags if x][:6]))

    return "；".join(parts).strip()


def to_public_slots_dict(slots: Slots) -> Dict[str, Any]:
    """给 API 返回用的 slots 快照（保证 JSON 友好）。"""

    return slots.model_dump()
