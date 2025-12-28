# -*- coding: utf-8 -*-
"""graph.py

M2：AgentOrchestration（LangGraph 多轮可追问问诊编排）。

节点设计（与需求对齐）：
- N1 SafetyGate：红旗症状检测/高风险分流
- N2 MemoryUpdate：结构化槽位抽取并更新 session_state
- N3 TriagePlanner：判断缺口 -> Ask 或 Answer
- N4 RAGRetrieve：调用 M1 的 RAG（进程内）
- N5 AnswerCompose：调用 LLM 生成带引用的回答
- N6 PersistState：写入本地存储（SQLite）

重要约束：
- 不在日志中输出完整用户文本，最多前 100 字符或 hash。
- LLM 不可用时必须能降级：使用规则抽取 + 模板回答。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from uuid import uuid4

from typing_extensions import TypedDict

from app.agent.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER_TEMPLATE,
    ESCALATE_ANSWER_TEMPLATE,
    ASK_TEXT_VARIANTS,
    FOLLOW_UP_BANK,
    SLOT_EXTRACTION_SYSTEM,
    SLOT_EXTRACTION_USER_TEMPLATE,
)
from app.agent.state import AgentSessionState, Slots, build_summary_from_slots
from app.agent.storage_sqlite import SqliteSessionStore


logger = logging.getLogger(__name__)


Mode = Literal["ask", "answer", "escalate"]


class AgentGraphState(TypedDict, total=False):
    session_id: str
    user_message: str
    top_k: int
    top_n: int
    use_rerank: bool

    session: AgentSessionState
    mode: Mode

    # ask 模式的结构化追问（前端优先消费）
    ask_text: str
    questions: List[Dict[str, Any]]
    next_questions: List[str]
    answer: str
    citations: List[Dict[str, Any]]

    # 中间态：RAG 召回结果（供 AnswerCompose 使用）
    evidence: List[Dict[str, Any]]

    # trace 为调试/验收使用，保持 JSON 友好即可，不做强类型约束
    trace: Dict[str, Any]


_STORE = SqliteSessionStore()
_GRAPH = None


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def _safe_text_for_log(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "(empty)"
    prefix = s[:100]
    if len(s) <= 100:
        return prefix
    return f"{prefix}…(sha256={_sha256_text(s)[:12]})"


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _now_ms() -> float:
    return time.perf_counter()


def _trace_start(state: AgentGraphState, node: str) -> float:
    tr = state.get("trace")
    if not isinstance(tr, dict):
        tr = {}
    order = tr.get("node_order")
    if not isinstance(order, list):
        order = []
    order.append(node)
    tr["node_order"] = order
    if "timings_ms" not in tr or not isinstance(tr.get("timings_ms"), dict):
        tr["timings_ms"] = {}
    state["trace"] = cast(Dict[str, Any], tr)
    return _now_ms()


def _trace_end(state: AgentGraphState, node: str, t0: float) -> None:
    try:
        ms = int((_now_ms() - t0) * 1000)
    except Exception:
        ms = 0
    tr = state.get("trace")
    if not isinstance(tr, dict):
        tr = {}
    timings = tr.get("timings_ms")
    if not isinstance(timings, dict):
        timings = {}
    timings[node] = ms
    tr["timings_ms"] = timings
    state["trace"] = cast(Dict[str, Any], tr)


def _stable_hash_text(text: str) -> str:
    return _sha256_text((text or "").strip())[:16]


def _pick_ask_text(sess: AgentSessionState) -> str:
    """为 ask 模式选择自然开场一句。

    使用可复现的方式，避免随机导致演示/自测不稳定。
    """

    variants = ASK_TEXT_VARIANTS or ["为了更准确判断，我想再确认几个关键信息。"]
    idx = len(sess.messages) % len(variants)
    return str(variants[idx]).strip()


def _normalize_question_text(q: str) -> str:
    return " ".join(str(q or "").strip().split())


def _looks_like_kb_question(user_message: str) -> Tuple[bool, float]:
    """粗粒度判断：用户在问科普/机制/传播途径等知识问题，而非个体问诊。

    返回：(是否科普问题, 置信度 0.0-1.0)

    置信度分层：
    - 0.9: 强信号（典型科普句式）
    - 0.7: 中信号（含科普关键词但句式不典型）
    - 0.0: 非科普问题

    目标：避免对纯科普问题反复追问年龄/性别等。
    批次3增强：支持置信度分层。
    """

    msg = str(user_message or "").strip()
    if not msg:
        return False, 0.0

    # 强信号：典型科普问句模式（高置信度）
    strong_patterns = [
        r"是什么[病症疾]?[？?]?$",
        r"(怎么|如何)(感染|传播|预防)",
        r"(原因|原理|机制)是",
        r"(潜伏期|传染期)多[长久]",
        r"会不会(传染|遗传)",
        r"能不能(传染|治愈)",
        r"有什么(区别|不同)",
    ]
    for pattern in strong_patterns:
        if re.search(pattern, msg):
            # 如果包含个人标记，降低置信度
            if any(w in msg for w in ["我", "我的", "本人"]):
                return True, 0.7
            return True, 0.9

    # 中信号：含科普意图词但句式不典型
    intent_words = [
        "是什么", "为什么", "怎么", "如何", "原因", "原理", "机制",
        "传播", "传染", "感染", "途径", "预防", "潜伏期", "传染期",
        "会不会", "能不能", "区别",
    ]
    if not any(w in msg for w in intent_words):
        return False, 0.0

    # 如果用户明确在描述自己症状/就医求助，优先走问诊
    personal_markers = ["我", "我的", "本人", "孩子", "家人", "发烧", "头痛", "咳嗽", "腹痛", "呕吐", "腹泻", "出血", "疼"]
    if any(w in msg for w in personal_markers):
        # 仍允许"我想问…怎么感染"这种
        if "怎么感染" in msg or ("传播" in msg and "怎么" in msg) or ("传染" in msg and "怎么" in msg):
            return True, 0.7
        return False, 0.0

    return True, 0.7


# ===== 批次2新增：追问次数上限 =====
MAX_ASK_PER_SLOT = 2  # 同一槽位最多追问次数


def _user_declined_slot(user_message: str) -> bool:
    """检测用户是否拒绝回答当前追问。

    用于避免反复追问用户明确不愿回答的内容。
    批次2新增。
    """
    msg = (user_message or "").strip().lower()
    if not msg:
        return False

    decline_patterns = [
        "不想说", "不方便", "不告诉", "跳过", "下一个",
        "不知道", "不确定", "不清楚", "不想回答", "不方便透露",
        "跳过这个", "不说了", "算了", "不用问了", "不回答",
    ]
    return any(p in msg for p in decline_patterns)


def _questions_hash(questions: List[Dict[str, Any]], ask_text: str) -> str:
    parts: List[str] = ["ask_text=" + _normalize_question_text(ask_text)]
    for it in questions or []:
        parts.append(f"{it.get('slot','')}:" + _normalize_question_text(str(it.get("question") or "")))
    joined = "|".join(parts)
    return _stable_hash_text(joined)


def _missing_slots(slots: Slots, user_message: str, sess: Optional[AgentSessionState] = None) -> List[str]:
    """根据当前 slots 缺口动态决定追问槽位顺序（优先级而非写死）。

    批次2增强：
    - 新增 sess 参数
    - 过滤已达追问上限的槽位
    - 检测用户拒绝回答

    约束：每轮最多 3 个。
    """

    # 检测用户是否在本轮拒绝回答
    user_declined = _user_declined_slot(user_message)

    missing: List[str] = []

    # P1：优先级（越靠前越重要）
    if slots.symptoms is None or (isinstance(slots.symptoms, list) and len(slots.symptoms) == 0):
        missing.append("symptoms")

    if slots.duration is None or not str(slots.duration or "").strip():
        missing.append("duration")

    if slots.age is None:
        missing.append("age")

    if not str(slots.sex or "").strip():
        missing.append("sex")

    if slots.severity is None or not str(slots.severity or "").strip():
        missing.append("severity")

    # 这些属于"可选但有帮助"的缺口
    if slots.fever == "unknown":
        missing.append("fever")

    if slots.location is None or not str(slots.location or "").strip():
        missing.append("location")

    # 妊娠：仅在可能相关时追问
    if str(slots.sex or "") == "女" and slots.pregnancy == "unknown":
        msg = str(user_message or "")
        if any(k in msg for k in ["腹痛", "出血", "月经", "恶心", "呕吐"]):
            missing.append("pregnancy")

    # 用药/过敏/既往史：更多用于安全建议
    if slots.meds is None or not str(slots.meds or "").strip():
        missing.append("meds")
    if slots.allergy is None or not str(slots.allergy or "").strip():
        missing.append("allergy")
    if slots.history is None or not str(slots.history or "").strip():
        missing.append("history")

    # ===== 批次2新增：过滤已达追问上限或用户拒绝的槽位 =====
    filtered: List[str] = []
    seen = set()
    for s in missing:
        if s in seen:
            continue
        seen.add(s)
        if s not in FOLLOW_UP_BANK:
            continue

        # 检查追问次数
        if sess is not None:
            count = sess.slot_ask_counts.get(s, 0)
            if count >= MAX_ASK_PER_SLOT:
                continue
            # 如果用户本轮拒绝，且该槽位上一轮刚追问过，跳过
            if user_declined and s in sess.asked_slots[-3:]:
                continue

        filtered.append(s)
        if len(filtered) >= 3:
            break

    return filtered


def _build_structured_questions(sess: AgentSessionState, user_message: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """生成结构化 questions + 兼容 next_questions。

    批次2增强：
    - 传递 sess 到 _missing_slots()
    - 更新 slot_ask_counts
    - anti-repeat：同一槽位每轮用不同句式（尽量避免连续重复同一句）
    - asked_slots/last_questions_hash：用于降低重复刷屏概率
    """

    ask_text = _pick_ask_text(sess)
    slots = sess.slots
    # ===== 批次2改动：传递 sess =====
    slots_to_ask = _missing_slots(slots, user_message, sess)

    questions: List[Dict[str, Any]] = []
    for slot in slots_to_ask:
        bank = FOLLOW_UP_BANK.get(slot) or {}
        variants = list(bank.get("question_variants") or [])
        if not variants:
            continue

        # 可复现的"换措辞"：基于轮次与上次追问 hash 选择不同 variant
        seed = f"{sess.session_id}|{slot}|{sess.last_questions_hash}|{len(sess.messages)}"
        idx = int(_sha256_text(seed)[:8], 16) % len(variants)
        question_text = str(variants[idx]).strip()

        q: Dict[str, Any] = {
            "slot": slot,
            "question": question_text,
            "type": bank.get("type") or "text",
        }
        if bank.get("placeholder"):
            q["placeholder"] = str(bank.get("placeholder"))
        choices = bank.get("choices")
        if isinstance(choices, (list, tuple)) and len(choices) > 0:
            q["choices"] = list(choices)
        r = bank.get("range")
        if isinstance(r, (list, tuple)) and len(r) == 2:
            q["range"] = [r[0], r[1]]

        questions.append(q)

        # ===== 批次2新增：更新追问次数 =====
        sess.slot_ask_counts[slot] = sess.slot_ask_counts.get(slot, 0) + 1

    next_questions = [str(q.get("question") or "").strip() for q in questions if str(q.get("question") or "").strip()]

    # 记录用于 anti-repeat / 去重刷屏
    qh = _questions_hash(questions, ask_text)
    sess.last_questions_hash = qh
    for q in questions:
        s = str(q.get("slot") or "").strip()
        if not s:
            continue
        if s not in sess.asked_slots:
            sess.asked_slots.append(s)
    # 防止列表无限增长
    if len(sess.asked_slots) > 100:
        sess.asked_slots = sess.asked_slots[-100:]

    return ask_text, questions, next_questions


def _looks_like_red_flag(text: str) -> List[str]:
    """极简红旗识别：宁可保守一点，触发就医建议。"""

    msg = (text or "").strip()
    if not msg:
        return []

    keywords = [
        "胸痛",
        "呼吸困难",
        "喘不上气",
        "意识不清",
        "昏迷",
        "抽搐",
        "口唇发紫",
        "严重过敏",
        "喉头水肿",
        "偏瘫",
        "言语不清",
        "呕血",
        "便血",
        "黑便",
        "休克",
        "持续高热",
    ]
    hits = [k for k in keywords if k in msg]
    return hits[:6]


def _guess_department(slots: Slots, user_message: str) -> Optional[str]:
    """科室猜测（用于 RAG filter）。"""

    msg = (user_message or "")
    if slots.department_guess:
        return slots.department_guess

    # 极简规则：可持续扩展
    if any(k in msg for k in ["月经", "怀孕", "孕", "阴道出血", "白带", "产后"]):
        return "妇产科"
    if any(k in msg for k in ["婴儿", "宝宝", "小孩", "儿童", "幼儿", "发育"]):
        return "儿科"
    if any(k in msg for k in ["皮疹", "瘙痒", "荨麻疹", "痘", "湿疹"]):
        return "皮肤科"
    if any(k in msg for k in ["焦虑", "抑郁", "失眠", "恐慌"]):
        return "心理科"

    return None


_KNOWN_DEPARTMENTS = {
    "传染病科",
    "传染科",
    "儿科",
    "妇产科",
    "家居健康",
    "男科",
    "内科",
    "皮肤科",
    "皮肤性病科",
    "其他",
    "外科",
    "未分类",
    "五官科",
    "心理科",
    "整形美容",
    "中医科",
    "肿瘤科",
}


def _normalize_department(dept: Optional[str]) -> Optional[str]:
    """将科室名归一化到知识库可用的严格等值集合。

    RAG 的 department 过滤是严格等值匹配；如果 LLM 生成了更细分的科室名
    （例如“呼吸内科”），直接过滤会导致 0 命中。
    """

    if not dept:
        return None
    d = str(dept).strip()
    if not d:
        return None

    mapping = {
        "呼吸内科": "内科",
        "呼吸科": "内科",
        "消化内科": "内科",
        "心内科": "内科",
        "神经内科": "内科",
        "肾内科": "内科",
        "内分泌科": "内科",
        "感染科": "传染科",
        "皮肤性病": "皮肤性病科",
        "五官": "五官科",
        "耳鼻喉": "五官科",
        "耳鼻喉科": "五官科",
        "肿瘤": "肿瘤科",
    }
    d = mapping.get(d, d)
    return d if d in _KNOWN_DEPARTMENTS else None


def _rule_extract_slots(user_message: str) -> Slots:
    """LLM 失败时的规则兜底抽取。"""

    msg = (user_message or "").strip()
    out = Slots()

    # 年龄
    m = re.search(r"(\d{1,3})\s*岁", msg)
    if m:
        try:
            age = int(m.group(1))
            if 0 < age < 130:
                out.age = age
        except Exception:
            pass

    # 性别
    if any(k in msg for k in ["男", "先生"]):
        out.sex = "男"
    if any(k in msg for k in ["女", "女士"]):
        out.sex = "女"

    # 发热
    if any(k in msg for k in ["发烧", "发热", "高烧", "低烧", "体温"]):
        out.fever = "yes"
    if any(k in msg for k in ["不发烧", "没发烧", "无发热"]):
        out.fever = "no"

    # 妊娠
    if any(k in msg for k in ["怀孕", "孕", "妊娠"]):
        out.pregnancy = "yes"
    if any(k in msg for k in ["未怀孕", "不怀孕"]):
        out.pregnancy = "no"

    # 症状：用关键词集合简单命中
    symptom_keywords = [
        "咳嗽",
        "发热",
        "发烧",
        "流鼻涕",
        "鼻塞",
        "咽痛",
        "头痛",
        "腹痛",
        "腹泻",
        "呕吐",
        "胸痛",
        "胸闷",
        "气短",
        "皮疹",
        "瘙痒",
        "尿频",
        "尿痛",
    ]
    hits = []
    for k in symptom_keywords:
        if k in msg and k not in hits:
            hits.append(k)
    out.symptoms = hits[:10]

    # 部位
    for loc in ["胸", "腹", "胃", "喉", "咽", "头", "背", "腰"]:
        if loc in msg:
            out.location = loc
            break

    # 时长
    m2 = re.search(r"([0-9一二三四五六七八九十]+)\s*(分钟|小时|天|周|个月|月|年)", msg)
    if m2:
        out.duration = f"{m2.group(1)}{m2.group(2)}"

    # 严重程度
    m3 = re.search(r"\b(10|[0-9])\s*/\s*10\b", msg)
    if m3:
        out.severity = f"{m3.group(1)}/10"

    # 红旗
    out.red_flags = _looks_like_red_flag(msg)

    # 科室
    out.department_guess = _guess_department(out, msg)

    return out


def _call_llm_text(system: str, user: str) -> str:
    """统一的 LLM 调用封装（DeepSeek/OpenAI 兼容）。

    注意：
    - 该函数可能抛异常；上层必须捕获并降级。
    """

    try:
        try:
            from app import config_llm  # type: ignore
        except Exception:
            import config_llm  # type: ignore

        llm = config_llm.get_llm(temperature=0.0)

        prompt = f"{system}\n\n{user}".strip()

        if hasattr(llm, "predict"):
            return str(llm.predict(prompt))  # type: ignore[attr-defined]
        # 兼容一些 LangChain 版本
        return str(llm(prompt))
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败：{type(e).__name__}: {e}") from e


def _call_llm_json(system: str, user: str) -> Dict[str, Any]:
    """调用 LLM 并解析 JSON（抽取用）。"""

    text = _call_llm_text(system, user)
    s = (text or "").strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    json_text = m.group(0) if m else s
    try:
        parsed = json.loads(json_text)
    except Exception as e:
        raise RuntimeError(f"LLM JSON 解析失败：{type(e).__name__}: {e}") from e
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM JSON 不是对象")
    return parsed


def _extract_slots_with_llm(user_message: str) -> Slots:
    raw = _call_llm_json(
        SLOT_EXTRACTION_SYSTEM,
        SLOT_EXTRACTION_USER_TEMPLATE.format(user_message=(user_message or "").strip()),
    )
    try:
        # 允许缺字段；Pydantic 会自动填默认
        slots = Slots.model_validate(raw)
    except Exception as e:
        raise RuntimeError(f"槽位校验失败：{type(e).__name__}: {e}") from e

    # 二次补充：department_guess 若缺则规则猜测
    if not slots.department_guess:
        slots.department_guess = _guess_department(slots, user_message)

    # red_flags 再跑一次规则（避免模型漏掉）
    if not slots.red_flags:
        slots.red_flags = _looks_like_red_flag(user_message)

    return slots


def _missing_questions(slots: Slots) -> List[str]:
    """按优先级生成最多 3 个追问问题。"""

    qs: List[str] = []

    # 红旗优先（若出现，则外层会走 escalate；这里仅用于非红旗补问）

    if slots.age is None:
        qs.append("请问你的年龄大概是多少岁？")

    if not (slots.sex or "").strip():
        qs.append("请问你的性别是？")

    if not slots.symptoms:
        qs.append("你最主要的不适/症状是什么？可以列 1-3 个关键词。")

    if not (slots.duration or "").strip():
        qs.append("这些症状大概持续多久了？（例如：2小时/3天/一周左右）")

    if not (slots.severity or "").strip():
        qs.append("目前症状大概有多严重？可以用 0-10 分或‘轻/中/重’描述。")

    # 妊娠：仅在女性且育龄段或用户提到相关词时追问
    if (slots.sex == "女") and (slots.pregnancy == "unknown"):
        if slots.age is None or (12 <= int(slots.age or 0) <= 55):
            if any(k in "、".join(slots.symptoms) + (slots.location or "") for k in ["腹痛", "出血", "月经", "恶心"]):
                qs.append("请问是否有怀孕/备孕的可能？（有/无/不确定）")

    return qs[:3]


def _slots_sufficient_for_answer(slots: Slots) -> bool:
    """最小可回答条件（偏保守，保证问诊质量）：

    - 症状 + 时长 + 年龄
    - 性别（部分建议与用药/妊娠相关）
    - 严重程度（0-10 或 轻/中/重）
    - 对部分常见症状：发热未知时继续追问（避免漏掉感染/高热风险）
    """

    if not slots.symptoms:
        return False
    if not (slots.duration or "").strip():
        return False
    # 年龄不足时也可以退化到询问
    if slots.age is None:
        return False
    if not (slots.sex or "").strip():
        return False
    if not (slots.severity or "").strip():
        return False

    symptom_text = "、".join([str(x) for x in (slots.symptoms or []) if str(x).strip()])
    if slots.fever == "unknown" and any(k in symptom_text for k in ["头痛", "咳嗽", "腹痛", "腹泻", "呕吐", "咽痛"]):
        return False
    return True


def _format_evidence_for_llm(evidence: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for ev in evidence or []:
        eid = str(ev.get("eid") or "").strip()
        text = str(ev.get("text") or "").strip().replace("\n", " ")
        if len(text) > 360:
            text = text[:357] + "..."
        source = str(ev.get("source") or "").strip()
        lines.append(f"[{eid}] source={source} text={text}")
    return "\n".join(lines)


def _citations_from_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in evidence or []:
        md = ev.get("metadata") if isinstance(ev.get("metadata"), dict) else {}
        text = str(ev.get("text") or "").strip().replace("\n", " ")
        snippet = text[:160] + ("…" if len(text) > 160 else "")
        out.append(
            {
                "eid": ev.get("eid"),
                "score": ev.get("score"),
                "department": (md.get("department") if md else None),
                "title": (md.get("title") if md else None),
                "snippet": snippet,
                "source": ev.get("source"),
                "chunk_id": ev.get("chunk_id"),
                "rerank_score": ev.get("rerank_score"),
            }
        )
    return out


def _ensure_answer_contract(answer: str, evidence: List[Dict[str, Any]]) -> str:
    """确保回答满足契约：引用与免责声明。

    - 校验 LLM 输出中的引用 ID 是否在 evidence 中存在，移除无效引用
    - 若 evidence 非空：至少包含一行 `引用：[E1][E2]`（按现有 eid 顺序）
    - 若 evidence 为空：输出 `引用：[]`
    - 末尾必须包含标准化免责声明（固定位置）
    """

    s = (answer or "").strip()

    # 获取有效的 evidence IDs
    valid_eids = {str(ev.get("eid") or "").strip() for ev in (evidence or []) if str(ev.get("eid") or "").strip()}

    # 提取回答中的引用 ID 并校验
    cited_in_text = set(re.findall(r'\[(E\d+)\]', s))
    invalid_cites = cited_in_text - valid_eids
    if invalid_cites:
        # 移除无效引用
        for invalid in invalid_cites:
            s = s.replace(f"[{invalid}]", "")

    # 标准化免责声明（统一格式）
    DISCLAIMER = "免责声明：本回答仅供信息参考，不能替代医生面诊。"

    # 移除现有的引用行和免责声明（可能格式不一致），后续统一添加
    s = re.sub(r"\n*引用[：:]\s*\[?[^\n]*\]?\s*\n*", "\n", s)
    s = re.sub(r"\n*免责声明[：:][^\n]*\n*", "\n", s)
    s = s.strip()

    # 构建标准化引用行
    eids_sorted = sorted([eid for eid in valid_eids], key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    cite_line = "引用：[]" if not eids_sorted else "引用：" + "".join([f"[{eid}]" for eid in eids_sorted])

    # 强制格式：正文 + 空行 + 引用 + 换行 + 免责
    s = s + "\n\n" + cite_line + "\n" + DISCLAIMER

    return s.strip()


def _node_safety_gate(state: AgentGraphState) -> Dict[str, Any]:
    node = "SafetyGate"
    t0 = _trace_start(state, node)

    sess = state.get("session")
    if not isinstance(sess, AgentSessionState):
        raise RuntimeError("SafetyGate: session 缺失")

    msg = str(state.get("user_message") or "").strip()

    hits = _looks_like_red_flag(msg)

    # 记录 safety_flags 到 trace
    tr = state.get("trace") if isinstance(state.get("trace"), dict) else {}
    tr["safety_flags"] = hits if hits else []
    state["trace"] = cast(Dict[str, Any], tr)

    if hits:
        sess.slots = sess.slots.model_copy(update={"red_flags": hits})
        sess.safety_level = "critical"
        state["mode"] = "escalate"
        state["answer"] = ESCALATE_ANSWER_TEMPLATE.format(red_flags="、".join(hits))
        state["ask_text"] = ""
        state["questions"] = []
        state["next_questions"] = []
        state["citations"] = []
    else:
        sess.safety_level = sess.safety_level or "none"

    _trace_end(state, node, t0)
    return {
        "session": sess,
        "mode": state.get("mode"),
        "answer": state.get("answer"),
        "ask_text": state.get("ask_text"),
        "questions": state.get("questions"),
        "next_questions": state.get("next_questions"),
        "citations": state.get("citations"),
    }


def _node_memory_update(state: AgentGraphState) -> Dict[str, Any]:
    node = "MemoryUpdate"
    t0 = _trace_start(state, node)

    sess = state.get("session")
    if not isinstance(sess, AgentSessionState):
        raise RuntimeError("MemoryUpdate: session 缺失")

    msg = str(state.get("user_message") or "").strip()

    # 记录用户消息
    sess.append_message("user", msg)

    # ===== 批次1新增：记录更新前的槽位 =====
    old_slots = sess.slots.model_dump()

    # 抽取策略：优先 LLM，失败则规则兜底；支持强制 rules（用于离线自测/CI）
    patch_slots: Slots
    forced = str(os.getenv("AGENT_SLOT_EXTRACTOR", "auto")).strip().lower()
    if forced == "rules":
        patch_slots = _rule_extract_slots(msg)
    else:
        try:
            patch_slots = _extract_slots_with_llm(msg)
        except Exception as e:
            if _env_flag("RAG_DEBUG", "0"):
                logger.warning("[Agent] 槽位抽取降级为规则：%s", str(e))
            patch_slots = _rule_extract_slots(msg)

    sess.slots = sess.slots.merge_from_partial(patch_slots)

    # 科室猜测补充
    if not sess.slots.department_guess:
        sess.slots.department_guess = _guess_department(sess.slots, msg)

    # 归一化科室名：不在知识库集合内则不启用过滤
    sess.slots.department_guess = _normalize_department(sess.slots.department_guess)

    # summary 用规则稳定构建
    sess.summary = build_summary_from_slots(sess.slots)

    # ===== 批次1新增：记录槽位变化到 trace =====
    new_slots = sess.slots.model_dump()
    slots_changed = [k for k in old_slots if old_slots[k] != new_slots[k]]

    tr = state.get("trace")
    if not isinstance(tr, dict):
        tr = {}
    tr["slots_changed"] = slots_changed
    state["trace"] = cast(Dict[str, Any], tr)

    _trace_end(state, node, t0)
    return {"session": sess}


def _node_triage_planner(state: AgentGraphState) -> Dict[str, Any]:
    node = "TriagePlanner"
    t0 = _trace_start(state, node)

    sess = state.get("session")
    if not isinstance(sess, AgentSessionState):
        raise RuntimeError("TriagePlanner: session 缺失")

    # 若已 escalate，直接保持
    if state.get("mode") == "escalate":
        _trace_end(state, node, t0)
        return {"mode": "escalate"}

    slots = sess.slots
    user_msg = str(state.get("user_message") or "").strip()

    # ===== 批次3改动：使用带置信度的 kb_qa 检测 =====
    is_kb, kb_confidence = _looks_like_kb_question(user_msg)

    tr_any = state.get("trace")
    tr = tr_any if isinstance(tr_any, dict) else {}

    # 记录 kb_qa 统计
    tr["kb_qa_stats"] = {
        "detected": is_kb,
        "confidence": kb_confidence,
    }

    if is_kb and kb_confidence >= 0.7:
        # 知识问答直达：不要求补全问诊槽位
        state["mode"] = "answer"
        state["ask_text"] = ""
        state["questions"] = []
        state["next_questions"] = []
        tr["planner_strategy"] = "kb_qa"
        state["trace"] = cast(Dict[str, Any], tr)
        _trace_end(state, node, t0)
        return {
            "mode": state.get("mode"),
            "ask_text": state.get("ask_text"),
            "questions": state.get("questions"),
            "next_questions": state.get("next_questions"),
            "answer": state.get("answer"),
            "citations": state.get("citations"),
            "session": sess,
        }

    # 正常问诊流程
    tr["planner_strategy"] = "triage"
    state["trace"] = cast(Dict[str, Any], tr)

    if not _slots_sufficient_for_answer(slots):
        ask_text, questions, next_questions = _build_structured_questions(sess, user_msg)
        state["mode"] = "ask"
        state["ask_text"] = ask_text
        state["questions"] = questions
        state["next_questions"] = next_questions
        state["answer"] = ""
        state["citations"] = []
    else:
        state["mode"] = "answer"

    _trace_end(state, node, t0)
    return {
        "mode": state.get("mode"),
        "ask_text": state.get("ask_text"),
        "questions": state.get("questions"),
        "next_questions": state.get("next_questions"),
        "answer": state.get("answer"),
        "citations": state.get("citations"),
        "session": sess,
    }


def _node_rag_retrieve(state: AgentGraphState) -> Dict[str, Any]:
    node = "RAGRetrieve"
    t0 = _trace_start(state, node)

    if state.get("mode") != "answer":
        _trace_end(state, node, t0)
        return {}

    sess = state.get("session")
    if not isinstance(sess, AgentSessionState):
        raise RuntimeError("RAGRetrieve: session 缺失")

    msg = str(state.get("user_message") or "").strip()
    top_k = int(state.get("top_k") or 5)
    top_n = int(state.get("top_n") or 30)
    use_rerank = bool(state.get("use_rerank") if state.get("use_rerank") is not None else True)

    rag_query = (sess.summary + "；问题：" + msg).strip("； ")

    dept = sess.slots.department_guess

    # 调用 M1：必须走兼容层 app.rag.retriever.retrieve
    evidence: List[Dict[str, Any]]
    try:
        from app.rag.retriever import retrieve as rag_retrieve  # type: ignore
        from app.rag.rag_core import get_stats  # type: ignore

        rag_t0 = _now_ms()
        evidence = cast(
            List[Dict[str, Any]],
            rag_retrieve(
            rag_query,
            top_k=top_k,
            top_n=top_n,
            department=dept,
            use_rerank=use_rerank,
            ),
        )
        rag_latency_ms = int((_now_ms() - rag_t0) * 1000)

        st = get_stats()
        tr_any = state.get("trace")
        tr: Dict[str, Any] = tr_any if isinstance(tr_any, dict) else {}
        tr["rag_stats"] = {
            "device": st.device,
            "collection": st.collection,
            "count": st.count,
            "hits": len(evidence or []),
            "latency_ms": rag_latency_ms,
        }
        state["trace"] = cast(Dict[str, Any], tr)

    except Exception as e:
        # RAG 失败不能直接崩：退化为无证据回答
        if _env_flag("RAG_DEBUG", "0"):
            logger.warning("[Agent] RAG 检索失败：%s", str(e))
        evidence = []
        tr_any2 = state.get("trace")
        tr = tr_any2 if isinstance(tr_any2, dict) else {}
        tr["rag_stats"] = {"hits": 0, "latency_ms": 0}
        tr["rag_error"] = f"{type(e).__name__}: {e}"
        state["trace"] = cast(Dict[str, Any], tr)

    _trace_end(state, node, t0)
    return {"evidence": evidence}


def _node_answer_compose(state: AgentGraphState) -> Dict[str, Any]:
    node = "AnswerCompose"
    t0 = _trace_start(state, node)

    sess = state.get("session")
    if not isinstance(sess, AgentSessionState):
        raise RuntimeError("AnswerCompose: session 缺失")

    mode = state.get("mode")

    # escalate 已在 SafetyGate 生成 answer
    if mode == "escalate":
        ans = str(state.get("answer") or "").strip()
        sess.append_message("assistant", ans)
        _trace_end(state, node, t0)
        return {"answer": ans, "citations": []}

    if mode == "ask":
        ask_text = str(state.get("ask_text") or "").strip()
        qs = state.get("next_questions") or []
        # 也存一条 assistant 消息，便于对话回放（不包含用户隐私字段）
        lines: List[str] = []
        if ask_text:
            lines.append(ask_text)
        if qs:
            lines.extend([f"- {q}" for q in qs])
        if lines:
            sess.append_message("assistant", "\n".join(lines))
        _trace_end(state, node, t0)
        return {"answer": "", "citations": []}

    # answer 模式
    evidence = cast(List[Dict[str, Any]], state.get("evidence") if isinstance(state.get("evidence"), list) else [])
    evidence_block = _format_evidence_for_llm(evidence)

    if not evidence:
        # 无证据：不强行调用 LLM（减少不确定性与成本）
        ans = (
            "目前未检索到可靠的本地资料支持你的问题。\n"
            "建议：\n"
            "- 若症状轻微：注意休息、补液、观察变化；\n"
            "- 若症状持续加重、出现胸痛/呼吸困难/意识改变等危险信号，请尽快就医。\n\n"
            "引用：[]\n"
            "免责声明：本回答仅供信息参考，不能替代医生面诊。"
        )
        ans = _ensure_answer_contract(ans, [])
        sess.append_message("assistant", ans)
        state["citations"] = []
        _trace_end(state, node, t0)
        return {"answer": ans, "citations": []}

    # 有证据：调用 LLM 生成回答
    user_msg = str(state.get("user_message") or "").strip()
    summary = sess.summary

    try:
        user_prompt = ANSWER_USER_TEMPLATE.format(
            user_message=user_msg,
            summary=summary or "(无)",
            evidence_block=evidence_block or "(无)",
        )
        text = _call_llm_text(ANSWER_SYSTEM, user_prompt)
        ans = (text or "").strip()
    except Exception as e:
        if _env_flag("RAG_DEBUG", "0"):
            logger.warning("[Agent] LLM 回答失败，退化为模板：%s", str(e))
        ans = (
            "我根据检索到的资料做了初步整理：\n"
            "- 你可以先关注：症状变化、是否出现持续高热/呼吸困难等危险信号。\n"
            "- 若症状持续或加重，建议尽快线下就医。\n\n"
            "引用：[E1]\n"
            "免责声明：本回答仅供信息参考，不能替代医生面诊。"
        )

    citations = _citations_from_evidence(evidence)
    ans = _ensure_answer_contract(ans, evidence)
    sess.append_message("assistant", ans)
    state["citations"] = citations

    _trace_end(state, node, t0)
    return {"answer": ans, "citations": citations}


def _node_persist_state(state: AgentGraphState) -> Dict[str, Any]:
    node = "PersistState"
    t0 = _trace_start(state, node)

    sess = state.get("session")
    if not isinstance(sess, AgentSessionState):
        raise RuntimeError("PersistState: session 缺失")

    try:
        _STORE.save_session(sess)
        tr_any3 = state.get("trace")
        tr: Dict[str, Any] = tr_any3 if isinstance(tr_any3, dict) else {}
        tr["storage"] = {"type": "sqlite", "db_path": _STORE.db_path}
        state["trace"] = cast(Dict[str, Any], tr)
    except Exception as e:
        raise RuntimeError(f"会话持久化失败：{type(e).__name__}: {e}") from e

    _trace_end(state, node, t0)
    return {"session": sess}


def _route_after_safety(state: AgentGraphState) -> str:
    if state.get("mode") == "escalate":
        return "persist"
    return "memory"


def _route_after_planner(state: AgentGraphState) -> str:
    m = state.get("mode")
    if m == "ask":
        return "persist"
    if m == "answer":
        return "rag"
    if m == "escalate":
        return "persist"
    return "persist"


def _route_after_rag(state: AgentGraphState) -> str:
    return "compose"


def _get_graph():
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except Exception as e:
        raise RuntimeError("LangGraph 未安装或不可用，请确认 requirements.txt 已安装 langgraph==0.0.20") from e

    g = StateGraph(AgentGraphState)
    g.add_node("safety", _node_safety_gate)
    g.add_node("memory", _node_memory_update)
    g.add_node("plan", _node_triage_planner)
    g.add_node("rag", _node_rag_retrieve)
    g.add_node("compose", _node_answer_compose)
    g.add_node("persist", _node_persist_state)

    g.set_entry_point("safety")

    g.add_conditional_edges("safety", _route_after_safety, {"persist": "persist", "memory": "memory"})
    g.add_edge("memory", "plan")
    g.add_conditional_edges("plan", _route_after_planner, {"persist": "persist", "rag": "rag"})
    g.add_conditional_edges("rag", _route_after_rag, {"compose": "compose"})
    g.add_edge("compose", "persist")
    g.add_edge("persist", END)

    _GRAPH = g.compile()
    return _GRAPH


def load_or_create_session(session_id: Optional[str] = None) -> AgentSessionState:
    sid = (session_id or "").strip()
    if not sid:
        sid = str(uuid4())

    st = _STORE.load_session(sid)
    if st is not None:
        return st

    return AgentSessionState(session_id=sid)


def run_chat_v2_turn(
    *,
    session_id: Optional[str],
    user_message: str,
    top_k: int = 5,
    top_n: int = 30,
    use_rerank: bool = True,
) -> Dict[str, Any]:
    """执行一轮对话，返回 API 需要的结构化结果（不含 FastAPI 依赖）。"""

    sid = (session_id or "").strip() or str(uuid4())
    user_msg = (user_message or "").strip()
    if not user_msg:
        raise ValueError("user_message 不能为空")

    sess = load_or_create_session(sid)

    state: AgentGraphState = {
        "session_id": sid,
        "user_message": user_msg,
        "top_k": int(top_k) if int(top_k) > 0 else 5,
        "top_n": int(top_n) if int(top_n) > 0 else 30,
        "use_rerank": bool(use_rerank),
        "session": sess,
        "mode": "ask",
        "ask_text": "",
        "questions": [],
        "next_questions": [],
        "answer": "",
        "citations": [],
        "evidence": [],
        "trace": {"node_order": [], "timings_ms": {}, "rag_stats": {}},
    }

    if _env_flag("RAG_DEBUG", "0"):
        logger.info(
            "[Agent] chat_v2 start session_id=%s msg=%s top_k=%s top_n=%s use_rerank=%s",
            sid,
            _safe_text_for_log(user_msg),
            state["top_k"],
            state["top_n"],
            state["use_rerank"],
        )

    g = _get_graph()
    out = cast(Dict[str, Any], g.invoke(state))

    sess2: AgentSessionState = sess
    tmp_session = out.get("session")
    if isinstance(tmp_session, AgentSessionState):
        sess2 = tmp_session

    # 安全：slots 可能被更新
    slots_dict = sess2.slots.model_dump()

    resp = {
        "session_id": sess2.session_id,
        "mode": out.get("mode"),
        "ask_text": out.get("ask_text") or "",
        "questions": out.get("questions") or [],
        "next_questions": out.get("next_questions") or [],
        "answer": out.get("answer") or "",
        "citations": out.get("citations") or [],
        "slots": slots_dict,
        "summary": sess2.summary,
        "trace": out.get("trace") if isinstance(out.get("trace"), dict) else {},
    }
    return resp
