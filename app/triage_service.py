# -*- coding: utf-8 -*-
"""
triage_service.py
阶段2：FastAPI包装前的“单次分诊服务层”
- 输入：user_text(用户一句话描述)
- 输出：固定JSON结构(answer_json) + evidence_list + rag_query
- 支持 mode=fast|safe：safe会走安全审查链(更慢但更稳)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

try:
    from . import config_llm  # type: ignore
except Exception:
    import config_llm  # type: ignore

try:
    from .triage_protocol import build_triage_payload  # type: ignore
except Exception:
    from triage_protocol import build_triage_payload  # type: ignore

try:
    from .privacy import redact_pii_for_llm  # type: ignore
except Exception:
    from privacy import redact_pii_for_llm  # type: ignore

try:
    from .safety.record_guard import apply_record_conflicts_to_triage_json, detect_record_conflicts  # type: ignore
except Exception:
    from safety.record_guard import apply_record_conflicts_to_triage_json, detect_record_conflicts  # type: ignore

try:
    from .safety.conflict_judge import judge_json_conflicts  # type: ignore
except Exception:
    from safety.conflict_judge import judge_json_conflicts  # type: ignore

try:
    from .safety.safety_fuse import apply_confirmed_safety_fuse_to_triage_answer  # type: ignore
except Exception:
    from safety.safety_fuse import apply_confirmed_safety_fuse_to_triage_answer  # type: ignore

try:
    # 你的项目里已有 rag/retriever.py
    from .rag import retriever as rag_retriever  # type: ignore
    from .rag.evidence_policy import is_low_evidence, summarize_evidence_quality  # type: ignore
except Exception:
    try:
        from rag import retriever as rag_retriever  # type: ignore
        from rag.evidence_policy import is_low_evidence, summarize_evidence_quality  # type: ignore
    except Exception:
        rag_retriever = None
        def summarize_evidence_quality(evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
            count = len(evidence_list or [])
            return {"level": "none" if count <= 0 else "ok", "reason": "fallback", "count": count}

        def is_low_evidence(summary: Dict[str, Any]) -> bool:
            return str((summary or {}).get("level") or "").strip().lower() in {"low", "none"}


logger = logging.getLogger(__name__)


TRIAGE_LEVELS = ["EMERGENCY_NOW", "URGENT_24H", "ROUTINE", "SELF_CARE"]

# 从低到高的紧急程度顺序（用于确定性兜底规则）
TRIAGE_ORDER = ["SELF_CARE", "ROUTINE", "URGENT_24H", "EMERGENCY_NOW"]


def _count_distinct_citations(text: str) -> int:
    citations = set(re.findall(r"\[E(\d+)\]", text or ""))
    return len(citations)


def _extract_citations_used(text: str) -> List[str]:
    nums = sorted(set(re.findall(r"\[E(\d+)\]", text or "")), key=lambda x: int(x))
    return [f"E{n}" for n in nums]


def _allowed_citation_numbers(evidence_list: List[Dict[str, Any]]) -> List[int]:
    allowed: List[int] = []
    for ev in evidence_list or []:
        eid = str(ev.get("eid", "")).strip()
        m = re.fullmatch(r"E(\d+)", eid)
        if m:
            allowed.append(int(m.group(1)))
    return sorted(set(allowed))


def _invalid_citation_numbers(text: str, allowed_numbers: List[int]) -> List[int]:
    allowed_set = set(allowed_numbers or [])
    used = sorted(set(int(n) for n in re.findall(r"\[E(\d+)\]", text or "")))
    return [n for n in used if n not in allowed_set]


def _strip_invalid_citations_from_obj(obj: Any, allowed_numbers: List[int]) -> Any:
    if not allowed_numbers:
        return obj

    allowed_set = set(allowed_numbers)

    def _strip_in_str(s: str) -> str:
        def repl(m: re.Match) -> str:
            n = int(m.group(1))
            return m.group(0) if n in allowed_set else ""

        return re.sub(r"\[E(\d+)\]", repl, s or "")

    if isinstance(obj, str):
        return _strip_in_str(obj)
    if isinstance(obj, list):
        return [_strip_invalid_citations_from_obj(x, allowed_numbers) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_invalid_citations_from_obj(v, allowed_numbers) for k, v in obj.items()}
    return obj


def _should_escalate_to_emergency(answer_json: Dict[str, Any]) -> bool:
    def _is_meta_not_red_flag(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return True
        meta_markers = ["信息不足", "无法评估", "不确定", "证据不足", "需要补充", "模糊", "不能判断"]
        return any(m in s for m in meta_markers)

    def _looks_like_real_red_flag(s: str) -> bool:
        s = (s or "").strip()
        if not s or _is_meta_not_red_flag(s):
            return False
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
            "突发",
            "剧烈头痛",
            "呕血",
            "便血",
            "黑便",
            "持续高热",
            "休克",
        ]
        return any(k in s for k in keywords)

    red_flags = answer_json.get("red_flags")
    if isinstance(red_flags, list):
        real = [str(x).strip() for x in red_flags if _looks_like_real_red_flag(str(x))]
        # Only escalate when there are real red-flag symptoms, not meta statements.
        if real:
            return True

    # 额外兜底：如果明确出现急救动作指令，也提升
    actions = answer_json.get("immediate_actions")
    if isinstance(actions, list):
        joined = " \n ".join(str(x) for x in actions).lower()
        if any(k in joined for k in ["call 911", "call 999", "call emergency", "ambulance", "emergency services"]):
            return True
    return False


def _apply_guardrails(answer_json: Dict[str, Any], evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(answer_json, dict):
        return {}

    out = dict(answer_json)

    # Guardrail 1: triage_level 下限
    triage_level = str(out.get("triage_level", "ROUTINE")).strip()
    if triage_level not in TRIAGE_LEVELS:
        triage_level = "ROUTINE"
    if _should_escalate_to_emergency(out):
        triage_level = "EMERGENCY_NOW"

    # If model escalated solely due to ambiguity/insufficient info, soften it.
    try:
        unc = str(out.get("uncertainty") or "").lower()
        red_flags = out.get("red_flags")
        has_real_red_flags = False
        if isinstance(red_flags, list):
            has_real_red_flags = any(str(x).strip() for x in red_flags) and _should_escalate_to_emergency(out)
        if triage_level == "EMERGENCY_NOW" and ("evidence_insufficient" in unc or "no_local_evidence" in unc) and not has_real_red_flags:
            triage_level = "URGENT_24H"
    except Exception:
        pass
    out["triage_level"] = triage_level

    # Guardrail 2: citations_used 重算 + 合法性标签（真正的严格校验在 _ensure_valid_answer_json 中处理）
    serialized = json.dumps(out, ensure_ascii=False)
    out["citations_used"] = _extract_citations_used(serialized)
    return out


def _append_uncertainty_tag(text: str, tag: str) -> str:
    parts = [p.strip() for p in str(text or "").split("|") if p.strip()]
    if tag not in parts:
        parts.append(tag)
    return " | ".join(parts)


def _apply_record_safety(
    answer_json: Dict[str, Any],
    clinical_record_text: str,
    longitudinal_records: Optional[List[Any]] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if trace is None:
        trace = []

    longitudinal_records = list(longitudinal_records or [])
    record_text = str(clinical_record_text or "").strip()
    fused = apply_confirmed_safety_fuse_to_triage_answer(
        answer_json=answer_json,
        longitudinal_records=longitudinal_records,
    )
    fuse_trace = dict(fused.get("trace") or {})
    out = dict(answer_json if isinstance(answer_json, dict) else {})

    if longitudinal_records:
        out = dict(fused.get("answer_json") or {})
        status = "blocked" if int(fuse_trace.get("blocked_count") or 0) > 0 else "clear"
    elif record_text:
        status = "disabled_unconfirmed_source"
    else:
        status = "skipped_no_confirmed_records"

    trace.append(
        {
            "step": "record.safety",
            "status": status,
            "reason": "clinical_record_path requires in-dialog confirmation" if (record_text and not longitudinal_records) else "",
            "constraint_count": int(fuse_trace.get("constraint_count") or 0),
            "candidate_count": int(fuse_trace.get("candidate_count") or 0),
            "blocked_count": int(fuse_trace.get("blocked_count") or 0),
            "warning_count": int(fuse_trace.get("warning_count") or 0),
            "rule_checks": int(fuse_trace.get("candidate_count") or 0),
            "rule_blocked": int(fuse_trace.get("blocked_count") or 0),
            "model_judge_used": bool(fuse_trace.get("model_judge_used") or False),
            "model_confirmed": 0,
            "rewrite_used": bool(fuse_trace.get("rewrite_used") or False),
            "blocked_items": list(fuse_trace.get("blocked_items") or []),
        }
    )
    if not longitudinal_records:
        out["record_conflicts"] = []
    return out


def _default_low_evidence_questions() -> List[str]:
    return [
        "症状从什么时候开始，是持续存在还是间断发作？",
        "除了当前不适外，是否还有发热、呼吸困难、胸痛、呕吐或意识改变等伴随症状？",
        "你的年龄、基础病、正在使用的药物和过敏情况分别是什么？",
    ]


def _format_evidence_block(evidence_list: List[Dict[str, Any]]) -> str:
    lines = []
    for ev in (evidence_list or []):
        eid = (ev.get("eid") or "").strip() or "E?"
        source = (ev.get("source") or "").strip()
        page = ev.get("page", None)
        chunk_id = (ev.get("chunk_id") or "").strip()
        text = (ev.get("text") or ev.get("snippet") or "").strip()

        snippet = " ".join(text.split())
        if len(snippet) > 260:
            snippet = snippet[:257] + "..."

        page_str = "" if page is None else str(page)
        lines.append(f"[{eid}] source={source} page={page_str} chunk_id={chunk_id} snippet={snippet}")
    return "\n".join(lines) if lines else "(No local evidence retrieved.)"


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    从模型输出中提取第一个 JSON object 字符串，允许模型偶尔输出多余文本时自愈。
    """
    if not text:
        return None
    s = text.strip()

    # 1) 直接尝试整体解析
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    # 2) 截取第一个 { ... }，用简单括号配对
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _normalize_answer_schema(answer: Any, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    统一输出字段，补默认值，强制 triage_level 合法，
    并用正则从全文计算 citations_used（不信任模型自填）。
    """
    if not isinstance(answer, dict):
        answer = {}

    triage_level = str(answer.get("triage_level", "ROUTINE")).strip()
    if triage_level not in TRIAGE_LEVELS:
        triage_level = "ROUTINE"

    def _list(v) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]

    safety_notice = answer.get("safety_notice")
    if not safety_notice:
        safety_notice = (
            "本回答由AI生成，仅供信息参考，不能替代医生的面诊、诊断或治疗。"
            "如出现胸痛、呼吸困难、意识改变、抽搐、严重过敏等危急情况，请立即拨打当地急救电话或尽快前往急诊。"
        )

    normalized: Dict[str, Any] = {
        "triage_level": triage_level,
        "red_flags": _list(answer.get("red_flags")),
        "immediate_actions": _list(answer.get("immediate_actions")),
        "what_not_to_do": _list(answer.get("what_not_to_do")),
        "key_questions": _list(answer.get("key_questions")),
        "reasoning": str(answer.get("reasoning", "")).strip(),
        "uncertainty": str(answer.get("uncertainty", "")).strip(),
        "safety_notice": str(safety_notice).strip(),
        "citations_used": [],
    }

    # 计算 citations_used
    serialized = json.dumps(normalized, ensure_ascii=False)
    normalized["citations_used"] = _extract_citations_used(serialized)

    # 如果有证据但完全没引用，打一个标记，方便上层触发重写
    if evidence_list and len(normalized["citations_used"]) < 1:
        normalized["uncertainty"] = _append_uncertainty_tag(normalized.get("uncertainty", ""), "insufficient_citations")

    # 若没有本地证据，也标记一下
    if not evidence_list:
        normalized["uncertainty"] = _append_uncertainty_tag(normalized.get("uncertainty", ""), "no_local_evidence")

    evidence_quality = summarize_evidence_quality(evidence_list)
    if is_low_evidence(evidence_quality):
        normalized["uncertainty"] = _append_uncertainty_tag(normalized.get("uncertainty", ""), "low_local_evidence")
        if not normalized["key_questions"]:
            normalized["key_questions"] = _default_low_evidence_questions()
        reasoning = str(normalized.get("reasoning") or "").strip()
        prefix = "本地证据较少，以下判断仅作初步参考。"
        if prefix not in reasoning:
            normalized["reasoning"] = f"{prefix} {reasoning}".strip()

    # 确定性 guardrails v2：triage_level 下限提升 + citations_used 重算
    return _apply_guardrails(normalized, evidence_list=evidence_list)


# -------- Prompts（阶段2只保留“分诊JSON”所需链条；safe模式额外跑安全审查链） --------

TEMPLATE_SUGGESTION_JSON = """你是一名面向急诊分诊的AI医生助手。

语言要求：
- 你输出的 JSON 中，除 triage_level、uncertainty、citations_used 这些枚举/ID 字段外，所有自然语言字符串都必须为简体中文。
- 不要输出英文句子。

You MUST output a single JSON object and NOTHING ELSE. No markdown, no extra text.

JSON schema (keys must exist; arrays can be empty):
{{
  "triage_level": "EMERGENCY_NOW|URGENT_24H|ROUTINE|SELF_CARE",
  "red_flags": ["...each item should include citations like [E1] if supported..."],
  "immediate_actions": ["...each item MUST end with citations like [E1][E2]..."],
  "what_not_to_do": ["...optional, include citations if possible..."],
  "key_questions": ["...next questions to ask if info is insufficient..."],
  "reasoning": "...short, must include citations like [E1] for key points...",
  "uncertainty": "evidence_insufficient|needs_exam|needs_imaging|... (keep it brief)",
  "safety_notice": "...must clearly state AI limitations and emergency warning...",
  "citations_used": ["E1","E2"]
}}

Strict rules:
- Do NOT fabricate evidence. You may ONLY cite [E\\d] that exist in the evidence block.
- If evidence is insufficient, say so in reasoning/uncertainty and recommend offline examination/tests.
- Every key medical judgment and every immediate action MUST have citations if supported; otherwise say "insufficient evidence".

Additional safety rules:
- red_flags 只能列出“患者已明确提到或强烈提示”的危险信号（症状/体征）。不得用“信息不足/无法评估/证据不足”等元描述填充 red_flags；若未知请输出空数组 []。
- 不要仅因为信息不足就建议“立即呼叫急救/拨打120”。信息不足时优先在 key_questions 里提出需要澄清的问题，并给出“如出现X危险信号则立即急诊/呼救”的条件性提醒。

User input:
{user_text}

Current date: {date}

Evidence block (ONLY source of citations):
{evidence}
"""

TEMPLATE_JSON_FIX = """你是一个严格的 JSON 格式化器。

语言要求：
- 你输出的 JSON 中，除 triage_level、uncertainty、citations_used 这些枚举/ID 字段外，所有自然语言字符串都必须为简体中文。
- 不要输出英文句子。

Task:
- Convert the content into ONE valid JSON object ONLY (no extra text).
- The JSON MUST follow the exact schema below.
- Keep the original medical meaning as much as possible.
- Preserve ALL existing citations like [E1]. If missing for key points, add "insufficient evidence".

Schema:
{{
  "triage_level": "EMERGENCY_NOW|URGENT_24H|ROUTINE|SELF_CARE",
  "red_flags": ["..."],
  "immediate_actions": ["..."],
  "what_not_to_do": ["..."],
  "key_questions": ["..."],
  "reasoning": "...",
  "uncertainty": "...",
  "safety_notice": "...",
  "citations_used": ["E1","E2"]
}}

Strict constraints:
- Do NOT invent evidence. Only cite [E\\d] present in Evidence block.
- Output JSON only.

Evidence block:
{evidence}

Original content to convert:
{content}

JSON:
"""

TEMPLATE_CITATION_REWRITE = """你是一名资深医疗助手。

语言要求：
- 你输出的 JSON 中，除 triage_level、uncertainty、citations_used 这些枚举/ID 字段外，所有自然语言字符串都必须为简体中文。
- 不要输出英文句子。

The previous AI doctor's JSON is missing required citations or has format issues.

You MUST output ONE valid JSON object ONLY following the schema. Use ONLY the provided evidence block for citations.

Schema:
{{
  "triage_level": "EMERGENCY_NOW|URGENT_24H|ROUTINE|SELF_CARE",
  "red_flags": ["...each item include citations like [E1] if supported..."],
  "immediate_actions": ["...each item MUST end with citations like [E1][E2]..."],
  "what_not_to_do": ["...optional..."],
  "key_questions": ["..."],
  "reasoning": "...must include citations like [E1] for key points...",
  "uncertainty": "...",
  "safety_notice": "...",
  "citations_used": ["E1","E2"]
}}

Strict constraints:
- Do NOT invent evidence. Only cite [E\\d] present in Evidence block.
- Output JSON only.

Evidence block:
{evidence}

Previous content (to be rewritten):
{response}

JSON:
"""

# safe模式：安全审查链（可选）
TEMPLATE_SAFETY_STATE = """你是安全监督员。请检查AI医生的回复是否清楚说明：AI局限性 + 紧急情况警示。
如果缺失，请用简体中文给出一句话的补充建议。
AI医生回复：{suggestion}
"""

TEMPLATE_SAFETY_SERIOUS = """你是安全监督员。判断症状是否可能提示紧急/危急。
如可能严重，请用简体中文简要说明原因，并给出应对建议（例如尽快就医/急诊/拨打急救电话等）。
AI医生回复：{suggestion}
"""

TEMPLATE_SAFETY_ERRORS = """你是安全监督员。检查AI医生回复是否存在明显事实性错误或不安全表述。
如有，请用简体中文指出并给出简短纠正建议。
AI医生回复：{suggestion}
"""

TEMPLATE_SAFETY_MODIFICATION_JSON = """你是安全监督员。

语言要求：
- 你输出的 JSON 中，除 triage_level、uncertainty、citations_used 这些枚举/ID 字段外，所有自然语言字符串都必须为简体中文。
- 不要输出英文句子。

You MUST output ONE valid JSON object ONLY. No extra text.
You MUST preserve all citation markers like [E1] and MUST NOT delete them.
Do NOT add new citations that are not present in the JSON unless they already exist in the doctor's response.

Follow the suggestions below to modify the AI doctor's JSON while keeping the same schema:
- Safety notice improvement suggestion: {state_suggestion}
- Seriousness and emergency suggestion: {serious_suggestion}
- Factual error correction suggestion: {errors_suggestion}

AI doctor's JSON response (to modify):
{response}

Modified JSON:
"""


@dataclass
class TriageResult:
    answer_json: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    rag_query: str
    mode: str
    created_at: str
    rag_status: str = "unknown"  # ok|error|disabled
    rag_error: Optional[Dict[str, str]] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)


class TriageEngine:
    """
    单例引擎：复用LLM与Chain，避免每次请求重新构建，降低延迟。
    """

    def __init__(self):
        self._initialized = False
        self.llm = None

        self.chain_suggest: Optional[LLMChain] = None
        self.chain_json_fix: Optional[LLMChain] = None
        self.chain_citation_rewrite: Optional[LLMChain] = None

        self.chain_safety_state: Optional[LLMChain] = None
        self.chain_safety_serious: Optional[LLMChain] = None
        self.chain_safety_errors: Optional[LLMChain] = None
        self.chain_safety_modification: Optional[LLMChain] = None

    @staticmethod
    def _trace_step(trace: List[Dict[str, Any]], step: str, status: str, t0: float, **info) -> None:
        try:
            ms = int((time.perf_counter() - t0) * 1000)
        except Exception:
            ms = None
        entry: Dict[str, Any] = {"step": step, "status": status}
        if ms is not None:
            entry["ms"] = ms
        # Avoid leaking raw text; keep only small scalars.
        for k, v in info.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                entry[k] = v
            elif isinstance(v, list):
                entry[k] = v
            elif isinstance(v, dict):
                entry[k] = v
        trace.append(entry)

    def retrieve_evidence(
        self,
        rag_query: str,
        top_k: int,
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], str, Optional[Dict[str, str]]]:
        """Retrieve evidence with best-effort failure handling.

        Returns: (evidence_list, rag_status, rag_error)
        """

        self.init()

        if trace is None:
            trace = []

        evidence_list: List[Dict[str, Any]] = []
        rag_status = "unknown"
        rag_error: Optional[Dict[str, str]] = None

        if rag_retriever is not None:
            t0 = time.perf_counter()
            try:
                evidence_list = rag_retriever.retrieve(rag_query, top_k=top_k)
                rag_meta_getter = getattr(rag_retriever, "get_last_retrieval_meta", None)
                rag_meta = rag_meta_getter() if callable(rag_meta_getter) else {}
                rag_status = "ok"
                trace_info: Dict[str, Any] = {"top_k": top_k, "n_evidence": len(evidence_list)}
                if isinstance(rag_meta, dict):
                    for key in ("cache_hit", "cache_mode", "hybrid_enabled", "search_query"):
                        if key in rag_meta:
                            trace_info[key] = rag_meta.get(key)
                self._trace_step(trace, "rag.retrieve", "ok", t0, **trace_info)
                quality = summarize_evidence_quality(evidence_list)
                trace.append(
                    {
                        "step": "rag.evidence_quality",
                        "status": str(quality.get("level") or "unknown"),
                        "count": int(quality.get("count") or 0),
                        "reason": str(quality.get("reason") or ""),
                    }
                )
            except Exception as e:
                rag_status = "error"
                try:
                    logger.exception(
                        "rag.retrieve_failed | error_type=%s | message=%s | rag_query_prefix=%s",
                        type(e).__name__,
                        str(e),
                        (rag_query or "")[:300],
                    )
                except Exception:
                    pass

                rag_error = {"type": type(e).__name__, "message": str(e)}
                evidence_list = []
                self._trace_step(
                    trace,
                    "rag.retrieve",
                    "error",
                    t0,
                    top_k=top_k,
                    n_evidence=0,
                    error_type=type(e).__name__,
                )
                quality = summarize_evidence_quality(evidence_list)
                trace.append(
                    {
                        "step": "rag.evidence_quality",
                        "status": str(quality.get("level") or "unknown"),
                        "count": int(quality.get("count") or 0),
                        "reason": str(quality.get("reason") or ""),
                    }
                )
        else:
            evidence_list = []
            rag_status = "disabled"
            rag_error = {"type": "RAG_DISABLED", "message": "rag_retriever is not available"}
            trace.append(
                {
                    "step": "rag.retrieve",
                    "status": "disabled",
                    "top_k": top_k,
                    "n_evidence": 0,
                    "error_type": "RAG_DISABLED",
                }
            )
            quality = summarize_evidence_quality(evidence_list)
            trace.append(
                {
                    "step": "rag.evidence_quality",
                    "status": str(quality.get("level") or "unknown"),
                    "count": int(quality.get("count") or 0),
                    "reason": str(quality.get("reason") or ""),
                }
            )

        return evidence_list, rag_status, rag_error

    def suggest_answer_raw(
        self,
        user_text: str,
        evidence_block: str,
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self.init()
        if trace is None:
            trace = []
        assert self.chain_suggest is not None
        t0 = time.perf_counter()
        raw = self.chain_suggest.run(
            {
                "user_text": redact_pii_for_llm((user_text or "").strip()),
                "date": datetime.now(),
                "evidence": evidence_block,
            }
        )
        self._trace_step(trace, "llm.suggest", "ok", t0)
        return raw

    def ensure_answer_json(
        self,
        raw_text: str,
        evidence_block: str,
        evidence_list: List[Dict[str, Any]],
        trace: Optional[List[Dict[str, Any]]] = None,
        trace_step: str = "answer.ensure_json",
    ) -> Dict[str, Any]:
        self.init()
        if trace is None:
            trace = []
        t0 = time.perf_counter()
        answer_json = self._ensure_valid_answer_json(raw_text, evidence_block, evidence_list)
        self._trace_step(trace, trace_step, "ok", t0)
        return answer_json

    def run_safety_chain(
        self,
        answer_json: Dict[str, Any],
        evidence_block: str,
        evidence_list: List[Dict[str, Any]],
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.init()
        if trace is None:
            trace = []

        suggestion_text = json.dumps(answer_json or {}, ensure_ascii=False)

        assert self.chain_safety_state is not None
        assert self.chain_safety_serious is not None
        assert self.chain_safety_errors is not None
        assert self.chain_safety_modification is not None

        t0 = time.perf_counter()
        state = self.chain_safety_state.run({"suggestion": suggestion_text})
        self._trace_step(trace, "safety.state", "ok", t0)

        t0 = time.perf_counter()
        serious = self.chain_safety_serious.run({"suggestion": suggestion_text})
        self._trace_step(trace, "safety.serious", "ok", t0)

        t0 = time.perf_counter()
        errors = self.chain_safety_errors.run({"suggestion": suggestion_text})
        self._trace_step(trace, "safety.errors", "ok", t0)

        t0 = time.perf_counter()
        modified_raw = self.chain_safety_modification.run(
            {
                "response": suggestion_text,
                "state_suggestion": state,
                "serious_suggestion": serious,
                "errors_suggestion": errors,
            }
        )
        self._trace_step(trace, "safety.modify_json", "ok", t0)

        # Re-validate after safety modifications.
        return self.ensure_answer_json(
            modified_raw,
            evidence_block,
            evidence_list,
            trace=trace,
            trace_step="answer.ensure_json_after_safety",
        )

    def init(self):
        if self._initialized:
            return

        self.llm = config_llm.get_llm(temperature=0.0)

        self.chain_suggest = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["user_text", "date", "evidence"],
                template=TEMPLATE_SUGGESTION_JSON,
            ),
        )
        self.chain_json_fix = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content", "evidence"],
                template=TEMPLATE_JSON_FIX,
            ),
        )
        self.chain_citation_rewrite = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["response", "evidence"],
                template=TEMPLATE_CITATION_REWRITE,
            ),
        )

        # safe模式链条
        self.chain_safety_state = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["suggestion"],
                template=TEMPLATE_SAFETY_STATE,
            ),
        )
        self.chain_safety_serious = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["suggestion"],
                template=TEMPLATE_SAFETY_SERIOUS,
            ),
        )
        self.chain_safety_errors = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["suggestion"],
                template=TEMPLATE_SAFETY_ERRORS,
            ),
        )
        self.chain_safety_modification = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["response", "state_suggestion", "serious_suggestion", "errors_suggestion"],
                template=TEMPLATE_SAFETY_MODIFICATION_JSON,
            ),
        )

        self._initialized = True

    def _ensure_valid_answer_json(
        self,
        raw_text: str,
        evidence_block: str,
        evidence_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        将 raw_text 变成可 json.loads 的 dict，并确保至少有1个[E#]引用(如果有证据)。
        """
        candidate = _extract_first_json_object(raw_text)
        parsed = None
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None

        if parsed is None:
            assert self.chain_json_fix is not None
            fixed = self.chain_json_fix.run({"content": raw_text, "evidence": evidence_block})
            candidate2 = _extract_first_json_object(fixed) or fixed
            try:
                parsed = json.loads(candidate2)
            except Exception:
                parsed = {}

        normalized = _normalize_answer_schema(parsed, evidence_list=evidence_list)

        allowed_numbers = _allowed_citation_numbers(evidence_list)

        # 引用校验：1) 有证据但完全没有[E#]；2) 出现了不存在的EID引用
        serialized = json.dumps(normalized, ensure_ascii=False)
        invalid_nums = _invalid_citation_numbers(serialized, allowed_numbers)
        missing_all = bool(evidence_list) and _count_distinct_citations(serialized) < 1

        if missing_all or (invalid_nums and evidence_list):
            assert self.chain_citation_rewrite is not None
            rewritten = self.chain_citation_rewrite.run({"response": serialized, "evidence": evidence_block})
            candidate3 = _extract_first_json_object(rewritten) or rewritten
            try:
                parsed3 = json.loads(candidate3)
            except Exception:
                parsed3 = normalized
            normalized = _normalize_answer_schema(parsed3, evidence_list=evidence_list)

            # rewrite 后仍出现非法引用：严格剥离非法引用，保证“禁止引用不存在的EID”
            serialized_after = json.dumps(normalized, ensure_ascii=False)
            invalid_after = _invalid_citation_numbers(serialized_after, allowed_numbers)
            if invalid_after:
                normalized = _strip_invalid_citations_from_obj(normalized, allowed_numbers)
                normalized = _normalize_answer_schema(normalized, evidence_list=evidence_list)

        # 最终重算 citations_used（以最终文本为准）
        serialized2 = json.dumps(normalized, ensure_ascii=False)
        normalized["citations_used"] = _extract_citations_used(serialized2)

        return normalized

    def triage_once(
        self,
        user_text: str,
        top_k: int = 5,
        mode: str = "fast",
        clinical_record_text: str = "",
    ) -> TriageResult:
        """
        单次分诊API核心逻辑。
        - user_text：用户描述
        - top_k：RAG召回数
        - mode：fast(不跑安全链)/safe(跑安全链更慢更稳)
        - clinical_record_text：可选，将历史病历/规则摘要拼进去(未来扩展)
        """
        self.init()

        trace: List[Dict[str, Any]] = []

        if not user_text or not user_text.strip():
            raise ValueError("user_text is empty")

        mode = (mode or "fast").strip().lower()
        if mode not in ("fast", "safe"):
            mode = "fast"

        # 构造rag_query：阶段2先简单用user_text，后续可加query改写
        rag_query = user_text.strip()
        if clinical_record_text:
            rag_query = f"{rag_query}\n\nclinical_record:\n{clinical_record_text.strip()}"

        # Tool: RAG检索（失败不得中断整体分诊；但要记录可观测信息）
        evidence_list, rag_status, rag_error = self.retrieve_evidence(
            rag_query=rag_query,
            top_k=top_k,
            trace=trace,
        )

        evidence_block = _format_evidence_block(evidence_list)

        raw = self.suggest_answer_raw(user_text=user_text, evidence_block=evidence_block, trace=trace)
        answer_json = self.ensure_answer_json(
            raw_text=raw,
            evidence_block=evidence_block,
            evidence_list=evidence_list,
            trace=trace,
            trace_step="answer.ensure_json",
        )

        # Tool: safe模式安全审查链（显式步骤化）
        if mode == "safe":
            answer_json = self.run_safety_chain(
                answer_json=answer_json,
                evidence_block=evidence_block,
                evidence_list=evidence_list,
                trace=trace,
            )

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return TriageResult(
            answer_json=answer_json,
            evidence=evidence_list,
            rag_query=rag_query,
            mode=mode,
            created_at=created_at,
            rag_status=rag_status,
            rag_error=rag_error,
            trace=trace,
        )


# 进程级单例
_ENGINE = TriageEngine()


def triage_once(
    user_text: str,
    top_k: int = 5,
    mode: str = "fast",
    clinical_record_path: Optional[str] = None,
    longitudinal_records: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    供FastAPI调用的轻量函数包装，返回可直接json序列化的dict。
    clinical_record_path：可选，读取规则摘要/病历文本（不强依赖）
    """
    clinical_record_text = ""
    if clinical_record_path:
        try:
            if os.path.exists(clinical_record_path):
                with open(clinical_record_path, "r", encoding="utf-8") as f:
                    clinical_record_text = f.read()
        except Exception:
            clinical_record_text = ""

    result = _ENGINE.triage_once(
        user_text=user_text,
        top_k=top_k,
        mode=mode,
        clinical_record_text=clinical_record_text,
    )

    return triage_step_build_payload(
        answer_json=result.answer_json,
        evidence_list=result.evidence,
        rag_query=result.rag_query,
        mode=result.mode,
        created_at=result.created_at,
        rag_status=result.rag_status,
        rag_error=result.rag_error,
        trace=result.trace,
        longitudinal_records=longitudinal_records,
        clinical_record_text=clinical_record_text,
    )


def build_rag_query(user_text: str, clinical_record_text: str = "") -> str:
    """Build the internal RAG query string from user input + optional clinical record text."""

    q = (user_text or "").strip()
    if clinical_record_text and clinical_record_text.strip():
        q = f"{q}\n\nclinical_record:\n{clinical_record_text.strip()}"
    return q


def format_evidence_block(evidence_list: List[Dict[str, Any]]) -> str:
    """Public wrapper for evidence block formatting (for orchestration layers)."""

    return _format_evidence_block(evidence_list)


def triage_step_retrieve(
    rag_query: str,
    top_k: int = 5,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Step 1: retrieve evidence for a prepared rag_query."""

    if trace is None:
        trace = []
    evidence_list, rag_status, rag_error = _ENGINE.retrieve_evidence(
        rag_query=rag_query,
        top_k=top_k,
        trace=trace,
    )
    return {
        "rag_query": rag_query,
        "evidence": evidence_list,
        "rag_status": rag_status,
        "rag_error": rag_error,
        "trace": trace,
    }


def triage_step_assess(
    user_text: str,
    evidence_list: List[Dict[str, Any]],
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Step 2: generate + validate triage answer JSON from user_text + evidence."""

    if trace is None:
        trace = []
    evidence_block = _format_evidence_block(evidence_list)
    raw = _ENGINE.suggest_answer_raw(user_text=user_text, evidence_block=evidence_block, trace=trace)
    answer_json = _ENGINE.ensure_answer_json(
        raw_text=raw,
        evidence_block=evidence_block,
        evidence_list=evidence_list,
        trace=trace,
        trace_step="answer.ensure_json",
    )
    return {"evidence_block": evidence_block, "raw": raw, "answer": answer_json, "trace": trace}


def triage_step_safety(
    answer_json: Dict[str, Any],
    evidence_block: str,
    evidence_list: List[Dict[str, Any]],
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Step 3: optional safety review chain, returns updated answer_json."""

    if trace is None:
        trace = []
    out = _ENGINE.run_safety_chain(
        answer_json=answer_json,
        evidence_block=evidence_block,
        evidence_list=evidence_list,
        trace=trace,
    )
    return {"answer": out, "trace": trace}


def triage_step_build_payload(
    answer_json: Dict[str, Any],
    evidence_list: List[Dict[str, Any]],
    rag_query: str,
    mode: str,
    created_at: str,
    rag_status: str = "unknown",
    rag_error: Optional[Dict[str, str]] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
    longitudinal_records: Optional[List[Any]] = None,
    clinical_record_text: str = "",
) -> Dict[str, Any]:
    """Step 4: build canonical payload and attach observability meta."""

    if trace is None:
        trace = []
    safe_answer = _apply_record_safety(
        answer_json,
        clinical_record_text,
        longitudinal_records=longitudinal_records,
        trace=trace,
    )

    payload = build_triage_payload(
        answer=safe_answer,
        evidence=evidence_list,
        rag_query=rag_query,
        mode=mode,
        created_at=created_at,
    )

    meta = payload.get("meta")
    if isinstance(meta, dict):
        meta["rag_status"] = rag_status
        meta["evidence_quality"] = summarize_evidence_quality(evidence_list)
        if isinstance(rag_error, dict) and rag_error:
            meta["rag_error"] = rag_error
        if isinstance(trace, list) and trace:
            meta["trace"] = trace

    return payload
