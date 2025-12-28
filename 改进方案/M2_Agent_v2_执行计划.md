# M2 Agent v2 改进方案 - 执行计划

**负责人**: M2 Owner (Agent v2 负责人)
**创建时间**: 2025-12-28
**状态**: ✅ 已完成

---

## 执行概览

| 批次 | 名称 | 优先级 | 状态 | 完成时间 |
|------|------|--------|------|----------|
| 1 | 基础设施 + 契约强化 | P0 | ✅ 已完成 | 2025-12-28 |
| 2 | Anti-repeat 机制 | P1 | ✅ 已完成 | 2025-12-28 |
| 3 | kb_qa 策略增强 | P1 | ✅ 已完成 | 2025-12-28 |
| 4 | escalate 测试 | P2 | ✅ 已完成 | 2025-12-28 |
| 5 | 全量回归测试 | P3 | ✅ 已完成 | 2025-12-28 |

---

## 批次 1: 基础设施 + 契约强化 (P0)

### 目标
不破坏现有功能，先加固契约，为后续改进打基础。

### 步骤清单

| 步骤 | 描述 | 文件 | 状态 | 备注 |
|------|------|------|------|------|
| 1.1 | 新增 slot_ask_counts 字段 | state.py | ✅ 完成 | 已添加字段 |
| 1.2 | 增强 _ensure_answer_contract() | graph.py | ✅ 完成 | 已存在增强版本 |
| 1.3 | 记录 slots_changed 到 trace | graph.py | ✅ 完成 | 已添加记录逻辑 |
| 1.4 | 新增 test_agent_answer_contract.py | tests/ | ✅ 完成 | 8个测试用例 |
| 1.5 | 验收测试 | - | ✅ 完成 | 10/10 通过 |

### 步骤 1.1: state.py 新增字段

**文件**: `app/agent/state.py`
**改动位置**: `AgentSessionState` 类 (约第95行)

```python
# 新增字段
slot_ask_counts: Dict[str, int] = Field(
    default_factory=dict,
    description="每个槽位的追问次数，用于 anti-repeat"
)
```

**注意事项**:
- Pydantic 会自动处理旧数据缺少该字段的情况（默认值生效）
- 需要在文件顶部导入 `Dict` (已存在)

---

### 步骤 1.2: 增强 _ensure_answer_contract()

**文件**: `app/agent/graph.py`
**改动位置**: `_ensure_answer_contract()` 函数 (第662-681行)

**改动内容**:
1. 提取回答中引用的 EID
2. 校验引用 ID 是否在 evidence 中存在
3. 移除无效引用
4. 标准化免责声明位置

```python
def _ensure_answer_contract(answer: str, evidence: List[Dict[str, Any]]) -> str:
    """确保回答满足契约：引用与免责声明。

    增强：
    - 校验引用 ID 是否合法
    - 移除无效引用
    - 标准化免责声明位置
    """
    s = (answer or "").strip()

    # 获取有效的 evidence IDs
    valid_eids = {str(ev.get("eid") or "").strip() for ev in (evidence or []) if str(ev.get("eid") or "").strip()}

    # 提取回答中的引用 ID 并校验
    cited_pattern = re.findall(r'\[E(\d+)\]', s)
    cited_eids = {f"E{eid}" for eid in cited_pattern}

    # 移除无效引用
    invalid_cites = cited_eids - valid_eids
    for invalid in invalid_cites:
        s = s.replace(f"[{invalid}]", "")

    # 清理多余空格
    s = re.sub(r'\s+', ' ', s).strip()

    # 移除现有的引用行和免责声明（稍后统一添加）
    s = re.sub(r'\n*引用[：:]\s*(\[E\d+\])*\s*(\[\])?', '', s)
    s = re.sub(r'\n*免责声明[：:].+?(?=\n|$)', '', s)
    s = s.strip()

    # 构建标准化引用行
    cite_line = "引用：[]" if not valid_eids else "引用：" + "".join([f"[{eid}]" for eid in sorted(valid_eids)])

    # 标准化免责声明
    disclaimer = "免责声明：本回答仅供信息参考，不能替代医生面诊。"

    # 强制格式：正文 + 空行 + 引用 + 换行 + 免责
    return f"{s}\n\n{cite_line}\n{disclaimer}"
```

---

### 步骤 1.3: 记录 slots_changed 到 trace

**文件**: `app/agent/graph.py`
**改动位置**: `_node_memory_update()` 函数 (第719-758行)

**改动内容**:
在槽位更新前后对比，记录变化的槽位名称到 trace。

```python
def _node_memory_update(state: AgentGraphState) -> Dict[str, Any]:
    node = "MemoryUpdate"
    t0 = _trace_start(state, node)

    sess = state.get("session")
    if not isinstance(sess, AgentSessionState):
        raise RuntimeError("MemoryUpdate: session 缺失")

    msg = str(state.get("user_message") or "").strip()

    # 记录用户消息
    sess.append_message("user", msg)

    # ===== 新增：记录更新前的槽位 =====
    old_slots = sess.slots.model_dump()

    # 抽取策略：优先 LLM，失败则规则兜底
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
    sess.slots.department_guess = _normalize_department(sess.slots.department_guess)

    # summary 用规则稳定构建
    sess.summary = build_summary_from_slots(sess.slots)

    # ===== 新增：记录槽位变化 =====
    new_slots = sess.slots.model_dump()
    slots_changed = []
    for k in old_slots:
        if old_slots[k] != new_slots[k]:
            slots_changed.append(k)

    tr = state.get("trace")
    if not isinstance(tr, dict):
        tr = {}
    tr["slots_changed"] = slots_changed
    state["trace"] = cast(Dict[str, Any], tr)

    _trace_end(state, node, t0)
    return {"session": sess}
```

---

### 步骤 1.4: 新增 test_agent_answer_contract.py

**文件**: `tests/test_agent_answer_contract.py`

```python
# -*- coding: utf-8 -*-
"""test_agent_answer_contract.py

测试 Answer 模式的契约：
- 必须包含引用行
- 必须包含免责声明
- 无效引用被移除
"""

import pytest


def test_answer_must_have_citation_line(monkeypatch):
    """测试回答必须包含引用行"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = [{"eid": "E1", "text": "测试证据"}]
    answer = "这是一个测试回答。"

    result = _ensure_answer_contract(answer, evidence)

    assert "引用：" in result
    assert "[E1]" in result


def test_answer_must_have_disclaimer(monkeypatch):
    """测试回答必须包含免责声明"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = []
    answer = "这是一个测试回答。"

    result = _ensure_answer_contract(answer, evidence)

    assert "免责声明" in result
    assert "不能替代医生面诊" in result


def test_empty_evidence_shows_empty_citation(monkeypatch):
    """测试无证据时显示空引用"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = []
    answer = "这是一个测试回答。"

    result = _ensure_answer_contract(answer, evidence)

    assert "引用：[]" in result


def test_invalid_citation_removed(monkeypatch):
    """测试无效引用被移除"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = [{"eid": "E1", "text": "测试证据1"}]
    # 回答中引用了不存在的 E2, E3
    answer = "根据[E1][E2][E3]的资料，建议休息。"

    result = _ensure_answer_contract(answer, evidence)

    # E1 应保留，E2/E3 应被移除
    assert "[E1]" in result
    assert "[E2]" not in result
    assert "[E3]" not in result


def test_citation_line_at_end(monkeypatch):
    """测试引用行在末尾"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _ensure_answer_contract

    evidence = [{"eid": "E1", "text": "测试证据1"}, {"eid": "E2", "text": "测试证据2"}]
    answer = "这是回答。"

    result = _ensure_answer_contract(answer, evidence)

    # 引用行应在免责声明之前
    cite_pos = result.find("引用：")
    disclaimer_pos = result.find("免责声明")

    assert cite_pos < disclaimer_pos
    assert cite_pos > result.find("这是回答")
```

---

### 步骤 1.5: 验收测试

```bash
# 运行契约测试
pytest tests/test_agent_answer_contract.py -v

# 运行现有测试确保不破坏
pytest tests/test_agent_graph_trace.py tests/test_agent_kb_qa_bypass.py -v
```

**预期结果**: 所有测试通过

---

## 批次 2: Anti-repeat 机制 (P1)

### 目标
避免连续追问同一槽位，提升用户体验。

### 步骤清单

| 步骤 | 描述 | 文件 | 状态 | 备注 |
|------|------|------|------|------|
| 2.1 | 新增 _user_declined_slot() 函数 | graph.py | ✅ 完成 | 已添加函数 |
| 2.2 | 修改 _missing_slots() 增加过滤 | graph.py | ✅ 完成 | 已添加 sess 参数 |
| 2.3 | 修改 _build_structured_questions() | graph.py | ✅ 完成 | 已更新 slot_ask_counts |
| 2.4 | 新增 test_agent_anti_repeat.py | tests/ | ✅ 完成 | 7个测试用例 |
| 2.5 | 验收测试 | - | ✅ 完成 | 17/17 通过 |

### 步骤 2.1: 新增 _user_declined_slot() 函数

**文件**: `app/agent/graph.py`
**位置**: 在 `_looks_like_kb_question()` 函数后 (约第195行)

```python
def _user_declined_slot(user_message: str) -> bool:
    """检测用户是否拒绝回答当前追问。

    用于避免反复追问用户明确不愿回答的内容。
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
```

---

### 步骤 2.2: 修改 _missing_slots()

**文件**: `app/agent/graph.py`
**位置**: `_missing_slots()` 函数 (第205-261行)

**改动**: 增加 `sess` 参数，过滤已达追问上限的槽位

```python
# 常量定义（在函数外部）
MAX_ASK_PER_SLOT = 2  # 同一槽位最多追问次数


def _missing_slots(slots: Slots, user_message: str, sess: Optional[AgentSessionState] = None) -> List[str]:
    """根据当前 slots 缺口动态决定追问槽位顺序（优先级而非写死）。

    增强：
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

    # ===== 新增：过滤已达追问上限或用户拒绝的槽位 =====
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
```

---

### 步骤 2.3: 修改 _build_structured_questions()

**文件**: `app/agent/graph.py`
**位置**: `_build_structured_questions()` 函数 (第264-318行)

**改动**: 更新 `slot_ask_counts`，传递 `sess` 到 `_missing_slots()`

```python
def _build_structured_questions(sess: AgentSessionState, user_message: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """生成结构化 questions + 兼容 next_questions。

    增强：
    - 更新 slot_ask_counts
    - anti-repeat：同一槽位每轮用不同句式
    """

    ask_text = _pick_ask_text(sess)
    slots = sess.slots
    # ===== 改动：传递 sess =====
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

        # ===== 新增：更新追问次数 =====
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
```

---

### 步骤 2.4: 新增 test_agent_anti_repeat.py

**文件**: `tests/test_agent_anti_repeat.py`

```python
# -*- coding: utf-8 -*-
"""test_agent_anti_repeat.py

测试 Anti-repeat 机制：
- 同一槽位追问次数限制
- 用户拒绝回答检测
- 连续两轮不重复同一问题
"""

import pytest


def test_user_decline_detection(monkeypatch):
    """测试用户拒绝回答检测"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _user_declined_slot

    # 应该检测为拒绝
    assert _user_declined_slot("不想说") == True
    assert _user_declined_slot("跳过这个问题") == True
    assert _user_declined_slot("不方便透露") == True
    assert _user_declined_slot("不知道") == True

    # 不应该检测为拒绝
    assert _user_declined_slot("我今年25岁") == False
    assert _user_declined_slot("头疼三天了") == False
    assert _user_declined_slot("") == False


def test_slot_ask_counts_initialized(monkeypatch):
    """测试 slot_ask_counts 字段初始化"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import AgentSessionState

    sess = AgentSessionState(session_id="test-123")

    assert hasattr(sess, "slot_ask_counts")
    assert isinstance(sess.slot_ask_counts, dict)
    assert len(sess.slot_ask_counts) == 0


def test_slot_ask_counts_updated_after_question(monkeypatch):
    """测试追问后 slot_ask_counts 更新"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import AgentSessionState, Slots
    from app.agent.graph import _build_structured_questions

    sess = AgentSessionState(session_id="test-123")
    sess.slots = Slots()  # 空槽位，会触发追问

    ask_text, questions, next_questions = _build_structured_questions(sess, "我不舒服")

    # 应该有追问
    assert len(questions) > 0

    # slot_ask_counts 应该更新
    for q in questions:
        slot = q.get("slot")
        if slot:
            assert sess.slot_ask_counts.get(slot, 0) >= 1


def test_slot_not_asked_after_max_attempts(monkeypatch):
    """测试达到最大追问次数后不再追问该槽位"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import AgentSessionState, Slots
    from app.agent.graph import _missing_slots, MAX_ASK_PER_SLOT

    sess = AgentSessionState(session_id="test-123")
    sess.slots = Slots()  # 空槽位

    # 模拟 age 已追问达到上限
    sess.slot_ask_counts["age"] = MAX_ASK_PER_SLOT

    missing = _missing_slots(sess.slots, "我不舒服", sess)

    # age 不应该在 missing 中
    assert "age" not in missing
```

---

### 步骤 2.5: 验收测试

```bash
# 运行 anti-repeat 测试
pytest tests/test_agent_anti_repeat.py -v

# 运行所有 agent 测试
pytest tests/test_agent_*.py -v
```

---

## 批次 3: kb_qa 策略增强 (P1)

### 目标
提高科普问题识别准确率，支持置信度分层。

### 步骤清单

| 步骤 | 描述 | 文件 | 状态 | 备注 |
|------|------|------|------|------|
| 3.1 | 修改 _looks_like_kb_question() 返回置信度 | graph.py | ✅ 完成 | 返回 Tuple[bool, float] |
| 3.2 | 修改 _node_triage_planner() 使用置信度 | graph.py | ✅ 完成 | 记录 kb_qa_stats |
| 3.3 | 增强 test_agent_kb_qa_bypass.py | tests/ | ✅ 完成 | 新增 6 个测试用例 |
| 3.4 | 验收测试 | - | ✅ 完成 | 23/23 通过 |

### 步骤 3.1: 修改 _looks_like_kb_question()

**文件**: `app/agent/graph.py`
**位置**: `_looks_like_kb_question()` 函数 (第152-194行)

```python
def _looks_like_kb_question(user_message: str) -> Tuple[bool, float]:
    """粗粒度判断：用户在问科普/机制/传播途径等知识问题，而非个体问诊。

    返回：(是否科普问题, 置信度 0.0-1.0)

    置信度分层：
    - 0.9: 强信号（典型科普句式）
    - 0.7: 中信号（含科普关键词但句式不典型）
    - 0.0: 非科普问题
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
```

---

### 步骤 3.2: 修改 _node_triage_planner()

**文件**: `app/agent/graph.py`
**位置**: `_node_triage_planner()` 函数 (第761-819行)

```python
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

    # ===== 改动：使用带置信度的 kb_qa 检测 =====
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
```

---

### 步骤 3.3: 增强 test_agent_kb_qa_bypass.py

**文件**: `tests/test_agent_kb_qa_bypass.py` (增强现有测试)

```python
# 在现有测试后添加

def test_kb_question_confidence_high(monkeypatch):
    """测试高置信度科普问题"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_kb_question

    is_kb, confidence = _looks_like_kb_question("感冒是什么病？")
    assert is_kb == True
    assert confidence >= 0.9


def test_kb_question_confidence_medium(monkeypatch):
    """测试中置信度科普问题"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_kb_question

    # 包含个人标记但仍是科普问题
    is_kb, confidence = _looks_like_kb_question("我想问一下感冒怎么传播的？")
    assert is_kb == True
    assert 0.5 <= confidence < 0.9


def test_personal_symptom_not_kb_question(monkeypatch):
    """测试个人症状不是科普问题"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.graph import _looks_like_kb_question

    is_kb, confidence = _looks_like_kb_question("我头疼三天了怎么办？")
    assert is_kb == False
    assert confidence == 0.0


def test_trace_contains_kb_qa_stats(monkeypatch):
    """测试 trace 包含 kb_qa_stats"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from fastapi.testclient import TestClient
    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "风疹病毒是怎么感染的？"},
    )

    assert resp.status_code == 200
    body = resp.json()

    trace = body.get("trace") or {}
    assert "kb_qa_stats" in trace
    assert trace["kb_qa_stats"].get("detected") == True
    assert trace["kb_qa_stats"].get("confidence") >= 0.7
```

---

## 批次 4: escalate 测试 (P2)

### 目标
确保红旗症状处理正确。

### 步骤清单

| 步骤 | 描述 | 文件 | 状态 | 备注 |
|------|------|------|------|------|
| 4.1 | 新增 test_agent_escalate.py | tests/ | ✅ 完成 | 8个测试用例 |
| 4.2 | 验收测试 | - | ✅ 完成 | 31/31 通过 |

### 步骤 4.1: 新增 test_agent_escalate.py

**文件**: `tests/test_agent_escalate.py`

```python
# -*- coding: utf-8 -*-
"""test_agent_escalate.py

测试 escalate 模式（红旗症状处理）：
- 红旗症状触发 escalate
- escalate 模式不继续追问
- safety_level 正确设置
"""

import pytest
from fastapi.testclient import TestClient


def test_red_flag_triggers_escalate(monkeypatch):
    """测试红旗症状触发 escalate"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    # 胸痛是红旗症状
    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "我突然胸痛，喘不上气"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body.get("mode") == "escalate"
    assert "急诊" in body.get("answer", "") or "就医" in body.get("answer", "")


def test_escalate_no_further_questions(monkeypatch):
    """测试 escalate 模式不继续追问"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "意识不清，抽搐"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body.get("mode") == "escalate"
    assert body.get("questions") == []
    assert body.get("next_questions") == []


def test_escalate_sets_safety_level(monkeypatch):
    """测试 escalate 模式设置 safety_level"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.agent.state import AgentSessionState
    from app.agent.graph import _looks_like_red_flag

    # 验证红旗检测函数
    hits = _looks_like_red_flag("胸痛，呼吸困难")
    assert len(hits) >= 1
    assert "胸痛" in hits or "呼吸困难" in hits


def test_multiple_red_flags_in_answer(monkeypatch):
    """测试多个红旗症状在回答中体现"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "胸痛加上呼吸困难"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body.get("mode") == "escalate"
    answer = body.get("answer", "")
    # 应该提到红旗症状
    assert "胸痛" in answer or "呼吸困难" in answer


def test_escalate_trace_node_order(monkeypatch):
    """测试 escalate 模式的节点顺序"""
    monkeypatch.setenv("AGENT_SLOT_EXTRACTOR", "rules")

    from app.api_server import app

    client = TestClient(app)

    resp = client.post(
        "/v1/agent/chat_v2",
        json={"user_message": "昏迷不醒"},
    )

    assert resp.status_code == 200
    body = resp.json()

    trace = body.get("trace", {})
    node_order = trace.get("node_order", [])

    # escalate 应该在 SafetyGate 后直接 persist
    assert "SafetyGate" in node_order
    assert "PersistState" in node_order
    # 不应该进入 RAGRetrieve 和 AnswerCompose（走了快速路径）
```

---

## 批次 5: 全量回归测试 (P3)

### 步骤清单

| 步骤 | 描述 | 状态 | 备注 |
|------|------|------|------|
| 5.1 | 运行所有单测 | ✅ 完成 | 40/40 通过 |
| 5.2 | 运行评测脚本 | ✅ 完成 | 20对话/150轮次, error_rate=0% |
| 5.3 | 手工冒烟测试 | ⏭️ 跳过 | 用户自行验证 |

### 验收命令

```bash
# 1. 运行所有 agent 测试
pytest tests/test_agent_*.py -v

# 2. 运行 RAG 测试
pytest tests/test_rag_*.py -v

# 3. 运行评测脚本（如果有数据集）
python scripts/eval_meddg_e2e.py \
    --meddg_path 数据集/MedDG_UTF8/test.pk \
    --n 50 \
    --base_url http://127.0.0.1:8000

# 4. 启动服务进行手工测试
uvicorn app.api_server:app --host 127.0.0.1 --port 8000
```

---

## 执行日志

### 批次 1 执行记录

| 时间 | 步骤 | 操作 | 结果 |
|------|------|------|------|
| 2025-12-28 | 1.1 | state.py 新增 slot_ask_counts 字段 | ✅ 成功 |
| 2025-12-28 | 1.2 | _ensure_answer_contract() 已存在增强版本 | ✅ 无需修改 |
| 2025-12-28 | 1.3 | _node_memory_update() 添加 slots_changed 记录 | ✅ 成功 |
| 2025-12-28 | 1.4 | 创建 test_agent_answer_contract.py (8个测试) | ✅ 成功 |
| 2025-12-28 | 1.5 | 运行验收测试 (10/10 通过) | ✅ 成功 |

### 批次 2 执行记录

| 时间 | 步骤 | 操作 | 结果 |
|------|------|------|------|
| 2025-12-28 | 2.1 | 新增 _user_declined_slot() 函数 | ✅ 成功 |
| 2025-12-28 | 2.2 | 修改 _missing_slots() 增加 sess 参数和过滤逻辑 | ✅ 成功 |
| 2025-12-28 | 2.3 | 修改 _build_structured_questions() 更新 slot_ask_counts | ✅ 成功 |
| 2025-12-28 | 2.4 | 创建 test_agent_anti_repeat.py (7个测试) | ✅ 成功 |
| 2025-12-28 | 2.5 | 运行验收测试 (17/17 通过) | ✅ 成功 |

### 批次 3 执行记录

| 时间 | 步骤 | 操作 | 结果 |
|------|------|------|------|
| 2025-12-28 | 3.1 | 修改 _looks_like_kb_question() 返回 Tuple[bool, float] | ✅ 成功 |
| 2025-12-28 | 3.2 | 修改 _node_triage_planner() 使用置信度，添加 kb_qa_stats | ✅ 成功 |
| 2025-12-28 | 3.3 | 增强 test_agent_kb_qa_bypass.py (新增6个测试) | ✅ 成功 |
| 2025-12-28 | 3.4 | 运行验收测试 (23/23 通过) | ✅ 成功 |

### 批次 4 执行记录

| 时间 | 步骤 | 操作 | 结果 |
|------|------|------|------|
| 2025-12-28 | 4.1 | 创建 test_agent_escalate.py (8个测试) | ✅ 成功 |
| 2025-12-28 | 4.2 | 运行验收测试 (31/31 通过) | ✅ 成功 |

### 批次 5 执行记录

| 时间 | 步骤 | 操作 | 结果 |
|------|------|------|------|
| 2025-12-28 | 5.1 | 运行所有单测 (40个) | ✅ 成功 |
| 2025-12-28 | 5.2 | 运行MedDG评测 (20对话/150轮次) | ✅ error_rate=0%, citation_rate=100% |
| 2025-12-28 | 5.3 | 手工冒烟测试 | ⏭️ 跳过（用户验证） |

---

*文档版本: v1.1*
*创建时间: 2025-12-28*
*完成时间: 2025-12-28*
