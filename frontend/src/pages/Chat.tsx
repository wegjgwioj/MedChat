import { useEffect, useMemo, useRef, useState } from 'react'
import EvidencePanel from '../components/EvidencePanel'
import FollowUpPanel from '../components/FollowUpPanel'
import DebugTrace from '../components/DebugTrace'
import { chatV2 } from '../lib/api'
import type { AgentChatV2Mode, AgentChatV2Response } from '../types/agent'

type Role = 'user' | 'assistant'

type UiMessage = {
  id: string
  role: Role
  content: string
  createdAt: number
  mode?: AgentChatV2Mode
  askText?: string
  questions?: AgentChatV2Response['questions']
  nextQuestions?: string[]
  citations?: AgentChatV2Response['citations']
  slots?: AgentChatV2Response['slots']
  summary?: string
  trace?: AgentChatV2Response['trace']
  askSig?: string
  isError?: boolean
}

const SESSION_ID_KEY = 'medchat_session_id'

function uuid() {
  return (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`) as string
}

function safeLocalStorageGet(key: string) {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function safeLocalStorageSet(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // 忽略：某些环境可能禁用 localStorage
  }
}

function safeLocalStorageRemove(key: string) {
  try {
    localStorage.removeItem(key)
  } catch {
    // ignore
  }
}

function renderWithEvidenceMarkers(text: string) {
  const parts: Array<{ t: string; isEid: boolean }> = []
  const re = /\[E\d+\]/g
  let lastIndex = 0
  for (const match of text.matchAll(re)) {
    const start = match.index ?? 0
    const end = start + match[0].length
    if (start > lastIndex) {
      parts.push({ t: text.slice(lastIndex, start), isEid: false })
    }
    parts.push({ t: match[0], isEid: true })
    lastIndex = end
  }
  if (lastIndex < text.length) {
    parts.push({ t: text.slice(lastIndex), isEid: false })
  }

  return (
    <span>
      {parts.map((p, i) =>
        p.isEid ? (
          <mark key={i} className="eid">
            {p.t}
          </mark>
        ) : (
          <span key={i}>{p.t}</span>
        ),
      )}
    </span>
  )
}

function normalizeNonEmptyQuestions(items: unknown): string[] {
  if (!Array.isArray(items)) return []
  const out: string[] = []
  const seen = new Set<string>()
  for (const it of items) {
    const v = String(it ?? '').trim()
    if (!v) continue
    if (seen.has(v)) continue
    seen.add(v)
    out.push(v)
  }
  return out
}

function normalizeQuestions(items: unknown): NonNullable<AgentChatV2Response['questions']> {
  if (!Array.isArray(items)) return []
  const out: NonNullable<AgentChatV2Response['questions']> = []
  for (const it of items) {
    if (!it || typeof it !== 'object') continue
    const anyIt = it as Record<string, unknown>
    const slot = String(anyIt.slot ?? '').trim()
    const question = String(anyIt.question ?? '').trim()
    if (!slot || !question) continue
    out.push({
      slot,
      question,
      type: (anyIt.type as any) ?? 'text',
      placeholder: typeof anyIt.placeholder === 'string' ? anyIt.placeholder : undefined,
      choices: Array.isArray(anyIt.choices) ? (anyIt.choices as any[]).map((x) => String(x)) : undefined,
      range: Array.isArray(anyIt.range) && anyIt.range.length === 2 ? (anyIt.range as any) : undefined,
    })
  }
  return out.slice(0, 3)
}

function buildAskSignature(askText: string, questions: NonNullable<AgentChatV2Response['questions']>, nextQuestions: string[]) {
  const qPart = questions.map((q) => `${q.slot}:${String(q.question ?? '').trim()}`).join('|')
  const nqPart = nextQuestions.map((q) => String(q ?? '').trim()).join('|')
  return `${String(askText ?? '').trim()}||${qPart}||${nqPart}`
}

export default function Chat() {
  const [sessionId, setSessionId] = useState<string>(() => (safeLocalStorageGet(SESSION_ID_KEY) ?? '').trim())

  const [messages, setMessages] = useState<UiMessage[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [topK, setTopK] = useState(5)
  const [topN, setTopN] = useState(30)
  const [useRerank, setUseRerank] = useState(true)

  const [debugEnabled, setDebugEnabled] = useState(false)

  const bottomRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const inFlightRef = useRef<AbortController | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length])

  const latestAssistant = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant') return messages[i]
    }
    return null
  }, [messages])

  function clearSession() {
    safeLocalStorageRemove(SESSION_ID_KEY)
    setSessionId('')
    setMessages([])
    setInput('')

    // 如果正在请求中，主动中断
    try {
      inFlightRef.current?.abort()
    } catch {
      // ignore
    } finally {
      inFlightRef.current = null
      setIsSending(false)
    }
  }

  function appendMessage(m: UiMessage) {
    setMessages((prev) => [...prev, m])
  }

  function upsertAssistantAsk(m: UiMessage) {
    setMessages((prev) => {
      if (!prev.length) return [...prev, m]
      const last = prev[prev.length - 1]
      if (last.role === 'assistant' && last.mode === 'ask' && last.askSig && m.askSig && last.askSig === m.askSig) {
        const next = [...prev]
        next[next.length - 1] = { ...last, ...m, id: last.id }
        return next
      }
      return [...prev, m]
    })
  }

  function updateSessionId(next: string) {
    const sid = String(next ?? '').trim()
    if (!sid) return
    setSessionId(sid)
    safeLocalStorageSet(SESSION_ID_KEY, sid)
  }

  function prefillInput(template: string) {
    const t = String(template ?? '')
    setInput(t)
    // 下一帧聚焦，避免 React state 更新时机问题
    setTimeout(() => {
      try {
        inputRef.current?.focus()
      } catch {
        // ignore
      }
    }, 0)
  }

  async function sendText(text: string) {
    const t = String(text ?? '').trim()
    if (!t || isSending) return

    // 防复读护栏：如果用户把追问问题文本原样发出，阻止发送
    if (latestAssistant?.role === 'assistant' && latestAssistant?.mode === 'ask') {
      // 收集所有追问问题文本（包括结构化 questions 和 fallback nextQuestions）
      const structuredQs = (latestAssistant.questions ?? []).map((q) => String(q.question ?? '').trim()).filter(Boolean)
      const fallbackQs = normalizeNonEmptyQuestions(latestAssistant.nextQuestions ?? [])
      const allQuestions = [...new Set([...structuredQs, ...fallbackQs])]

      // 规范化用户输入：去除可能的 `- ` 前缀
      const normalizedInput = t.replace(/^[-–—]\s*/, '').trim()

      // 策略1：精确匹配
      const exactHit = allQuestions.find((q) => {
        const normalizedQ = q.trim()
        return normalizedQ === t || normalizedQ === normalizedInput
      })

      // 策略2：模糊匹配 - 检测用户是否在"问问题"而非"回答问题"
      // 条件：(1) 以问号结尾 (2) 包含追问相关的关键词
      const isQuestion = /[？?]$/.test(normalizedInput)
      const questionKeywords = [
        '年龄', '多大', '几岁', '岁数',
        '性别', '男女',
        '多久', '多长时间', '持续', '什么时候',
        '程度', '几分', '疼痛', '严重',
        '症状', '不舒服', '哪里',
        '发烧', '发热', '体温',
      ]
      const containsKeyword = questionKeywords.some((kw) => normalizedInput.includes(kw))
      const fuzzyHit = isQuestion && containsKeyword

      if (exactHit || fuzzyHit) {
        appendMessage({
          id: uuid(),
          role: 'assistant',
          content: '请直接回答上面的追问内容（不要把问题句子原样发送）。例如：24岁/男/6分/没有发烧。',
          createdAt: Date.now(),
          isError: true,
        })
        return
      }
    }

    appendMessage({ id: uuid(), role: 'user', content: t, createdAt: Date.now() })
    setInput('')
    setIsSending(true)

    const ac = new AbortController()
    inFlightRef.current = ac

    try {
      const payload = {
        ...(sessionId ? { session_id: sessionId } : {}),
        user_message: t,
        top_k: Number.isFinite(topK) ? topK : 5,
        top_n: Number.isFinite(topN) ? topN : 30,
        use_rerank: !!useRerank,
      }

      const data = await chatV2(payload, { signal: ac.signal })
      updateSessionId(data.session_id)

      const nextQuestions = normalizeNonEmptyQuestions(data.next_questions)
      const askText = String((data as any).ask_text ?? '').trim()
      const questions = normalizeQuestions((data as any).questions)

      const askLines = questions.length ? questions.map((q) => q.question) : nextQuestions

      const content =
        data.mode === 'ask'
          ? [askText, ...askLines.map((q) => `- ${q}`)]
              .map((s) => String(s ?? '').trim())
              .filter(Boolean)
              .join('\n')
          : String(data.answer ?? '').trim() || String(data.summary ?? '').trim() || ''

      const msg: UiMessage = {
        id: uuid(),
        role: 'assistant',
        content,
        createdAt: Date.now(),
        mode: data.mode,
        askText,
        questions,
        nextQuestions,
        citations: Array.isArray(data.citations) ? data.citations : [],
        slots: data.slots ?? {},
        summary: data.summary ?? '',
        trace: data.trace ?? {},
        askSig: data.mode === 'ask' ? buildAskSignature(askText, questions, nextQuestions) : undefined,
      }

      if (data.mode === 'ask') upsertAssistantAsk(msg)
      else appendMessage(msg)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      appendMessage({
        id: uuid(),
        role: 'assistant',
        content: `请求失败：${msg}`,
        createdAt: Date.now(),
        isError: true,
      })
    } finally {
      setIsSending(false)
      inFlightRef.current = null
    }
  }

  function onSendFollowUpAnswer(answerText: string) {
    void sendText(answerText)
  }

  return (
    <div className="page">
      <div className="container">
        <header className="topbar">
          <div className="title">healthcare agent</div>

          <div className="session">
            <span className="label">session_id</span>
            <span className="sessionTag">{sessionId ? '已绑定' : '未绑定（首次请求自动生成）'}</span>
            <span className="sessionId">{sessionId || '-'}</span>
          </div>

          <label className="debugSwitch" title="开启后会展示 trace 摘要（已脱敏）">
            <input
              type="checkbox"
              checked={debugEnabled}
              onChange={(e) => setDebugEnabled(e.target.checked)}
              disabled={isSending}
            />
            <span>调试</span>
          </label>

          <button className="exportBtn" onClick={clearSession} disabled={isSending && messages.length === 0}>
            新建会话
          </button>
        </header>

        <div className="layout">
          <aside className="sessions">
            <div className="sessionsHeader">
              <div className="sessionsTitle">设置</div>
              <button className="newBtn" onClick={() => setShowAdvanced((v) => !v)}>
                {showAdvanced ? '收起' : '高级'}
              </button>
            </div>

            <div className="sessionsList">
              <div className="card">
                <div className="cardTitle">快速操作</div>
                <button className="newBtn" onClick={clearSession} disabled={isSending}>
                  清空会话
                </button>
                <div className="placeholder" style={{ marginTop: 8 }}>
                  清空会话会移除本地 session_id，并清空当前消息列表。
                </div>
              </div>

              {showAdvanced ? (
                <div className="card">
                  <div className="cardTitle">高级设置</div>

                  <div className="advRow">
                    <label className="advLabel">
                      top_k
                      <input
                        className="advInput"
                        type="number"
                        min={1}
                        max={50}
                        value={topK}
                        onChange={(e) => setTopK(Number(e.target.value))}
                        disabled={isSending}
                      />
                    </label>
                  </div>

                  <div className="advRow">
                    <label className="advLabel">
                      top_n
                      <input
                        className="advInput"
                        type="number"
                        min={1}
                        max={200}
                        value={topN}
                        onChange={(e) => setTopN(Number(e.target.value))}
                        disabled={isSending}
                      />
                    </label>
                  </div>

                  <div className="advRow">
                    <label className="advCheck">
                      <input
                        type="checkbox"
                        checked={useRerank}
                        onChange={(e) => setUseRerank(e.target.checked)}
                        disabled={isSending}
                      />
                      <span>use_rerank</span>
                    </label>
                  </div>

                  <div className="placeholder">默认：top_k=5, top_n=30, use_rerank=true</div>
                </div>
              ) : null}
            </div>
          </aside>

          <main className="chat">
            <div className="messages">
              {messages.length === 0 ? <div className="empty">请输入你的症状描述开始对话。</div> : null}

              {messages.map((m) => (
                <div key={m.id} className={`msg ${m.role === 'user' ? 'user' : 'assistant'}`}>
                  <div
                    className={
                      m.role === 'assistant' && m.mode === 'escalate'
                        ? 'bubble bubbleEscalate'
                        : m.isError
                          ? 'bubble bubbleError'
                          : 'bubble'
                    }
                  >
                    {m.role === 'assistant' ? renderWithEvidenceMarkers(m.content) : m.content}
                  </div>

                  {m.role === 'assistant' ? (
                    <>
                      {m.mode === 'escalate' ? <div className="escalateHint">就医建议（内容仅来自后端返回）</div> : null}

                      {m.mode === 'ask' ? (
                        <FollowUpPanel
                          askText={m.askText}
                          questions={m.questions ?? []}
                          nextQuestions={m.nextQuestions ?? []}
                          disabled={isSending}
                          onSendAnswer={onSendFollowUpAnswer}
                          onPrefill={prefillInput}
                        />
                      ) : null}

                      {m.mode === 'answer' ? <EvidencePanel citations={m.citations ?? []} /> : null}

                      <DebugTrace enabled={debugEnabled} trace={m.trace ?? null} slots={m.slots ?? null} />
                    </>
                  ) : null}
                </div>
              ))}

              <div ref={bottomRef} />
            </div>

            <div className="composer">
              <textarea
                className="input"
                placeholder="描述症状 / 提问..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                ref={inputRef}
                translate="no"
                spellCheck={false}
                autoCorrect="off"
                autoCapitalize="off"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    void sendText(input)
                  }
                }}
                rows={3}
                disabled={isSending}
              />
              <button className="sendBtn" onClick={() => void sendText(input)} disabled={isSending || !input.trim()}>
                {isSending ? '发送中…' : '发送'}
              </button>
            </div>

            {latestAssistant?.mode === 'ask' && ((latestAssistant?.questions?.length ?? 0) > 0 || (latestAssistant?.nextQuestions?.length ?? 0) > 0) ? (
              <div className="hintBar">提示：点击追问面板的选项会发送“答案”，不会发送问题文本。</div>
            ) : null}
          </main>

          <aside className="side">
            <div className="sideInner">
              <section className="card">
                <div className="cardTitle">状态</div>
                {latestAssistant ? (
                  <div className="illness">
                    <div className="kv">
                      <span className="k">mode</span>
                      <span className="v">{latestAssistant.mode ?? '-'}</span>
                    </div>
                    <div className="kv">
                      <span className="k">summary</span>
                      <span className="v">{String(latestAssistant.summary ?? '-')}</span>
                    </div>
                    <div className="kv">
                      <span className="k">citations</span>
                      <span className="v">{Array.isArray(latestAssistant.citations) ? latestAssistant.citations.length : 0}</span>
                    </div>
                    <div className="placeholder" style={{ marginTop: 8 }}>
                      右侧仅做摘要展示；详细证据请看每条回答下方“引用证据”。
                    </div>
                  </div>
                ) : (
                  <div className="placeholder">暂无内容：先发送一句话开始对话。</div>
                )}
              </section>

              <section className="card">
                <div className="cardTitle">调试说明</div>
                <div className="placeholder">
                  调试开关开启后，会在每条 assistant 消息下展示 trace 摘要（device/hits/latency 等）。
                </div>
              </section>
            </div>
          </aside>
        </div>
      </div>
    </div>
  )
}
