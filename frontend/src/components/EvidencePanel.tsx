import { useMemo, useState } from 'react'
import type { AgentCitation } from '../types/agent'

export type EvidencePanelProps = {
  citations: AgentCitation[]
}

function normalizeText(raw: string | undefined) {
  return String(raw ?? '')
    .replace(/^ï»¿/, '')
    .replace(/\uFEFF/g, '')
    .replace(/\u0000/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function shortSnippet(raw: string, maxLen: number) {
  const t = normalizeText(raw)
  if (t.length <= maxLen) return { text: t, isShort: true }
  return { text: `${t.slice(0, maxLen)}…`, isShort: false }
}

export default function EvidencePanel(props: EvidencePanelProps) {
  const citations = useMemo(() => {
    return Array.isArray(props.citations) ? props.citations.filter((c) => c && String(c.eid ?? '').trim()) : []
  }, [props.citations])

  // citations 为空：直接给出提示文案，避免用户必须点开才能看到
  if (!citations.length) {
    return <div className="evidenceEmptyInline">未检索到可靠资料/本次未引用知识库证据</div>
  }

  const [open, setOpen] = useState(false)

  return (
    <div className="evidencePanel">
      <button className="evidenceToggle" onClick={() => setOpen((v) => !v)}>
        <span className="evidenceToggleTitle">引用证据</span>
        <span className="pill">{citations.length}</span>
        <span className="evidenceToggleHint">{open ? '收起' : '展开'}</span>
      </button>

      {open ? (
        <div className="evidenceCards">
          {citations.map((c) => (
            <EvidenceCard key={`${c.eid}-${c.chunk_id ?? ''}`} c={c} />
          ))}
        </div>
      ) : null}
    </div>
  )
}

function EvidenceCard({ c }: { c: AgentCitation }) {
  const [expanded, setExpanded] = useState(false)

  const snippet = normalizeText(c.snippet ?? '')
  const short = shortSnippet(snippet, 120)

  return (
    <div className="evidenceCard">
      <div className="evidenceCardHead">
        <span className="pill">{c.eid}</span>
        {c.department ? <span className="pill">{c.department}</span> : null}
        {c.title ? <span className="evidenceTitle">{c.title}</span> : null}
      </div>

      <div className="evidenceCardMeta">
        <span className="metaKey">score</span>
        <span className="metaVal">{Number.isFinite(c.score) ? c.score.toFixed(4) : '-'}</span>
        {typeof c.bm25_score === 'number' ? (
          <>
            <span className="metaKey">bm25</span>
            <span className="metaVal">{c.bm25_score.toFixed(4)}</span>
          </>
        ) : null}
        {typeof c.hybrid_score === 'number' ? (
          <>
            <span className="metaKey">hybrid</span>
            <span className="metaVal">{c.hybrid_score.toFixed(4)}</span>
          </>
        ) : null}
        {typeof c.rerank_score === 'number' ? (
          <>
            <span className="metaKey">rerank</span>
            <span className="metaVal">{c.rerank_score.toFixed(4)}</span>
          </>
        ) : null}
      </div>

      {snippet ? (
        <div className="evidenceSnippet">
          <div className="evidenceSnippetText">{expanded ? snippet : short.text}</div>
          {!short.isShort ? (
            <button className="linkBtn" onClick={() => setExpanded((v) => !v)}>
              {expanded ? '收起全文' : '展开全文'}
            </button>
          ) : null}
        </div>
      ) : (
        <div className="evidenceEmpty">该证据未提供摘要片段</div>
      )}

      <div className="evidenceFoot">
        {c.source ? <span className="weak">source: {c.source}</span> : null}
        {c.chunk_id ? <span className="weak">chunk_id: {c.chunk_id}</span> : null}
      </div>
    </div>
  )
}
