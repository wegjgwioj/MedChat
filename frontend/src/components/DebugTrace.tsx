import type { AgentTrace } from '../types/agent'

export type DebugTraceProps = {
  enabled: boolean
  trace?: AgentTrace | null
  slots?: Record<string, unknown> | null
}

function safeKeys(slots: Record<string, unknown> | null | undefined) {
  if (!slots || typeof slots !== 'object') return []
  return Object.keys(slots).filter((k) => String(k).trim())
}

export default function DebugTrace(props: DebugTraceProps) {
  if (!props.enabled) return null

  const rag = props.trace?.rag_stats
  const timings = props.trace?.timings_ms ?? props.trace?.timings
  const nodeOrder = props.trace?.node_order
  const plannerStrategy = props.trace?.planner_strategy
  const chiefComplaint = props.trace?.chief_complaint
  const slotKeys = safeKeys(props.slots)

  return (
    <div className="debugPanel">
      <div className="debugTitle">调试信息（已脱敏）</div>

      <div className="debugGrid">
        <div className="debugItem">
          <div className="debugK">device</div>
          <div className="debugV">{rag?.device ?? '-'}</div>
        </div>
        <div className="debugItem">
          <div className="debugK">collection</div>
          <div className="debugV">{rag?.collection ?? '-'}</div>
        </div>
        <div className="debugItem">
          <div className="debugK">count</div>
          <div className="debugV">{typeof rag?.count === 'number' ? rag.count : '-'}</div>
        </div>
        <div className="debugItem">
          <div className="debugK">hits</div>
          <div className="debugV">{typeof rag?.hits === 'number' ? rag.hits : '-'}</div>
        </div>
        <div className="debugItem">
          <div className="debugK">latency_ms</div>
          <div className="debugV">{typeof rag?.latency_ms === 'number' ? rag.latency_ms : '-'}</div>
        </div>
        <div className="debugItem">
          <div className="debugK">planner</div>
          <div className="debugV">{plannerStrategy ?? '-'}</div>
        </div>
      </div>

      {chiefComplaint ? (
        <div className="debugTimings">
          <div className="debugSub">结构化主诉</div>
          <div className="slotKeys">{chiefComplaint}</div>
        </div>
      ) : null}

      {Array.isArray(nodeOrder) && nodeOrder.length ? (
        <div className="debugTimings">
          <div className="debugSub">节点执行顺序</div>
          <div className="slotKeys">{nodeOrder.join(' → ')}</div>
        </div>
      ) : null}

      {timings && typeof timings === 'object' ? (
        <div className="debugTimings">
          <div className="debugSub">节点耗时概览</div>
          <div className="timingList">
            {Object.entries(timings)
              .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
              .map(([k, v]) => (
                <div key={k} className="timingRow">
                  <span className="timingK">{k}</span>
                  <span className="pill">{typeof v === 'number' ? `${v} ms` : '-'}</span>
                </div>
              ))}
          </div>
        </div>
      ) : null}

      <div className="debugSlots">
        <div className="debugSub">slots 字段（仅展示字段名）</div>
        {slotKeys.length ? <div className="slotKeys">{slotKeys.join('、')}</div> : <div className="placeholder">暂无 slots。</div>}
      </div>
    </div>
  )
}
