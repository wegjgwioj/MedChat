export type AgentChatV2Mode = 'ask' | 'answer' | 'escalate'

export type AgentCitation = {
  eid: string
  score: number
  department?: string
  title?: string
  snippet?: string
  source?: string
  chunk_id?: string
  rerank_score?: number
}

export type AgentTraceTimings = Record<string, number>

export type AgentRagStats = {
  collection?: string
  count?: number
  device?: string
  hits?: number
  latency_ms?: number
}

export type AgentTrace = {
  node_order?: string[]
  timings_ms?: AgentTraceTimings
  // 兼容旧字段（历史版本）
  timings?: AgentTraceTimings
  rag_stats?: AgentRagStats
  planner_strategy?: string
  chief_complaint?: string
}

export type AgentQuestion = {
  slot: string
  question: string
  type?: 'text' | 'enum' | 'number'
  placeholder?: string
  choices?: string[]
  range?: [number, number]
}

export type AgentChatV2Request = {
  session_id?: string
  user_message: string
  top_k?: number
  top_n?: number
  use_rerank?: boolean
}

export type AgentChatV2Response = {
  session_id: string
  mode: AgentChatV2Mode
  ask_text?: string
  questions?: AgentQuestion[]
  next_questions: string[]
  answer: string
  citations: AgentCitation[]
  slots: Record<string, unknown>
  summary: string
  trace: AgentTrace
}

export type AgentChatV2StreamEventName = 'ack' | 'stage' | 'final' | 'error'

export type AgentChatV2StreamAckPayload = {
  request_id?: string
  session_id?: string
  status?: string
  request?: {
    top_k?: number
    top_n?: number
    use_rerank?: boolean
  }
  trace_id?: string
}

export type AgentChatV2StreamStagePayload = {
  phase: string
  info?: string
  request_id?: string
  trace_id?: string
}

export type AgentChatV2StreamErrorPayload = {
  message: string
  code?: string
  request_id?: string
  trace_id?: string
}

export type AgentChatV2StreamEvent =
  | { event: 'ack'; data: AgentChatV2StreamAckPayload }
  | { event: 'stage'; data: AgentChatV2StreamStagePayload }
  | { event: 'final'; data: AgentChatV2Response }
  | { event: 'error'; data: AgentChatV2StreamErrorPayload }

export type AgentChatV2StreamHandlers = {
  onAck?: (payload: AgentChatV2StreamAckPayload) => void
  onStage?: (payload: AgentChatV2StreamStagePayload) => void
  onFinal?: (payload: AgentChatV2Response) => void
  onError?: (payload: AgentChatV2StreamErrorPayload) => void
}

export type OcrIngestRequest = {
  session_id?: string
  file_url?: string
  file?: File
}

export type OcrIngestResponse = {
  session_id: string
  task_id: string
  status: string
  trace_id?: string
  source_url?: string
  source_kind?: 'url' | 'upload'
}

export type OcrStatusResponse = {
  task_id: string
  status: string
  done: boolean
  ingested?: boolean
  session_id?: string
  trace_id?: string
  picked?: string
  message?: string
}
