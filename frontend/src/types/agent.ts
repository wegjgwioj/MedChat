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
