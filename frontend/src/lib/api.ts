import type {
  AgentChatV2Request,
  AgentChatV2Response,
  AgentChatV2StreamAckPayload,
  AgentChatV2StreamErrorPayload,
  AgentChatV2StreamEvent,
  AgentChatV2StreamHandlers,
  AgentChatV2StreamStagePayload,
  OcrIngestRequest,
  OcrIngestResponse,
  OcrStatusResponse,
} from '../types/agent'

function normalizeBaseUrl(raw: string | undefined): string {
  const v = (raw ?? '').trim()
  if (!v) return 'http://127.0.0.1:8000'
  return v.endsWith('/') ? v.slice(0, -1) : v
}

export function getApiBaseUrl(): string {
  return normalizeBaseUrl((import.meta as any).env?.VITE_API_BASE_URL as string | undefined)
}

function parseSseEvent(raw: string): AgentChatV2StreamEvent | null {
  const lines = raw
    .replace(/\r/g, '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)

  let eventName: string | undefined
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim()
      continue
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }

  if (!eventName || dataLines.length === 0) {
    return null
  }

  const payloadJson = dataLines.join('\n')
  let parsed: unknown
  try {
    parsed = JSON.parse(payloadJson)
  } catch (error) {
    if (eventName === 'error') {
      return { event: 'error', data: { message: payloadJson } }
    }
    throw error
  }

  switch (eventName) {
    case 'ack':
      return { event: 'ack', data: parsed as AgentChatV2StreamAckPayload }
    case 'stage':
      return { event: 'stage', data: parsed as AgentChatV2StreamStagePayload }
    case 'final':
      return { event: 'final', data: parsed as AgentChatV2Response }
    case 'error':
      return { event: 'error', data: parsed as AgentChatV2StreamErrorPayload }
    default:
      return null
  }
}

function makeStreamUnavailableError(): Error {
  return new Error('STREAM_UNAVAILABLE')
}

export async function chatV2Stream(
  req: AgentChatV2Request,
  handlers?: AgentChatV2StreamHandlers,
  opts?: { signal?: AbortSignal },
): Promise<AgentChatV2Response> {
  const baseUrl = getApiBaseUrl()

  const resp = await fetch(`${baseUrl}/v1/agent/chat_v2/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(req),
    signal: opts?.signal,
  })

  if (!resp.ok) {
    throw makeStreamUnavailableError()
  }

  const contentType = resp.headers.get('content-type') ?? ''
  if (!contentType.includes('text/event-stream')) {
    throw makeStreamUnavailableError()
  }

  const reader = resp.body?.getReader()
  if (!reader) {
    throw makeStreamUnavailableError()
  }

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r/g, '')

    let delimiter = buffer.indexOf('\n\n')
    while (delimiter !== -1) {
      const chunk = buffer.slice(0, delimiter)
      buffer = buffer.slice(delimiter + 2)
      delimiter = buffer.indexOf('\n\n')

      const evt = parseSseEvent(chunk)
      if (!evt) continue

      if (evt.event === 'ack') {
        handlers?.onAck?.(evt.data)
        continue
      }
      if (evt.event === 'stage') {
        handlers?.onStage?.(evt.data)
        continue
      }
      if (evt.event === 'error') {
        handlers?.onError?.(evt.data)
        throw new Error(evt.data.message || 'SSE error')
      }
      if (evt.event === 'final') {
        handlers?.onFinal?.(evt.data)
        return evt.data
      }
    }
  }

  throw makeStreamUnavailableError()
}

export async function chatV2(req: AgentChatV2Request, opts?: { signal?: AbortSignal }): Promise<AgentChatV2Response> {
  const baseUrl = getApiBaseUrl()

  const resp = await fetch(`${baseUrl}/v1/agent/chat_v2`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(req),
    signal: opts?.signal,
  })

  if (!resp.ok) {
    const t = await resp.text().catch(() => '')
    throw new Error(t || `HTTP ${resp.status}`)
  }

  const data = (await resp.json()) as unknown
  return data as AgentChatV2Response
}

export async function ocrIngest(req: OcrIngestRequest, opts?: { signal?: AbortSignal }): Promise<OcrIngestResponse> {
  const baseUrl = getApiBaseUrl()

  let resp: Response
  if (req.file) {
    const form = new FormData()
    if (req.session_id) form.append('session_id', req.session_id)
    form.append('file', req.file)
    resp = await fetch(`${baseUrl}/v1/ocr/ingest`, {
      method: 'POST',
      body: form,
      signal: opts?.signal,
    })
  } else {
    resp = await fetch(`${baseUrl}/v1/ocr/ingest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: req.session_id,
        file_url: req.file_url,
      }),
      signal: opts?.signal,
    })
  }

  if (!resp.ok) {
    const t = await resp.text().catch(() => '')
    throw new Error(t || `HTTP ${resp.status}`)
  }

  const data = (await resp.json()) as unknown
  return data as OcrIngestResponse
}

export async function ocrStatus(taskId: string, opts?: { sessionId?: string; sourceUrl?: string; signal?: AbortSignal }): Promise<OcrStatusResponse> {
  const baseUrl = getApiBaseUrl()
  const params = new URLSearchParams()
  if (opts?.sessionId) params.set('session_id', opts.sessionId)
  if (opts?.sourceUrl) params.set('source_url', opts.sourceUrl)
  const qs = params.toString()

  const resp = await fetch(`${baseUrl}/v1/ocr/status/${encodeURIComponent(taskId)}${qs ? `?${qs}` : ''}`, {
    method: 'GET',
    signal: opts?.signal,
  })

  if (!resp.ok) {
    const t = await resp.text().catch(() => '')
    throw new Error(t || `HTTP ${resp.status}`)
  }

  const data = (await resp.json()) as unknown
  return data as OcrStatusResponse
}
