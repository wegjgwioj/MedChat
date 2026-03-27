import type { AgentChatV2Request, AgentChatV2Response, OcrIngestRequest, OcrIngestResponse, OcrStatusResponse } from '../types/agent'

function normalizeBaseUrl(raw: string | undefined): string {
  const v = (raw ?? '').trim()
  if (!v) return 'http://127.0.0.1:8000'
  return v.endsWith('/') ? v.slice(0, -1) : v
}

export function getApiBaseUrl(): string {
  return normalizeBaseUrl((import.meta as any).env?.VITE_API_BASE_URL as string | undefined)
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
