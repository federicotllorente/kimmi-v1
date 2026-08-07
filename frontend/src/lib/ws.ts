import type { StatusEvent } from './types'

const DEFAULT_WS_URL = 'ws://localhost:8000/ws/status'

export function getWsUrl(): string {
  return import.meta.env.VITE_WS_URL?.trim() || DEFAULT_WS_URL
}

type StatusListener = (event: StatusEvent) => void
type ConnListener = (connected: boolean) => void

function isStatusEvent(data: unknown): data is StatusEvent {
  if (!data || typeof data !== 'object') return false
  const obj = data as Record<string, unknown>
  return (
    typeof obj.state === 'string' &&
    typeof obj.amplitude === 'number' &&
    typeof obj.message === 'string'
  )
}

/** WebSocket client with automatic reconnect for pipeline status events. */
export class StatusSocket {
  private ws: WebSocket | null = null
  private closedByUser = false
  private retryMs = 1000
  private readonly maxRetryMs = 8000
  private readonly onStatus: StatusListener
  private readonly onConn: ConnListener
  private readonly url: string

  constructor(onStatus: StatusListener, onConn: ConnListener, url = getWsUrl()) {
    this.onStatus = onStatus
    this.onConn = onConn
    this.url = url
  }

  connect(): void {
    this.closedByUser = false
    this.open()
  }

  disconnect(): void {
    this.closedByUser = true
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.onConn(false)
  }

  private open(): void {
    if (this.closedByUser) return

    const ws = new WebSocket(this.url)
    this.ws = ws

    ws.onopen = () => {
      this.retryMs = 1000
      this.onConn(true)
      // Nudge server keepalive loop (optional).
      try {
        ws.send('ping')
      } catch {
        /* ignore */
      }
    }

    ws.onmessage = (ev) => {
      try {
        const data: unknown = JSON.parse(String(ev.data))
        if (isStatusEvent(data)) {
          this.onStatus(data)
        }
      } catch {
        /* ignore malformed */
      }
    }

    ws.onclose = () => {
      this.onConn(false)
      this.ws = null
      if (!this.closedByUser) {
        window.setTimeout(() => this.open(), this.retryMs)
        this.retryMs = Math.min(this.retryMs * 1.5, this.maxRetryMs)
      }
    }

    ws.onerror = () => {
      ws.close()
    }
  }
}
