import { useEffect, useRef, useState } from 'react'
import type { ConnectionStatus, DetectionState } from '../types'
import { parseDetectionMessage } from '../utils/normalizeDetection'

// Fallback only — the real URL should come from VITE_DETECTION_WS_URL.
const DEFAULT_WS_URL = 'ws://localhost:8000/ws/detection'

const INITIAL_RECONNECT_DELAY_MS = 1000
const MAX_RECONNECT_DELAY_MS = 30000

export interface UseDetectionWebSocketResult {
  detection: DetectionState | null
  connectionStatus: ConnectionStatus
  lastMessageAt: Date | null
  error: string | null
}

/**
 * Connects to the FastAPI detection WebSocket on mount, keeps it alive with
 * exponential-backoff reconnects, and exposes the latest normalized
 * DetectionState. The last valid detection is retained across drops/retries
 * — only a new valid message ever replaces it.
 */
export function useDetectionWebSocket(url?: string): UseDetectionWebSocketResult {
  const resolvedUrl = url ?? import.meta.env.VITE_DETECTION_WS_URL ?? DEFAULT_WS_URL

  const [detection, setDetection] = useState<DetectionState | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting')
  const [lastMessageAt, setLastMessageAt] = useState<Date | null>(null)
  const [error, setError] = useState<string | null>(null)

  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const reconnectAttemptRef = useRef(0)
  // Guards async socket callbacks against acting after this effect instance
  // has been torn down (unmount, or React Strict Mode's mount/cleanup/mount).
  const activeRef = useRef(false)

  useEffect(() => {
    activeRef.current = true

    const clearReconnectTimer = () => {
      if (reconnectTimeoutRef.current !== null) {
        window.clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
    }

    const scheduleReconnect = () => {
      if (!activeRef.current) return
      clearReconnectTimer()
      const attempt = reconnectAttemptRef.current
      reconnectAttemptRef.current = attempt + 1
      const delay = Math.min(INITIAL_RECONNECT_DELAY_MS * 2 ** attempt, MAX_RECONNECT_DELAY_MS)
      setConnectionStatus('reconnecting')
      reconnectTimeoutRef.current = window.setTimeout(() => {
        reconnectTimeoutRef.current = null
        connect()
      }, delay)
    }

    const connect = () => {
      if (!activeRef.current) return
      // Never open a second socket while one is already live — protects
      // against duplicate connections from Strict Mode or overlapping timers.
      const existing = socketRef.current
      if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
        return
      }

      setConnectionStatus('connecting')

      let socket: WebSocket
      try {
        socket = new WebSocket(resolvedUrl)
      } catch {
        setError('Unable to open WebSocket connection')
        setConnectionStatus('error')
        scheduleReconnect()
        return
      }

      socketRef.current = socket

      socket.onopen = () => {
        if (socketRef.current !== socket) return
        reconnectAttemptRef.current = 0
        setError(null)
        setConnectionStatus('connected')
      }

      socket.onmessage = (event) => {
        if (socketRef.current !== socket) return
        if (typeof event.data !== 'string') return

        const normalized = parseDetectionMessage(event.data)
        if (normalized === null) {
          setError('Received malformed detection event')
          return
        }

        setDetection(normalized)
        setLastMessageAt(new Date())
        setError(null)
      }

      socket.onerror = () => {
        if (socketRef.current !== socket) return
        setError('WebSocket connection error')
        setConnectionStatus('error')
      }

      socket.onclose = () => {
        if (socketRef.current === socket) socketRef.current = null
        if (!activeRef.current) return
        setConnectionStatus('disconnected')
        scheduleReconnect()
      }
    }

    connect()

    return () => {
      activeRef.current = false
      clearReconnectTimer()

      const socket = socketRef.current
      if (socket) {
        socket.onopen = null
        socket.onmessage = null
        socket.onerror = null
        socket.onclose = null
        socket.close()
        socketRef.current = null
      }
    }
  }, [resolvedUrl])

  return { detection, connectionStatus, lastMessageAt, error }
}
