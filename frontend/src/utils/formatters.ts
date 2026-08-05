import type { AlertLevel, ConnectionStatus, HealthStatus } from '../types'

export function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`
}

export function formatClock(date: Date): string {
  return date.toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

export function formatDate(date: Date): string {
  return date.toLocaleDateString('en-GB', {
    weekday: 'short',
    day: '2-digit',
    month: 'short',
  })
}

export function formatHistoryTimestamp(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

interface AlertLevelStyle {
  label: string
  textClass: string
  bgClass: string
  borderClass: string
  dotClass: string
}

export function alertLevelStyle(level: AlertLevel): AlertLevelStyle {
  switch (level) {
    case 'Critical':
    case 'Confirmed':
      return {
        label: level === 'Critical' ? 'Critical' : 'Emergency Vehicle Confirmed',
        textClass: 'text-status-red',
        bgClass: 'bg-status-red/10',
        borderClass: 'border-status-red/40',
        dotClass: 'bg-status-red',
      }
    case 'Possible':
      return {
        label: 'Possible Emergency Vehicle',
        textClass: 'text-status-amber',
        bgClass: 'bg-status-amber/10',
        borderClass: 'border-status-amber/40',
        dotClass: 'bg-status-amber',
      }
    default:
      return {
        label: 'Monitoring',
        textClass: 'text-status-green',
        bgClass: 'bg-status-green/10',
        borderClass: 'border-status-green/40',
        dotClass: 'bg-status-green',
      }
  }
}

interface HealthStyle {
  label: string
  textClass: string
  dotClass: string
}

interface ConnectionStatusStyle {
  label: string
  textClass: string
  dotClass: string
}

export function connectionStatusStyle(status: ConnectionStatus): ConnectionStatusStyle {
  switch (status) {
    case 'connected':
      return { label: 'Connected', textClass: 'text-status-green', dotClass: 'bg-status-green' }
    case 'connecting':
      return { label: 'Connecting…', textClass: 'text-status-amber', dotClass: 'bg-status-amber' }
    case 'reconnecting':
      return { label: 'Reconnecting…', textClass: 'text-status-amber', dotClass: 'bg-status-amber' }
    case 'error':
      return { label: 'Connection Error', textClass: 'text-status-red', dotClass: 'bg-status-red' }
    default:
      return { label: 'Disconnected', textClass: 'text-status-red', dotClass: 'bg-status-red' }
  }
}

export function healthStyle(status: HealthStatus): HealthStyle {
  switch (status) {
    case 'online':
      return { label: 'Online', textClass: 'text-status-green', dotClass: 'bg-status-green' }
    case 'degraded':
      return { label: 'Degraded', textClass: 'text-status-amber', dotClass: 'bg-status-amber' }
    default:
      return { label: 'Offline', textClass: 'text-status-red', dotClass: 'bg-status-red' }
  }
}
