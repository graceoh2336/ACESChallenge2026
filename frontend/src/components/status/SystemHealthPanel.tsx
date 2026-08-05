import { HeartPulse } from 'lucide-react'
import { PanelHeader } from '../dashboard/PanelHeader'
import { StatusLight } from './StatusLight'
import { healthStyle, formatClock } from '../../utils/formatters'
import { systemHealthMock } from '../../data/mockData'
import type { ConnectionStatus, HealthStatus, SystemComponentHealth } from '../../types'

interface SystemHealthPanelProps {
  connectionStatus: ConnectionStatus
  isMock: boolean
  lastMessageAt: Date | null
  error: string | null
}

function backendHealthStatus(connectionStatus: ConnectionStatus): HealthStatus {
  switch (connectionStatus) {
    case 'connected':
      return 'online'
    case 'connecting':
    case 'reconnecting':
      return 'degraded'
    default:
      return 'offline'
  }
}

function backendDetail(
  connectionStatus: ConnectionStatus,
  isMock: boolean,
  lastMessageAt: Date | null,
  error: string | null,
): string {
  if (isMock) return 'Simulated · mock mode'

  switch (connectionStatus) {
    case 'connected':
      return lastMessageAt ? `Live · last event ${formatClock(lastMessageAt)}` : 'Live · awaiting first event'
    case 'connecting':
      return 'Connecting to backend…'
    case 'reconnecting':
      return error ? `Reconnecting · ${error}` : 'Reconnecting to backend…'
    default:
      return error ?? 'No connection to backend'
  }
}

export function SystemHealthPanel({ connectionStatus, isMock, lastMessageAt, error }: SystemHealthPanelProps) {
  const components: SystemComponentHealth[] = systemHealthMock.map((component) =>
    component.id === 'backend'
      ? {
          ...component,
          status: isMock ? 'degraded' : backendHealthStatus(connectionStatus),
          detail: backendDetail(connectionStatus, isMock, lastMessageAt, error),
        }
      : component,
  )

  return (
    <section className="glass-panel flex h-full flex-col rounded-2xl">
      <PanelHeader icon={<HeartPulse className="h-4 w-4" />} title="System Health" subtitle="Live Diagnostics" />

      <ul className="flex flex-1 flex-col divide-y divide-cockpit-600/50 px-4">
        {components.map((component) => {
          const style = healthStyle(component.status)
          return (
            <li key={component.id} className="flex items-center justify-between gap-3 py-2.5">
              <div className="flex items-center gap-2.5">
                <StatusLight status={component.status} />
                <div className="leading-tight">
                  <p className="text-sm font-medium text-cockpit-100">{component.name}</p>
                  <p className="text-[11px] text-cockpit-400">{component.detail}</p>
                </div>
              </div>
              <span className={`text-[11px] font-semibold uppercase tracking-wide ${style.textClass}`}>
                {style.label}
              </span>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
