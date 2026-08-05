import type { ComponentType } from 'react'
import { Gauge, Cpu, Signal, FlaskConical, Radio, Tag, Clock } from 'lucide-react'
import type { ConnectionStatus, DriveMode } from '../../types'
import { connectionStatusStyle, formatClock } from '../../utils/formatters'

interface BottomStatusBarProps {
  driveMode: DriveMode
  version: string
  connectionStatus: ConnectionStatus
  isMock: boolean
  lastMessageAt: Date | null
}

interface StatusItem {
  icon: ComponentType<{ className?: string }>
  label: string
  value: string
  valueClass?: string
}

export function BottomStatusBar({ driveMode, version, connectionStatus, isMock, lastMessageAt }: BottomStatusBarProps) {
  const connection = connectionStatusStyle(connectionStatus)
  const aiActive = connectionStatus === 'connected'

  const items: StatusItem[] = [
    { icon: Gauge, label: 'Drive Mode', value: driveMode },
    {
      icon: Cpu,
      label: 'AI Status',
      value: aiActive ? 'Active' : 'Standby',
      valueClass: aiActive ? 'text-status-green' : 'text-status-amber',
    },
    { icon: Signal, label: 'Connection', value: connection.label, valueClass: connection.textClass },
    {
      icon: isMock ? FlaskConical : Radio,
      label: 'Data Source',
      value: isMock ? 'Mock Data' : 'Live WebSocket',
      valueClass: isMock ? 'text-status-amber' : 'text-status-green',
    },
    {
      icon: Clock,
      label: 'Last Update',
      value: lastMessageAt ? formatClock(lastMessageAt) : '—',
    },
    { icon: Tag, label: 'Version', value: version },
  ]

  return (
    <footer className="glass-panel flex flex-wrap items-center justify-between gap-x-6 gap-y-2 rounded-2xl px-5 py-3">
      {items.map(({ icon: Icon, label, value, valueClass }) => (
        <div key={label} className="flex items-center gap-2">
          <Icon className="h-3.5 w-3.5 text-cockpit-400" />
          <span className="text-[11px] uppercase tracking-[0.18em] text-cockpit-400">{label}</span>
          <span className={`font-mono text-xs font-medium ${valueClass ?? 'text-cockpit-100'}`}>
            {value}
          </span>
        </div>
      ))}
    </footer>
  )
}
