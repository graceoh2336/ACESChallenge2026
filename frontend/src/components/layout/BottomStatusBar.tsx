import type { ComponentType } from 'react'
import { Gauge, Cpu, Signal, FlaskConical, Tag } from 'lucide-react'
import type { DriveMode } from '../../types'

interface BottomStatusBarProps {
  driveMode: DriveMode
  version: string
}

interface StatusItem {
  icon: ComponentType<{ className?: string }>
  label: string
  value: string
  valueClass?: string
}

export function BottomStatusBar({ driveMode, version }: BottomStatusBarProps) {
  const items: StatusItem[] = [
    { icon: Gauge, label: 'Drive Mode', value: driveMode },
    { icon: Cpu, label: 'AI Status', value: 'Active', valueClass: 'text-status-green' },
    { icon: Signal, label: 'Connection', value: 'Stable', valueClass: 'text-status-green' },
    { icon: FlaskConical, label: 'Simulation', value: 'Mock Data', valueClass: 'text-status-amber' },
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
