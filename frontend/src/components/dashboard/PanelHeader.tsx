import type { ReactNode } from 'react'

interface PanelHeaderProps {
  icon: ReactNode
  title: string
  subtitle?: string
  action?: ReactNode
}

export function PanelHeader({ icon, title, subtitle, action }: PanelHeaderProps) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-cockpit-600/60 px-4 py-3">
      <div className="flex items-center gap-2.5">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-cockpit-700/70 text-cockpit-200">
          {icon}
        </span>
        <div className="leading-tight">
          <h2 className="font-display text-sm font-semibold uppercase tracking-[0.14em] text-cockpit-100">
            {title}
          </h2>
          {subtitle && <p className="text-[11px] text-cockpit-400">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  )
}
