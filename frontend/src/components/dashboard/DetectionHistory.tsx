import { Area, AreaChart, ResponsiveContainer, Tooltip, YAxis } from 'recharts'
import { History, Ambulance, Truck, ShieldQuestion } from 'lucide-react'
import { PanelHeader } from './PanelHeader'
import { alertLevelStyle, formatConfidence, formatHistoryTimestamp } from '../../utils/formatters'
import type { DetectionHistoryEntry, VehicleType } from '../../types'

interface DetectionHistoryProps {
  entries: DetectionHistoryEntry[]
}

function vehicleIcon(type: VehicleType) {
  switch (type) {
    case 'Ambulance':
      return Ambulance
    case 'Fire Truck':
      return Truck
    case 'Police':
      return ShieldQuestion
    default:
      return ShieldQuestion
  }
}

interface TooltipPayloadItem {
  value: number
}

function ChartTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadItem[] }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-md border border-cockpit-600 bg-cockpit-900/95 px-2 py-1 text-[11px] font-mono text-cockpit-100">
      {formatConfidence(payload[0].value)}
    </div>
  )
}

export function DetectionHistory({ entries }: DetectionHistoryProps) {
  const chartData = [...entries]
    .slice(0, 10)
    .reverse()
    .map((entry) => ({ confidence: entry.confidence, id: entry.id }))

  return (
    <section className="glass-panel flex h-full flex-col rounded-2xl">
      <PanelHeader icon={<History className="h-4 w-4" />} title="Detection History" subtitle="Recent Activity Log" />

      <div className="h-16 px-2 pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 4, right: 6, bottom: 0, left: 6 }}>
            <defs>
              <linearGradient id="confidenceFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-status-green)" stopOpacity={0.5} />
                <stop offset="100%" stopColor="var(--color-status-green)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <YAxis domain={[0, 1]} hide />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'var(--color-cockpit-500)', strokeWidth: 1 }} />
            <Area
              type="monotone"
              dataKey="confidence"
              stroke="var(--color-status-green)"
              strokeWidth={2}
              fill="url(#confidenceFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <ul className="cockpit-scroll flex-1 divide-y divide-cockpit-600/50 overflow-y-auto px-4">
        {entries.map((entry) => {
          const style = alertLevelStyle(entry.alertLevel)
          const Icon = vehicleIcon(entry.vehicleType)
          return (
            <li key={entry.id} className="flex items-center justify-between gap-3 py-2.5">
              <div className="flex items-center gap-2.5">
                <span className={`flex h-7 w-7 items-center justify-center rounded-md ${style.bgClass} ${style.textClass}`}>
                  <Icon className="h-3.5 w-3.5" />
                </span>
                <div className="leading-tight">
                  <p className="text-sm font-medium text-cockpit-100">
                    {entry.vehicleType} <span className="text-cockpit-400">· {entry.direction}</span>
                  </p>
                  <p className="font-mono text-[11px] text-cockpit-400">
                    {formatHistoryTimestamp(entry.timestamp)}
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-end gap-0.5">
                <span className={`text-[10px] font-semibold uppercase tracking-wide ${style.textClass}`}>
                  {entry.alertLevel}
                </span>
                <span className="font-mono text-xs text-cockpit-200">{formatConfidence(entry.confidence)}</span>
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
