import { motion } from 'framer-motion'
import { cn } from '../../utils/cn'
import { formatConfidence } from '../../utils/formatters'

interface ConfidenceBarProps {
  label: string
  value: number
  tone?: 'green' | 'amber' | 'red'
}

const toneClasses: Record<NonNullable<ConfidenceBarProps['tone']>, string> = {
  green: 'bg-status-green',
  amber: 'bg-status-amber',
  red: 'bg-status-red',
}

export function ConfidenceBar({ label, value, tone = 'green' }: ConfidenceBarProps) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-[0.16em] text-cockpit-400">{label}</span>
        <span className="font-mono text-xs font-semibold text-cockpit-100">
          {formatConfidence(value)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-cockpit-700">
        <motion.div
          className={cn('h-full rounded-full', toneClasses[tone])}
          initial={false}
          animate={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
          transition={{ type: 'spring', stiffness: 90, damping: 20 }}
        />
      </div>
    </div>
  )
}
