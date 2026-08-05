import { motion } from 'framer-motion'
import { Siren, Wifi } from 'lucide-react'
import { useClock } from '../../hooks/useClock'
import { formatClock, formatDate } from '../../utils/formatters'

export function TopStatusBar() {
  const now = useClock()

  return (
    <header className="glass-panel flex items-center justify-between gap-4 rounded-2xl px-5 py-3">
      <div className="flex items-center gap-3">
        <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-status-red/20 to-status-blue/10 ring-1 ring-cockpit-400/30">
          <Siren className="h-5 w-5 text-status-red" strokeWidth={2} />
          <span className="absolute -inset-0.5 -z-10 rounded-xl bg-status-red/20 blur-md" />
        </div>
        <div className="leading-tight">
          <p className="font-display text-lg font-semibold tracking-[0.14em] text-cockpit-100">
            SIREN<span className="text-status-red">SEEKERS</span>
          </p>
          <p className="text-[11px] uppercase tracking-[0.28em] text-cockpit-300">
            AI Emergency Vehicle Detection
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden flex-col items-end leading-tight sm:flex">
          <span className="font-mono text-lg font-medium tabular-nums text-cockpit-100">
            {formatClock(now)}
          </span>
          <span className="text-[11px] uppercase tracking-[0.2em] text-cockpit-400">
            {formatDate(now)}
          </span>
        </div>

        <div className="h-8 w-px bg-cockpit-600" />

        <div className="flex items-center gap-2 rounded-full border border-status-green/30 bg-status-green/10 px-3 py-1.5">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-status-green opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-status-green" />
          </span>
          <span className="text-xs font-medium uppercase tracking-wider text-status-green">
            System Online
          </span>
        </div>

        <motion.div
          className="flex items-center gap-2 rounded-full border border-status-blue/30 bg-status-blue/10 px-3 py-1.5"
          animate={{ opacity: [0.75, 1, 0.75] }}
          transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
        >
          <Wifi className="h-3.5 w-3.5 text-status-blue" />
          <span className="text-xs font-medium uppercase tracking-wider text-status-blue">
            AI Active
          </span>
        </motion.div>
      </div>
    </header>
  )
}
