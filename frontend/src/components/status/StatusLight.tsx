import { motion } from 'framer-motion'
import { cn } from '../../utils/cn'
import { healthStyle } from '../../utils/formatters'
import type { HealthStatus } from '../../types'

export function StatusLight({ status }: { status: HealthStatus }) {
  const { dotClass } = healthStyle(status)

  return (
    <span className="relative flex h-2.5 w-2.5">
      {status === 'online' && (
        <motion.span
          className={cn('absolute inline-flex h-full w-full rounded-full', dotClass)}
          animate={{ scale: [1, 2], opacity: [0.6, 0] }}
          transition={{ duration: 1.6, repeat: Infinity, ease: 'easeOut' }}
        />
      )}
      <span className={cn('relative inline-flex h-2.5 w-2.5 rounded-full', dotClass)} />
    </span>
  )
}
