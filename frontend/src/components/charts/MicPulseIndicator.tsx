import { motion } from 'framer-motion'
import { Mic } from 'lucide-react'
import { cn } from '../../utils/cn'

interface MicPulseIndicatorProps {
  active: boolean
}

export function MicPulseIndicator({ active }: MicPulseIndicatorProps) {
  return (
    <div className="relative flex h-14 w-14 shrink-0 items-center justify-center">
      {active && (
        <>
          <motion.span
            className="absolute inset-0 rounded-full bg-status-green/30"
            animate={{ scale: [0.7, 1.6], opacity: [0.6, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeOut' }}
          />
          <motion.span
            className="absolute inset-0 rounded-full bg-status-green/30"
            animate={{ scale: [0.7, 1.6], opacity: [0.6, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeOut', delay: 0.6 }}
          />
        </>
      )}
      <span
        className={cn(
          'relative z-10 flex h-10 w-10 items-center justify-center rounded-full border',
          active ? 'border-status-green/60 bg-status-green/15 text-status-green' : 'border-cockpit-500 bg-cockpit-700/60 text-cockpit-400',
        )}
      >
        <Mic className="h-5 w-5" />
      </span>
    </div>
  )
}
