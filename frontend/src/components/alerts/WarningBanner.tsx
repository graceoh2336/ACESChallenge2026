import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'
import { formatVehicleLabel } from '../../utils/formatters'
import type { Direction, VehicleType } from '../../types'

interface WarningBannerProps {
  show: boolean
  vehicleType: VehicleType
  direction: Direction
}

export function WarningBanner({ show, vehicleType, direction }: WarningBannerProps) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, height: 0, marginBottom: 0 }}
          animate={{ opacity: 1, height: 'auto', marginBottom: 12 }}
          exit={{ opacity: 0, height: 0, marginBottom: 0 }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
          className="overflow-hidden"
        >
          <motion.div
            className="relative flex items-center gap-3 overflow-hidden rounded-xl border border-status-red/50 bg-status-red/15 px-4 py-3"
            animate={{
              boxShadow: [
                '0 0 0px rgba(255,59,78,0.0)',
                '0 0 24px rgba(255,59,78,0.45)',
                '0 0 0px rgba(255,59,78,0.0)',
              ],
            }}
            transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
          >
            <motion.span
              animate={{ scale: [1, 1.15, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-status-red/25"
            >
              <AlertTriangle className="h-5 w-5 text-status-red" />
            </motion.span>
            <div className="leading-tight">
              <p className="font-display text-sm font-bold uppercase tracking-[0.14em] text-status-red text-glow-red">
                Emergency Vehicle Confirmed
              </p>
              <p className="text-xs text-cockpit-200">
                {/* Only ever shown when isConfirmed (both sensors agree), so
                    a detection is guaranteed to be active here. */}
                {formatVehicleLabel(vehicleType, true)} approaching from{' '}
                <span className="font-semibold">{direction}</span> — yield right of way
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
