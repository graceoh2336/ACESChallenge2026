import { AnimatePresence, motion } from 'framer-motion'
import type { BoundingBox, VehicleType } from '../../types'
import { formatConfidence, formatVehicleLabel } from '../../utils/formatters'

interface BoundingBoxOverlayProps {
  box: BoundingBox | null
  vehicleType: VehicleType
  confidence: number
  cameraDetected: boolean
}

export function BoundingBoxOverlay({ box, vehicleType, confidence, cameraDetected }: BoundingBoxOverlayProps) {
  return (
    <AnimatePresence>
      {box && (
        <motion.div
          key={`${vehicleType}-${box.x}-${box.y}`}
          className="absolute rounded-md border-2 border-status-red shadow-[0_0_18px_rgba(255,59,78,0.45)]"
          style={{
            left: `${box.x}%`,
            top: `${box.y}%`,
            width: `${box.width}%`,
            height: `${box.height}%`,
          }}
          initial={{ opacity: 0, scale: 1.08 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
        >
          <span className="absolute -left-0.5 -top-0.5 h-3 w-3 border-l-2 border-t-2 border-status-red" />
          <span className="absolute -right-0.5 -top-0.5 h-3 w-3 border-r-2 border-t-2 border-status-red" />
          <span className="absolute -bottom-0.5 -left-0.5 h-3 w-3 border-b-2 border-l-2 border-status-red" />
          <span className="absolute -bottom-0.5 -right-0.5 h-3 w-3 border-b-2 border-r-2 border-status-red" />

          <div className="absolute -top-7 left-0 flex items-center gap-1.5 rounded-md bg-status-red px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-cockpit-950 whitespace-nowrap">
            {formatVehicleLabel(vehicleType, cameraDetected)}
            <span className="font-mono opacity-80">{formatConfidence(confidence)}</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
