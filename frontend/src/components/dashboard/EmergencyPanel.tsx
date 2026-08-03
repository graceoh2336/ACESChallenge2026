import { ShieldHalf, Compass, Truck } from 'lucide-react'
import { PanelHeader } from './PanelHeader'
import { ConfidenceBar } from './ConfidenceBar'
import { EmergencyStateSteps } from './EmergencyStateSteps'
import { WarningBanner } from '../alerts/WarningBanner'
import { alertLevelStyle } from '../../utils/formatters'
import type { DetectionState } from '../../types'

interface EmergencyPanelProps {
  detection: DetectionState
}

export function EmergencyPanel({ detection }: EmergencyPanelProps) {
  const { alertLevel, vehicleType, direction, vehicleConfidence, audioConfidence } = detection
  const style = alertLevelStyle(alertLevel)
  const overallConfidence = Math.max(vehicleConfidence, audioConfidence)
  const isConfirmed = alertLevel === 'Confirmed' || alertLevel === 'Critical'

  return (
    <section className="glass-panel flex h-full flex-col rounded-2xl">
      <PanelHeader
        icon={<ShieldHalf className="h-4 w-4" />}
        title="Emergency Detection"
        subtitle="Sensor Fusion · Audio + Vision"
        action={
          <span
            className={`rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide ${style.borderClass} ${style.bgClass} ${style.textClass}`}
          >
            {alertLevel}
          </span>
        }
      />

      <div className="flex flex-1 flex-col gap-4 p-4">
        <EmergencyStateSteps alertLevel={alertLevel} />

        <WarningBanner show={isConfirmed} vehicleType={vehicleType} direction={direction} />

        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-cockpit-600/60 bg-cockpit-800/40 p-3">
            <p className="mb-1 flex items-center gap-1.5 text-[11px] uppercase tracking-[0.16em] text-cockpit-400">
              <Truck className="h-3.5 w-3.5" /> Vehicle Type
            </p>
            <p className="font-display text-lg font-semibold text-cockpit-100">{vehicleType}</p>
          </div>
          <div className="rounded-lg border border-cockpit-600/60 bg-cockpit-800/40 p-3">
            <p className="mb-1 flex items-center gap-1.5 text-[11px] uppercase tracking-[0.16em] text-cockpit-400">
              <Compass className="h-3.5 w-3.5" /> Direction
            </p>
            <p className="font-display text-lg font-semibold text-cockpit-100">{direction}</p>
          </div>
        </div>

        <div className="mt-auto rounded-lg border border-cockpit-600/60 bg-cockpit-800/40 p-3">
          <ConfidenceBar
            label="Combined Confidence"
            value={overallConfidence}
            tone={overallConfidence > 0.75 ? 'red' : overallConfidence > 0.4 ? 'amber' : 'green'}
          />
        </div>
      </div>
    </section>
  )
}
