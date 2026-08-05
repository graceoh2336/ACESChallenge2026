import { Video, Compass } from 'lucide-react'
import { PanelHeader } from '../dashboard/PanelHeader'
import { ConfidenceBar } from '../dashboard/ConfidenceBar'
import { CameraFeedPlaceholder } from './CameraFeedPlaceholder'
import { CameraIndicator } from './CameraIndicator'
import { BoundingBoxOverlay } from './BoundingBoxOverlay'
import type { DetectionState } from '../../types'

interface CameraPanelProps {
  detection: DetectionState
}

export function CameraPanel({ detection }: CameraPanelProps) {
  const { cameraDetected, vehicleConfidence, vehicleType, boundingBox, direction } = detection

  return (
    <section className="glass-panel flex h-full flex-col overflow-hidden rounded-2xl">
      <PanelHeader
        icon={<Video className="h-4 w-4" />}
        title="Live Camera Feed"
        subtitle="Front-Facing Vision System · OpenCV"
        action={
          <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-cockpit-400">
            <Compass className="h-3.5 w-3.5" />
            {direction}
          </div>
        }
      />

      <div className="relative min-h-[280px] flex-1">
        <CameraFeedPlaceholder />
        <CameraIndicator />
        <BoundingBoxOverlay box={boundingBox} vehicleType={vehicleType} confidence={vehicleConfidence} />

        <div className="absolute inset-x-0 bottom-0 flex items-end justify-between gap-4 bg-gradient-to-t from-cockpit-950/90 to-transparent p-4">
          <div className="leading-tight">
            <p className="text-[11px] uppercase tracking-[0.2em] text-cockpit-300">Vehicle Label</p>
            <p className="font-display text-xl font-semibold text-cockpit-100">
              {cameraDetected ? vehicleType : 'No Vehicle Detected'}
            </p>
          </div>
          <div className="w-40">
            <ConfidenceBar
              label="AI Confidence"
              value={vehicleConfidence}
              tone={vehicleConfidence > 0.75 ? 'red' : vehicleConfidence > 0.4 ? 'amber' : 'green'}
            />
          </div>
        </div>
      </div>
    </section>
  )
}
