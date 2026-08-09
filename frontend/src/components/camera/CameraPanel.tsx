import { Video, Compass } from 'lucide-react'
import { PanelHeader } from '../dashboard/PanelHeader'
import { ConfidenceBar } from '../dashboard/ConfidenceBar'
import { CameraFeedPlaceholder } from './CameraFeedPlaceholder'
import { CameraIndicator } from './CameraIndicator'
import { BoundingBoxOverlay } from './BoundingBoxOverlay'
import { DemoStartOverlay } from './DemoStartOverlay'
import { DemoVideoControls } from './DemoVideoControls'
import { DevCameraReadout } from './DevCameraReadout'
import { useDemoVideoPlayer, DEMO_VIDEO_URL } from '../../hooks/useDemoVideoPlayer'
import { formatVehicleLabel } from '../../utils/formatters'
import type { DetectionState } from '../../types'

interface CameraPanelProps {
  detection: DetectionState
  /** Mock mode has no backend to stream from — show the decorative placeholder instead. */
  isMock: boolean
  /** Raw (pre-normalization) WebSocket payload, for DevCameraReadout only. */
  rawMessage: unknown
}

export function CameraPanel({ detection, isMock, rawMessage }: CameraPanelProps) {
  const { cameraDetected, vehicleConfidence, vehicleType, boundingBox, direction } = detection

  const { videoRef, hasStarted, isPlaying, isMuted, volume, start, togglePlay, restart, toggleMute, setVolume } =
    useDemoVideoPlayer()

  const showDemoVideo = !isMock

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
        {showDemoVideo ? (
          <>
            <video
              ref={videoRef}
              src={DEMO_VIDEO_URL}
              loop
              playsInline
              // object-fill (not object-cover): the bounding-box overlay's
              // percentages are computed against the *full* source frame, so
              // any cropping here would desync the box from what's on screen —
              // filling stretches instead of cropping, keeping the mapping 1:1.
              className="absolute inset-0 h-full w-full object-fill"
            />
            <DemoStartOverlay visible={!hasStarted} onStart={start} />
          </>
        ) : (
          <CameraFeedPlaceholder />
        )}
        <CameraIndicator />

        {/* Camera-only detection status — independent of the fused alert in
            EmergencyPanel, so this panel is verifiable on its own (req 6). */}
        <div
          className={`absolute right-4 top-4 flex items-center gap-1.5 rounded-md px-2.5 py-1 backdrop-blur ${
            cameraDetected ? 'bg-status-red/20 text-status-red' : 'bg-cockpit-950/70 text-cockpit-300'
          }`}
        >
          <span className={`h-2 w-2 rounded-full ${cameraDetected ? 'bg-status-red' : 'bg-status-green'}`} />
          <span className="font-mono text-[11px] font-medium tracking-wide">
            {cameraDetected ? 'VISUAL: DETECTED' : 'VISUAL: CLEAR'}
          </span>
        </div>

        <DevCameraReadout rawMessage={rawMessage} />

        <BoundingBoxOverlay
          box={boundingBox}
          vehicleType={vehicleType}
          confidence={vehicleConfidence}
          cameraDetected={cameraDetected}
        />

        <div className="absolute inset-x-0 bottom-0 flex flex-col gap-2.5 bg-gradient-to-t from-cockpit-950/90 to-transparent p-4">
          {showDemoVideo && (
            <DemoVideoControls
              isPlaying={isPlaying}
              isMuted={isMuted}
              volume={volume}
              onTogglePlay={togglePlay}
              onRestart={restart}
              onToggleMute={toggleMute}
              onVolumeChange={setVolume}
            />
          )}
          <div className="flex items-end justify-between gap-4">
            <div className="leading-tight">
              <p className="text-[11px] uppercase tracking-[0.2em] text-cockpit-300">Vehicle Label</p>
              <p className="font-display text-xl font-semibold text-cockpit-100">
                {formatVehicleLabel(vehicleType, cameraDetected)}
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
      </div>
    </section>
  )
}
