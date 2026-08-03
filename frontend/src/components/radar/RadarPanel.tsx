import { motion } from 'framer-motion'
import { Radar } from 'lucide-react'
import { PanelHeader } from '../dashboard/PanelHeader'
import { RadarBlip } from './RadarBlip'
import type { DetectionState } from '../../types'

const directionAngles: Record<DetectionState['direction'], number> = {
  Front: 0,
  'Front Right': 45,
  Right: 90,
  'Rear Right': 135,
  Rear: 180,
  'Rear Left': 225,
  Left: 270,
  'Front Left': 315,
}

const rings = [18, 32, 46]

interface RadarPanelProps {
  detection: DetectionState
}

export function RadarPanel({ detection }: RadarPanelProps) {
  const { direction, audioDetected, cameraDetected, audioConfidence, vehicleConfidence } = detection
  const isActive = audioDetected || cameraDetected
  const confidence = Math.max(audioConfidence, vehicleConfidence)
  const angle = directionAngles[direction]

  return (
    <section className="glass-panel flex h-full flex-col rounded-2xl">
      <PanelHeader icon={<Radar className="h-4 w-4" />} title="Radar" subtitle="360° Proximity Map" />

      <div className="flex flex-1 items-center justify-center p-4">
        <div className="relative aspect-square w-full max-w-[220px]">
          <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
            {rings.map((r) => (
              <circle
                key={r}
                cx="50"
                cy="50"
                r={r}
                fill="none"
                stroke="var(--color-cockpit-500)"
                strokeOpacity={0.35}
                strokeWidth={0.5}
              />
            ))}
            <line x1="50" y1="4" x2="50" y2="96" stroke="var(--color-cockpit-500)" strokeOpacity={0.25} strokeWidth={0.4} />
            <line x1="4" y1="50" x2="96" y2="50" stroke="var(--color-cockpit-500)" strokeOpacity={0.25} strokeWidth={0.4} />
          </svg>

          <motion.div
            className="absolute inset-0"
            animate={{ rotate: 360 }}
            transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
            style={{
              background:
                'conic-gradient(from 0deg, rgba(34,227,150,0.28), transparent 28%)',
              borderRadius: '9999px',
              maskImage: 'radial-gradient(circle, black 96%, transparent 100%)',
              WebkitMaskImage: 'radial-gradient(circle, black 96%, transparent 100%)',
            }}
          />

          <div className="absolute left-1/2 top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cockpit-100 shadow-[0_0_8px_rgba(216,220,230,0.7)]" />

          {isActive && <RadarBlip angle={angle} confidence={confidence} />}

          <span className="absolute left-1/2 top-1 -translate-x-1/2 text-[9px] uppercase tracking-widest text-cockpit-400">
            Front
          </span>
          <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] uppercase tracking-widest text-cockpit-400">
            Rear
          </span>
          <span className="absolute left-1 top-1/2 -translate-y-1/2 text-[9px] uppercase tracking-widest text-cockpit-400">
            L
          </span>
          <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[9px] uppercase tracking-widest text-cockpit-400">
            R
          </span>
        </div>
      </div>

      <div className="border-t border-cockpit-600/60 px-4 py-2.5 text-center">
        <span className="font-mono text-xs text-cockpit-300">
          {isActive ? `Contact bearing ${direction}` : 'No contacts in range'}
        </span>
      </div>
    </section>
  )
}
