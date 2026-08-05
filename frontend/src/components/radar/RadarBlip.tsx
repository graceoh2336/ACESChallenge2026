import { motion } from 'framer-motion'

interface RadarBlipProps {
  angle: number
  confidence: number
}

export function RadarBlip({ angle, confidence }: RadarBlipProps) {
  const radius = 42 - confidence * 27
  const rad = (angle * Math.PI) / 180
  const x = 50 + radius * Math.sin(rad)
  const y = 50 - radius * Math.cos(rad)
  const tone = confidence > 0.75 ? 'var(--color-status-red)' : 'var(--color-status-amber)'

  return (
    <div
      className="absolute -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${x}%`, top: `${y}%` }}
    >
      <motion.span
        className="absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{ backgroundColor: tone }}
        animate={{ scale: [0.6, 1.8], opacity: [0.55, 0] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: 'easeOut' }}
      />
      <span
        className="relative block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: tone, boxShadow: `0 0 10px ${tone}` }}
      />
    </div>
  )
}
