import { motion } from 'framer-motion'

export function CameraIndicator() {
  return (
    <div className="absolute left-4 top-4 flex items-center gap-1.5 rounded-md bg-cockpit-950/70 px-2.5 py-1 backdrop-blur">
      <motion.span
        className="h-2 w-2 rounded-full bg-status-red"
        animate={{ opacity: [1, 0.3, 1] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
      />
      <span className="font-mono text-[11px] font-medium tracking-wide text-cockpit-100">
        REC · CAM-01 FRONT
      </span>
    </div>
  )
}
