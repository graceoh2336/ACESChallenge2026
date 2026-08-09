import { Play } from 'lucide-react'

interface DemoStartOverlayProps {
  visible: boolean
  onStart: () => void
}

/**
 * Covers the video until the first user gesture — browsers block autoplay
 * with sound, so playback (and the backend demo-session reset) only ever
 * starts from this button's onClick, never automatically.
 */
export function DemoStartOverlay({ visible, onStart }: DemoStartOverlayProps) {
  if (!visible) return null

  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-cockpit-950/80 backdrop-blur-sm">
      <button
        type="button"
        onClick={onStart}
        className="flex items-center gap-2.5 rounded-xl border border-status-green/40 bg-status-green/15 px-6 py-3 font-display text-sm font-semibold uppercase tracking-[0.16em] text-status-green shadow-[0_0_24px_rgba(34,227,150,0.25)] transition hover:bg-status-green/25"
      >
        <Play className="h-4 w-4" />
        Start Demo
      </button>
      <p className="max-w-[240px] text-center text-[11px] text-cockpit-400">
        Plays the demo clip with audio and resets vision + audio detection to the same starting point.
      </p>
    </div>
  )
}
