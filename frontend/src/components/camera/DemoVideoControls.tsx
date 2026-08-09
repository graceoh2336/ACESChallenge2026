import { Pause, Play, RotateCcw, Volume2, VolumeX } from 'lucide-react'

interface DemoVideoControlsProps {
  isPlaying: boolean
  isMuted: boolean
  volume: number
  onTogglePlay: () => void
  onRestart: () => void
  onToggleMute: () => void
  onVolumeChange: (value: number) => void
}

const controlButtonClass =
  'flex h-7 w-7 items-center justify-center rounded-md text-cockpit-200 transition hover:bg-cockpit-700/70 hover:text-cockpit-100'

/**
 * Custom playback controls for the demo video — deliberately not the
 * browser's native <video controls>, so this can sit inline with the rest
 * of the cockpit-styled dashboard chrome.
 */
export function DemoVideoControls({
  isPlaying,
  isMuted,
  volume,
  onTogglePlay,
  onRestart,
  onToggleMute,
  onVolumeChange,
}: DemoVideoControlsProps) {
  return (
    <div className="flex items-center gap-1 rounded-lg border border-cockpit-600/60 bg-cockpit-950/70 px-2 py-1.5 backdrop-blur">
      <button
        type="button"
        onClick={onTogglePlay}
        aria-label={isPlaying ? 'Pause demo' : 'Play demo'}
        className={controlButtonClass}
      >
        {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
      </button>

      <button type="button" onClick={onRestart} aria-label="Restart demo" className={controlButtonClass}>
        <RotateCcw className="h-3.5 w-3.5" />
      </button>

      <button
        type="button"
        onClick={onToggleMute}
        aria-label={isMuted ? 'Unmute' : 'Mute'}
        className={controlButtonClass}
      >
        {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
      </button>

      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={isMuted ? 0 : volume}
        onChange={(event) => onVolumeChange(Number(event.target.value))}
        aria-label="Volume"
        className="ml-1 h-1 w-20 cursor-pointer accent-status-green"
      />
    </div>
  )
}
