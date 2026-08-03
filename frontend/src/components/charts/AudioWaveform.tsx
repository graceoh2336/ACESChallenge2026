import { cn } from '../../utils/cn'

interface AudioWaveformProps {
  bars: number[]
  active: boolean
}

export function AudioWaveform({ bars, active }: AudioWaveformProps) {
  return (
    <div className="flex h-20 w-full items-center gap-[3px] rounded-lg border border-cockpit-600/60 bg-cockpit-800/40 px-3">
      {bars.map((height, i) => (
        <div
          key={i}
          className={cn(
            'flex-1 rounded-full transition-colors duration-300',
            active ? 'bg-status-green' : 'bg-cockpit-500',
          )}
          style={{ height: `${Math.max(4, height * 100)}%` }}
        />
      ))}
    </div>
  )
}
