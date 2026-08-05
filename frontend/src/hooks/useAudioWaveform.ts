import { useEffect, useRef, useState } from 'react'

const BAR_COUNT = 40

export function useAudioWaveform(audioDetected: boolean, confidence: number): number[] {
  const [bars, setBars] = useState<number[]>(() => Array(BAR_COUNT).fill(0.06))
  const phaseRef = useRef(0)

  useEffect(() => {
    const id = window.setInterval(() => {
      phaseRef.current += 1
      const phase = phaseRef.current

      setBars(
        Array.from({ length: BAR_COUNT }, (_, i) => {
          if (!audioDetected) {
            return 0.04 + Math.random() * 0.08
          }
          const wave = Math.abs(Math.sin(phase * 0.22 + i * 0.45))
          const jitter = Math.random() * 0.15
          const amplitude = 0.25 + confidence * 0.7
          return Math.min(1, 0.08 + wave * amplitude + jitter * confidence)
        }),
      )
    }, 110)

    return () => window.clearInterval(id)
  }, [audioDetected, confidence])

  return bars
}
