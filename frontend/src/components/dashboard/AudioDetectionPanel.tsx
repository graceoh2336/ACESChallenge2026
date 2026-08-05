import { AudioLines } from 'lucide-react'
import { PanelHeader } from './PanelHeader'
import { ConfidenceBar } from './ConfidenceBar'
import { AudioWaveform } from '../charts/AudioWaveform'
import { MicPulseIndicator } from '../charts/MicPulseIndicator'
import { useAudioWaveform } from '../../hooks/useAudioWaveform'
import type { DetectionState } from '../../types'

interface AudioDetectionPanelProps {
  detection: DetectionState
}

export function AudioDetectionPanel({ detection }: AudioDetectionPanelProps) {
  const { audioDetected, audioConfidence, sirenType } = detection
  const bars = useAudioWaveform(audioDetected, audioConfidence)

  return (
    <section className="glass-panel flex h-full flex-col rounded-2xl">
      <PanelHeader
        icon={<AudioLines className="h-4 w-4" />}
        title="Audio Detection"
        subtitle="TensorFlow · YAMNet Classifier"
      />

      <div className="flex flex-1 flex-col gap-4 p-4">
        <div className="flex items-center gap-4">
          <MicPulseIndicator active={audioDetected} />
          <div className="flex-1 leading-tight">
            <p className="text-[11px] uppercase tracking-[0.16em] text-cockpit-400">Siren Type</p>
            <p className="font-display text-xl font-semibold text-cockpit-100">
              {audioDetected ? sirenType : 'No Siren Detected'}
            </p>
          </div>
        </div>

        <AudioWaveform bars={bars} active={audioDetected} />

        <ConfidenceBar
          label="Audio Confidence"
          value={audioConfidence}
          tone={audioConfidence > 0.75 ? 'red' : audioConfidence > 0.4 ? 'amber' : 'green'}
        />
      </div>
    </section>
  )
}
