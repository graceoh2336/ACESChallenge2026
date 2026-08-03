import { useEffect, useState } from 'react'
import { DashboardLayout } from '../components/layout/DashboardLayout'
import { CameraPanel } from '../components/camera/CameraPanel'
import { EmergencyPanel } from '../components/dashboard/EmergencyPanel'
import { AudioDetectionPanel } from '../components/dashboard/AudioDetectionPanel'
import { RadarPanel } from '../components/radar/RadarPanel'
import { SystemHealthPanel } from '../components/status/SystemHealthPanel'
import { DetectionHistory } from '../components/dashboard/DetectionHistory'
import { useSimulatedDetection } from '../hooks/useSimulatedDetection'
import { useDetectionWebSocket } from '../hooks/useDetectionWebSocket'
import { detectionScenarios } from '../data/mockData'
import type { ConnectionStatus, DetectionHistoryEntry, DetectionState } from '../types'

const HISTORY_LIMIT = 10

// Read once at module load — this is a build-time deployment switch, not
// runtime state, so branching hook usage on it below doesn't violate the
// rules of hooks (the branch never changes across renders of a given build).
const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA === 'true'

// Shown for live mode before the first WebSocket message arrives.
const IDLE_FALLBACK_STATE: DetectionState = detectionScenarios[0].state

function useLiveDetectionHistory(
  detection: DetectionState | null,
  lastMessageAt: Date | null,
): DetectionHistoryEntry[] {
  const [history, setHistory] = useState<DetectionHistoryEntry[]>([])

  useEffect(() => {
    if (!detection || !lastMessageAt || detection.alertLevel === 'Monitoring') return

    const entry: DetectionHistoryEntry = {
      id: `live-${lastMessageAt.getTime()}`,
      timestamp: lastMessageAt.toISOString(),
      confidence: Math.max(detection.audioConfidence, detection.vehicleConfidence),
      alertLevel: detection.alertLevel,
      vehicleType: detection.vehicleType,
      direction: detection.direction,
    }

    setHistory((prev) => (prev[0]?.id === entry.id ? prev : [entry, ...prev].slice(0, HISTORY_LIMIT)))
  }, [detection, lastMessageAt])

  return history
}

interface DashboardPanelsProps {
  state: DetectionState
  history: DetectionHistoryEntry[]
  connectionStatus: ConnectionStatus
  isMock: boolean
  lastMessageAt: Date | null
  error: string | null
  version: string
}

function DashboardPanels({
  state,
  history,
  connectionStatus,
  isMock,
  lastMessageAt,
  error,
  version,
}: DashboardPanelsProps) {
  return (
    <DashboardLayout
      driveMode="Comfort"
      version={version}
      connectionStatus={connectionStatus}
      isMock={isMock}
      lastMessageAt={lastMessageAt}
    >
      <div className="flex flex-col gap-3 sm:gap-4">
        <div className="grid grid-cols-1 gap-3 sm:gap-4 lg:grid-cols-3">
          <div className="h-[420px] lg:col-span-2 lg:h-[480px]">
            <CameraPanel detection={state} />
          </div>
          <div className="h-[480px] lg:col-span-1">
            <EmergencyPanel detection={state} />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 sm:gap-4 xl:grid-cols-4">
          <div className="h-[360px]">
            <AudioDetectionPanel detection={state} />
          </div>
          <div className="h-[360px]">
            <RadarPanel detection={state} />
          </div>
          <div className="h-[360px]">
            <SystemHealthPanel
              connectionStatus={connectionStatus}
              isMock={isMock}
              lastMessageAt={lastMessageAt}
              error={error}
            />
          </div>
          <div className="h-[360px]">
            <DetectionHistory entries={history} />
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}

function MockDashboardPage() {
  const { state, history, lastUpdated } = useSimulatedDetection()

  return (
    <DashboardPanels
      state={state}
      history={history}
      connectionStatus="connected"
      isMock
      lastMessageAt={lastUpdated}
      error={null}
      version="v0.2.0-sim"
    />
  )
}

function LiveDashboardPage() {
  const { detection, connectionStatus, lastMessageAt, error } = useDetectionWebSocket()
  const history = useLiveDetectionHistory(detection, lastMessageAt)

  return (
    <DashboardPanels
      state={detection ?? IDLE_FALLBACK_STATE}
      history={history}
      connectionStatus={connectionStatus}
      isMock={false}
      lastMessageAt={lastMessageAt}
      error={error}
      version="v0.2.0-live"
    />
  )
}

export function DashboardPage() {
  return USE_MOCK_DATA ? <MockDashboardPage /> : <LiveDashboardPage />
}
