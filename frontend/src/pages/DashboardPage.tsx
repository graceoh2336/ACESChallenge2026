import { DashboardLayout } from '../components/layout/DashboardLayout'
import { CameraPanel } from '../components/camera/CameraPanel'
import { EmergencyPanel } from '../components/dashboard/EmergencyPanel'
import { AudioDetectionPanel } from '../components/dashboard/AudioDetectionPanel'
import { RadarPanel } from '../components/radar/RadarPanel'
import { SystemHealthPanel } from '../components/status/SystemHealthPanel'
import { DetectionHistory } from '../components/dashboard/DetectionHistory'
import { useSimulatedDetection } from '../hooks/useSimulatedDetection'

export function DashboardPage() {
  const { state, history } = useSimulatedDetection()

  return (
    <DashboardLayout driveMode="Comfort" version="v0.1.0-sim">
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
            <SystemHealthPanel />
          </div>
          <div className="h-[360px]">
            <DetectionHistory entries={history} />
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}
