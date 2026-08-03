import type { ReactNode } from 'react'
import { TopStatusBar } from './TopStatusBar'
import { BottomStatusBar } from './BottomStatusBar'
import type { DriveMode } from '../../types'

interface DashboardLayoutProps {
  children: ReactNode
  driveMode: DriveMode
  version: string
}

export function DashboardLayout({ children, driveMode, version }: DashboardLayoutProps) {
  return (
    <div className="grid-overlay relative min-h-screen bg-cockpit-950 p-3 sm:p-4">
      <div
        className="pointer-events-none absolute inset-0 opacity-60"
        style={{
          background:
            'radial-gradient(1200px circle at 15% 0%, rgba(59,167,255,0.06), transparent 55%), radial-gradient(1000px circle at 85% 100%, rgba(255,59,78,0.05), transparent 55%)',
        }}
      />

      <div className="relative mx-auto flex max-w-[1800px] flex-col gap-3 sm:gap-4">
        <TopStatusBar />
        <main>{children}</main>
        <BottomStatusBar driveMode={driveMode} version={version} />
      </div>
    </div>
  )
}
