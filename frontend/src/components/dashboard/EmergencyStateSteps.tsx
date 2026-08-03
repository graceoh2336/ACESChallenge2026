import { motion } from 'framer-motion'
import { Eye, ShieldAlert, ShieldCheck } from 'lucide-react'
import { cn } from '../../utils/cn'
import type { AlertLevel } from '../../types'

interface StepDef {
  key: 'monitoring' | 'possible' | 'confirmed'
  label: string
  icon: typeof Eye
  activeClasses: string
}

const steps: StepDef[] = [
  {
    key: 'monitoring',
    label: 'Monitoring',
    icon: Eye,
    activeClasses: 'border-status-green/60 bg-status-green/10 text-status-green',
  },
  {
    key: 'possible',
    label: 'Possible Emergency Vehicle',
    icon: ShieldAlert,
    activeClasses: 'border-status-amber/60 bg-status-amber/10 text-status-amber',
  },
  {
    key: 'confirmed',
    label: 'Emergency Vehicle Confirmed',
    icon: ShieldCheck,
    activeClasses: 'border-status-red/60 bg-status-red/10 text-status-red',
  },
]

function stepKeyForLevel(level: AlertLevel): StepDef['key'] {
  if (level === 'Possible') return 'possible'
  if (level === 'Confirmed' || level === 'Critical') return 'confirmed'
  return 'monitoring'
}

export function EmergencyStateSteps({ alertLevel }: { alertLevel: AlertLevel }) {
  const activeKey = stepKeyForLevel(alertLevel)

  return (
    <div className="grid grid-cols-3 gap-2">
      {steps.map((step) => {
        const isActive = step.key === activeKey
        const Icon = step.icon
        return (
          <motion.div
            key={step.key}
            className={cn(
              'flex flex-col items-center gap-1.5 rounded-lg border px-2 py-2.5 text-center transition-colors',
              isActive ? step.activeClasses : 'border-cockpit-600/60 bg-cockpit-800/40 text-cockpit-400',
            )}
            animate={isActive ? { scale: [1, 1.03, 1] } : { scale: 1 }}
            transition={{ duration: 1.4, repeat: isActive ? Infinity : 0 }}
          >
            <Icon className="h-4 w-4 shrink-0" />
            <span className="text-[10.5px] font-medium leading-tight">{step.label}</span>
          </motion.div>
        )
      })}
    </div>
  )
}
