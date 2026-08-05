import { useEffect, useRef, useState } from 'react'
import { detectionScenarios, initialDetectionHistory } from '../data/mockData'
import type { DetectionHistoryEntry, DetectionState } from '../types'

const HISTORY_LIMIT = 10
const TICK_MS = 5000

function pickWeightedScenario(excludeId: string | null) {
  const pool = detectionScenarios.filter((s) => s.id !== excludeId)
  const totalWeight = pool.reduce((sum, s) => sum + s.weight, 0)
  let roll = Math.random() * totalWeight

  for (const scenario of pool) {
    roll -= scenario.weight
    if (roll <= 0) return scenario
  }
  return pool[0]
}

interface SimulatedDetection {
  state: DetectionState
  history: DetectionHistoryEntry[]
  lastUpdated: Date
}

export function useSimulatedDetection(): SimulatedDetection {
  const [scenarioId, setScenarioId] = useState(detectionScenarios[0].id)
  const [state, setState] = useState<DetectionState>(detectionScenarios[0].state)
  const [history, setHistory] = useState<DetectionHistoryEntry[]>(initialDetectionHistory)
  const [lastUpdated, setLastUpdated] = useState(() => new Date())
  const scenarioIdRef = useRef(scenarioId)
  scenarioIdRef.current = scenarioId

  useEffect(() => {
    const id = window.setInterval(() => {
      const next = pickWeightedScenario(scenarioIdRef.current)
      setScenarioId(next.id)
      setState(next.state)
      setLastUpdated(new Date())

      if (next.state.alertLevel !== 'Monitoring') {
        setHistory((prev) => {
          const entry: DetectionHistoryEntry = {
            id: `${next.id}-${Date.now()}`,
            timestamp: new Date().toISOString(),
            confidence: Math.max(next.state.audioConfidence, next.state.vehicleConfidence),
            alertLevel: next.state.alertLevel,
            vehicleType: next.state.vehicleType,
            direction: next.state.direction,
          }
          return [entry, ...prev].slice(0, HISTORY_LIMIT)
        })
      }
    }, TICK_MS)

    return () => window.clearInterval(id)
  }, [])

  return { state, history, lastUpdated }
}
