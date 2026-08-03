export type AlertLevel = 'Monitoring' | 'Possible' | 'Confirmed' | 'Critical'

export type VehicleType = 'Ambulance' | 'Police' | 'Fire Truck' | 'None'

export type SirenType = 'Ambulance' | 'Police' | 'Fire Truck' | 'Air Horn' | 'None'

export type Direction =
  | 'Front'
  | 'Front Left'
  | 'Front Right'
  | 'Rear'
  | 'Rear Left'
  | 'Rear Right'
  | 'Left'
  | 'Right'

export type HealthStatus = 'online' | 'degraded' | 'offline'

export type DriveMode = 'Comfort' | 'Eco' | 'Sport' | 'Off-Road'

export interface DetectionState {
  audioDetected: boolean
  audioConfidence: number
  sirenType: SirenType
  cameraDetected: boolean
  vehicleConfidence: number
  vehicleType: VehicleType
  direction: Direction
  alertLevel: AlertLevel
  boundingBox: BoundingBox | null
}

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface DetectionHistoryEntry {
  id: string
  timestamp: string
  confidence: number
  alertLevel: AlertLevel
  vehicleType: VehicleType
  direction: Direction
}

export interface SystemComponentHealth {
  id: string
  name: string
  status: HealthStatus
  detail: string
}

export interface DetectionScenario {
  id: string
  label: string
  weight: number
  state: DetectionState
}
