import type {
  AlertLevel,
  BoundingBox,
  DetectionState,
  Direction,
  SirenType,
  VehicleType,
} from '../types'

const VALID_SIREN_TYPES: readonly SirenType[] = ['Ambulance', 'Police', 'Fire Truck', 'Air Horn', 'None']

const VALID_DIRECTIONS: readonly Direction[] = [
  'Front',
  'Front Left',
  'Front Right',
  'Rear',
  'Rear Left',
  'Rear Right',
  'Left',
  'Right',
]

// The FastAPI backend's enums don't line up 1:1 with the frontend's — these
// maps translate its vocabulary (AlertLevel.WARNING, VehicleType.UNKNOWN, ...)
// into the DetectionState shape the dashboard components already render.
const ALERT_LEVEL_MAP: Record<string, AlertLevel> = {
  None: 'Monitoring',
  Info: 'Possible',
  Warning: 'Confirmed',
  Critical: 'Critical',
  Monitoring: 'Monitoring',
  Possible: 'Possible',
  Confirmed: 'Confirmed',
}

const VEHICLE_TYPE_MAP: Record<string, VehicleType> = {
  Ambulance: 'Ambulance',
  Police: 'Police',
  'Fire Truck': 'Fire Truck',
  Unknown: 'None',
  None: 'None',
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function clampConfidence(value: unknown): number | null {
  if (typeof value !== 'number' || Number.isNaN(value)) return null
  return Math.min(1, Math.max(0, value))
}

function normalizeSirenType(value: unknown): SirenType {
  return typeof value === 'string' && (VALID_SIREN_TYPES as readonly string[]).includes(value)
    ? (value as SirenType)
    : 'None'
}

function normalizeVehicleType(value: unknown): VehicleType {
  return typeof value === 'string' && value in VEHICLE_TYPE_MAP ? VEHICLE_TYPE_MAP[value] : 'None'
}

function normalizeDirection(value: unknown): Direction {
  return typeof value === 'string' && (VALID_DIRECTIONS as readonly string[]).includes(value)
    ? (value as Direction)
    : 'Front'
}

function normalizeAlertLevel(value: unknown): AlertLevel {
  return typeof value === 'string' && value in ALERT_LEVEL_MAP ? ALERT_LEVEL_MAP[value] : 'Monitoring'
}

// Fallback only: used when the backend event carries no real boundingBox
// (older events, or a state where the detector hasn't tracked anything yet)
// or is missing the source frame dimensions needed to convert one. Derives
// a plausible box from direction + confidence purely to keep the camera
// overlay working instead of going blank.
const DIRECTION_BOX_ORIGIN: Record<Direction, { x: number; y: number }> = {
  Front: { x: 40, y: 30 },
  'Front Left': { x: 15, y: 32 },
  'Front Right': { x: 62, y: 32 },
  Rear: { x: 40, y: 40 },
  'Rear Left': { x: 18, y: 38 },
  'Rear Right': { x: 58, y: 34 },
  Left: { x: 8, y: 36 },
  Right: { x: 70, y: 36 },
}

function deriveBoundingBox(
  cameraDetected: boolean,
  direction: Direction,
  vehicleConfidence: number,
): BoundingBox | null {
  if (!cameraDetected) return null
  const origin = DIRECTION_BOX_ORIGIN[direction]
  const size = 24 + vehicleConfidence * 16
  return { x: origin.x, y: origin.y, width: size, height: size * 1.1 }
}

interface PixelBoundingBox {
  x: number
  y: number
  width: number
  height: number
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

/** Parses the raw (pixel-space) boundingBox the backend attaches to a camera reading. */
function parsePixelBoundingBox(value: unknown): PixelBoundingBox | null {
  if (!isRecord(value)) return null
  const { x, y, width, height } = value
  if (!isFiniteNumber(x) || !isFiniteNumber(y) || !isFiniteNumber(width) || !isFiniteNumber(height)) {
    return null
  }
  return { x, y, width, height }
}

/** A positive frame dimension, or null — used to gate pixel->percent conversion. */
function parseFrameDimension(value: unknown): number | null {
  return isFiniteNumber(value) && value > 0 ? value : null
}

/**
 * Converts a pixel-space box (in the detector's source frame) into the
 * percentage-space BoundingBox the overlay renders with. Percentages scale
 * automatically with the camera panel's own size/aspect ratio via CSS, so no
 * resize handling is needed beyond this one conversion. All values are
 * clamped so the box can never render outside the panel bounds.
 */
function pixelBoxToPercentage(box: PixelBoundingBox, frameWidth: number, frameHeight: number): BoundingBox {
  const x = clamp((box.x / frameWidth) * 100, 0, 100)
  const y = clamp((box.y / frameHeight) * 100, 0, 100)
  const width = clamp((box.width / frameWidth) * 100, 0, 100 - x)
  const height = clamp((box.height / frameHeight) * 100, 0, 100 - y)
  return { x, y, width, height }
}

/**
 * Validates and normalizes a raw (already JSON-parsed) detection payload into
 * the frontend's DetectionState shape. Returns null when the payload is
 * missing/mistyped core fields, so callers can discard it without crashing.
 */
export function normalizeDetectionEvent(raw: unknown): DetectionState | null {
  if (!isRecord(raw)) return null

  const { audioDetected, cameraDetected } = raw
  const audioConfidence = clampConfidence(raw.audioConfidence)
  const vehicleConfidence = clampConfidence(raw.vehicleConfidence)

  if (
    typeof audioDetected !== 'boolean' ||
    typeof cameraDetected !== 'boolean' ||
    audioConfidence === null ||
    vehicleConfidence === null
  ) {
    return null
  }

  const direction = normalizeDirection(raw.direction)

  const pixelBoundingBox = parsePixelBoundingBox(raw.boundingBox)
  const frameWidth = parseFrameDimension(raw.frameWidth)
  const frameHeight = parseFrameDimension(raw.frameHeight)

  // Prefer the real detector box whenever the backend sent one with frame
  // dimensions to convert against; only fall back to the derived placeholder
  // for older events (no boundingBox/frame fields yet) or mock data.
  const boundingBox =
    pixelBoundingBox && frameWidth && frameHeight
      ? pixelBoxToPercentage(pixelBoundingBox, frameWidth, frameHeight)
      : deriveBoundingBox(cameraDetected, direction, vehicleConfidence)

  return {
    audioDetected,
    audioConfidence,
    sirenType: normalizeSirenType(raw.sirenType),
    cameraDetected,
    vehicleConfidence,
    vehicleType: normalizeVehicleType(raw.vehicleType),
    direction,
    alertLevel: normalizeAlertLevel(raw.alertLevel),
    boundingBox,
  }
}

/** Safely parses a raw WebSocket text frame; never throws. */
export function parseDetectionMessage(data: string): DetectionState | null {
  try {
    return normalizeDetectionEvent(JSON.parse(data))
  } catch {
    return null
  }
}
