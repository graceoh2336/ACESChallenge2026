function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

interface DevCameraReadoutProps {
  /** Raw (pre-normalization) parsed WebSocket payload, or null in mock mode. */
  rawMessage: unknown
}

/**
 * Temporary dev-only readout of the raw WebSocket camera fields, added to
 * verify the backend-to-frontend mapping while chasing a display bug where
 * cameraDetected=true wasn't visibly reflected in the UI. Gated on
 * import.meta.env.DEV (statically false in `vite build`, so this is dead
 * code — and stripped — in production). Safe to delete this file once it's
 * no longer needed for verification.
 */
export function DevCameraReadout({ rawMessage }: DevCameraReadoutProps) {
  if (!import.meta.env.DEV) return null

  if (!isRecord(rawMessage)) {
    return (
      <pre className="absolute left-4 top-14 z-20 rounded-md bg-black/75 px-2 py-1.5 font-mono text-[10px] leading-tight text-lime-300">
        DEV raw · waiting for WebSocket data…
      </pre>
    )
  }

  const fields = {
    cameraDetected: rawMessage.cameraDetected,
    vehicleConfidence: rawMessage.vehicleConfidence,
    vehicleType: rawMessage.vehicleType,
    direction: rawMessage.direction,
    boundingBox: rawMessage.boundingBox,
    frameWidth: rawMessage.frameWidth,
    frameHeight: rawMessage.frameHeight,
  }

  return (
    <pre className="absolute left-4 top-14 z-20 max-w-[240px] whitespace-pre-wrap break-words rounded-md bg-black/75 px-2 py-1.5 font-mono text-[10px] leading-tight text-lime-300">
      {`DEV raw camera fields\n${JSON.stringify(fields, null, 1)}`}
    </pre>
  )
}
