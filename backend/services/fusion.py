"""Sensor fusion: combines audio and camera readings into one DetectionEvent
and derives an overall alert level.

This is the seam where audio + camera evidence get reconciled. Keeping it
separate from the individual sensor services means the alert-level policy can
evolve (or later factor in real model confidences) without touching either
sensor's simulation/inference logic.
"""

from datetime import datetime, timezone

from models import AlertLevel, DetectionEvent
from services.audio import AudioDetectionService
from services.camera import CameraDetectionService


class FusionService:
    def __init__(
        self,
        audio_service: AudioDetectionService,
        camera_service: CameraDetectionService,
    ):
        self._audio_service = audio_service
        self._camera_service = camera_service

    def generate_event(self) -> DetectionEvent:
        audio = self._audio_service.generate_reading()
        camera = self._camera_service.generate_reading()

        alert_level = self._determine_alert_level(
            audio_detected=audio.audioDetected,
            camera_detected=camera.cameraDetected,
        )

        return DetectionEvent(
            audioDetected=audio.audioDetected,
            audioConfidence=audio.audioConfidence,
            sirenType=audio.sirenType,
            cameraDetected=camera.cameraDetected,
            vehicleConfidence=camera.vehicleConfidence,
            vehicleType=camera.vehicleType,
            direction=camera.direction,
            boundingBox=camera.boundingBox,
            frameWidth=camera.frameWidth,
            frameHeight=camera.frameHeight,
            alertLevel=alert_level,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _determine_alert_level(audio_detected: bool, camera_detected: bool) -> AlertLevel:
        # Presence-only policy: exactly one sensor firing is "possible"
        # (AlertLevel.INFO, frontend label "Possible Emergency Vehicle"),
        # both firing together is "confirmed" (AlertLevel.WARNING, frontend
        # label "Emergency Vehicle Confirmed"). No confidence-based upgrade
        # to CRITICAL — a single fused "Confirmed" tier is what's wanted.
        if audio_detected and camera_detected:
            return AlertLevel.WARNING

        if audio_detected or camera_detected:
            return AlertLevel.INFO

        return AlertLevel.NONE
