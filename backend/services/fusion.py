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

_HIGH_CONFIDENCE_THRESHOLD = 0.85
_MODERATE_CONFIDENCE_THRESHOLD = 0.75


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
            audio_confidence=audio.audioConfidence,
            camera_detected=camera.cameraDetected,
            vehicle_confidence=camera.vehicleConfidence,
        )

        return DetectionEvent(
            audioDetected=audio.audioDetected,
            audioConfidence=audio.audioConfidence,
            sirenType=audio.sirenType,
            cameraDetected=camera.cameraDetected,
            vehicleConfidence=camera.vehicleConfidence,
            vehicleType=camera.vehicleType,
            direction=camera.direction,
            alertLevel=alert_level,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _determine_alert_level(
        audio_detected: bool,
        audio_confidence: float,
        camera_detected: bool,
        vehicle_confidence: float,
    ) -> AlertLevel:
        if audio_detected and camera_detected:
            if audio_confidence >= _HIGH_CONFIDENCE_THRESHOLD and vehicle_confidence >= _HIGH_CONFIDENCE_THRESHOLD:
                return AlertLevel.CRITICAL
            return AlertLevel.WARNING

        if audio_detected or camera_detected:
            confidence = audio_confidence if audio_detected else vehicle_confidence
            return AlertLevel.WARNING if confidence >= _MODERATE_CONFIDENCE_THRESHOLD else AlertLevel.INFO

        return AlertLevel.NONE
