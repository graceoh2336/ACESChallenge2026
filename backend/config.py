"""Application configuration, sourced from environment variables with sane defaults."""

import os
from dataclasses import dataclass, field
from typing import List


def _parse_origins(raw: str) -> List[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Emergency Vehicle Detection API"
    app_version: str = "1.0.0"

    # How often (seconds) the simulated detection pipeline emits a new event.
    broadcast_interval_seconds: float = float(os.getenv("BROADCAST_INTERVAL_SECONDS", "1.0"))

    # Probability that a given simulated tick detects a siren.
    audio_detection_probability: float = float(os.getenv("AUDIO_DETECTION_PROBABILITY", "0.55"))

    # Unused by the real OpenCV camera service; kept so CameraDetectionService's
    # constructor signature (and websocket.py's call site) don't need to change.
    camera_detection_probability: float = float(os.getenv("CAMERA_DETECTION_PROBABILITY", "0.5"))

    # "0" (or another integer) for a live webcam index, or a video file path
    # (e.g. "demo.mp4") to loop a recorded clip through the detector.
    camera_source: str = os.getenv("CAMERA_SOURCE", "0")

    # When the configured camera_source can't be opened and a demo video
    # exists under backend/demo/, automatically switch to it instead of just
    # logging that it's available. Off by default — falling back to a demo
    # clip silently is a surprising thing for a "live" feed to do.
    camera_auto_fallback: bool = os.getenv("CAMERA_AUTO_FALLBACK", "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    cors_origins: List[str] = field(
        default_factory=lambda: _parse_origins(
            os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
        )
    )


settings = Settings()
