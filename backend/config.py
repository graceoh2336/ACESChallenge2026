"""Application configuration, sourced from environment variables (optionally
via a backend/.env file) with sane defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Loads backend/.env if it exists. Real environment variables set in the
# shell/process still win over anything in the file (load_dotenv's default
# override=False), so `CAMERA_SOURCE=0 uvicorn ...` still works as before.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _parse_origins(raw: str) -> List[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes")


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
    camera_auto_fallback: bool = _parse_bool(os.getenv("CAMERA_AUTO_FALLBACK", "false"))

    # Verbose per-frame detection logging, saved debug frames under
    # backend/debug_output/, and the /api/camera/debug-stream endpoint (raw
    # frame + detector display + blue mask + change mask). Off by default —
    # meaningful CPU/disk cost, and only useful while investigating detector
    # accuracy, not for normal operation.
    opencv_debug: bool = _parse_bool(os.getenv("OPENCV_DEBUG", "false"))

    cors_origins: List[str] = field(
        default_factory=lambda: _parse_origins(
            os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
        )
    )


settings = Settings()
