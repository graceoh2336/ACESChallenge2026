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
    # (e.g. "demo/video.mp4") to loop a recorded clip through the detector.
    # Defaults to the bundled demo clip, which also supplies YAMNet's audio
    # (see audio_source below) — one file, one source of truth for both.
    camera_source: str = os.getenv("CAMERA_SOURCE", "demo/video.mp4")

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

    # When true, audio detection is real YAMNet/TensorFlow inference
    # (services/tensorflow_audio.py). When false, falls back to the random
    # simulated service (services/audio.py) — same behaviour as before this
    # was wired up.
    use_real_tensorflow: bool = _parse_bool(os.getenv("USE_REAL_TENSORFLOW", "true"))

    # "video" (the default) extracts and loops the audio track baked into
    # CAMERA_SOURCE's own video file — no separate WAV to maintain. "mic" (or
    # "live") switches to live microphone capture via sounddevice instead
    # (Raspberry Pi deployment). Any other value is treated as a WAV/audio
    # file path, resolved the same way as CAMERA_SOURCE.
    audio_source: str = os.getenv("AUDIO_SOURCE", "video")

    # YAMNet only accepts 16kHz mono audio; any other source sample rate is
    # resampled to this before inference.
    audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

    # Minimum YAMNet class score before a window counts as a detection.
    audio_confidence_threshold: float = float(os.getenv("AUDIO_CONFIDENCE_THRESHOLD", "0.65"))

    # When the real TensorFlow audio service can't start (model failed to
    # load, no working audio source) and this is true, it falls back to the
    # random simulated readings instead of just reporting "no detection".
    # Off by default — a real "live" sensor silently turning into a random
    # number generator is a surprising and misleading failure mode.
    audio_allow_simulation_fallback: bool = _parse_bool(
        os.getenv("AUDIO_ALLOW_SIMULATION_FALLBACK", "false")
    )

    cors_origins: List[str] = field(
        default_factory=lambda: _parse_origins(
            os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
        )
    )


settings = Settings()
