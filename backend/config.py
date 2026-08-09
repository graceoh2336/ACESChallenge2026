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


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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
        default_factory=lambda: _parse_csv(
            os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
        )
    )

    # --- YOLO vehicle-first vision pipeline -------------------------------
    # When true, services/camera.py runs Ultralytics YOLO vehicle detection
    # alongside lights.py's blue-light detector and only trusts flashing-light
    # evidence that's spatially associated with a detected vehicle (see
    # services/yolo_detector.py and services/vision_fusion.py). When false,
    # the system behaves exactly as it did before this upgrade: lights.py's
    # raw blue/flashing-blob output drives CameraReading directly. This is
    # the safe rollback switch — flip to false to restore the pre-YOLO
    # pipeline with no other changes needed.
    use_yolo: bool = _parse_bool(os.getenv("USE_YOLO", "true"))

    # Ultralytics model name/path. A bare name (e.g. "yolo11n.pt") is
    # resolved under backend/.cache/yolo/ and auto-downloaded there on first
    # use (mirrors how AUDIO's YAMNet model is cached under
    # backend/.tfhub_cache/) — never littered into the repo root or whatever
    # directory the process happened to be launched from. An absolute path,
    # or one that already exists relative to the repo root/backend dir, is
    # used as-is. "n" (nano) is the smallest/fastest variant in the YOLO11
    # family and the one meant for eventual Raspberry Pi 5 deployment.
    yolo_model: str = os.getenv("YOLO_MODEL", "yolo11n.pt")

    # Minimum YOLO detection confidence before a box counts as a vehicle.
    # Measured against this project's own demo footage (see the ACES
    # report): daytime footage (video2.mp4, video3.mp4) clears 0.5-0.85 for
    # real cars, comfortably above any reasonable bar. The close-range night
    # Garda car in video.mp4 is a much harder case for a pretrained COCO
    # model — extreme blue lens-flare bloom and full-frame overexposure push
    # its own "car" confidence down to 0.15-0.45 even though the same box
    # position is picked up consistently frame after frame. 0.45 (the
    # initially-planned default) missed that vehicle almost entirely; 0.25
    # catches it reliably while still sitting well below daytime cars'
    # typical confidence, so it doesn't meaningfully add noise there — a
    # low-confidence vehicle box alone still can't trigger an emergency
    # detection without associated flashing-light evidence (see
    # services/vision_fusion.py), so the risk of lowering this is small.
    yolo_confidence_threshold: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))

    # How often (times per second) YOLO actually runs inference. Running it
    # on every frame is unnecessary — vehicles don't move fast enough
    # relative to the frame rate to need it, and it would compete with
    # lights.py's own per-frame flash-timing analysis for CPU. Detections are
    # cached between runs (see services/camera.py) so every frame still has
    # a vehicle ROI to test light evidence against.
    # 15, not the originally-planned 5: measured on video.mp4 (the hardest
    # demo clip — see yolo_confidence_threshold's comment), YOLO's own "car"
    # confidence for the real vehicle flickers above and below the
    # acceptance threshold frame to frame under heavy lens flare, so a
    # sparser inference rate leaves the vehicle cache stale (and therefore
    # "no vehicle") right when a light-association check needs it most. An
    # offline sweep (8/15/20 Hz, plus an every-frame upper bound) against all
    # three demo clips found 15Hz recovers most of that gap on the hard clip
    # with zero measurable difference on the two easier ones, and it's still
    # cheap: this model runs ~25-27ms/frame on an M-series CPU, so 15Hz costs
    # well under 40% of one core on the desktop this was built and demoed
    # on. Lower this substantially for Raspberry Pi 5 (see the ACES report's
    # Raspberry Pi section) — that CPU won't sustain 15Hz.
    yolo_inference_fps: float = float(os.getenv("YOLO_INFERENCE_FPS", "15"))

    # COCO classes (Ultralytics' pretrained vocabulary) treated as "vehicle"
    # for the purposes of this pipeline. Comma-separated; anything else YOLO
    # detects (person, traffic light, stop sign, ...) is ignored outright.
    yolo_vehicle_classes: List[str] = field(
        default_factory=lambda: _parse_csv(os.getenv("YOLO_VEHICLE_CLASSES", "car,truck,bus,motorcycle"))
    )

    # Fractional padding added around a YOLO vehicle box (relative to that
    # box's own width/height) before testing whether a flashing-light
    # candidate from lights.py falls inside it — a roof-mounted lightbar or
    # grille light often sits just outside the tight vehicle box YOLO draws,
    # so a small ROI margin catches those without accepting evidence that's
    # actually off the vehicle entirely.
    yolo_roi_padding: float = float(os.getenv("YOLO_ROI_PADDING", "0.15"))

    # How many frames a cached YOLO detection remains valid for before being
    # treated as stale (vehicle presumed gone) if no newer YOLO run has
    # confirmed it. Expressed as a multiple of the YOLO inference interval
    # (1 / yolo_inference_fps) rather than a fixed second count, so it scales
    # sensibly if the inference rate is changed.
    yolo_stale_intervals: float = float(os.getenv("YOLO_STALE_INTERVALS", "4"))

    # Rolling window (in main detection-loop frames, i.e. lights.py's own
    # frame rate — not YOLO's) used to judge whether vehicle+light
    # association is a stable, repeated pattern rather than one-frame noise
    # (Phase 4 temporal stability). Deliberately separate from lights.py's
    # own confirmed_flash_count bookkeeping — this only tracks *whether the
    # flashing light kept overlapping the same vehicle region*, not whether
    # the flash pattern itself looks genuine (lights.py already owns that).
    yolo_association_window: int = int(os.getenv("YOLO_ASSOCIATION_WINDOW", "15"))

    # Fraction of the association window that must show vehicle+light
    # overlap before the combined visual state is allowed to reach
    # "confirmed" (as opposed to the weaker "possible").
    yolo_association_ratio_required: float = float(os.getenv("YOLO_ASSOCIATION_RATIO_REQUIRED", "0.45"))


settings = Settings()
