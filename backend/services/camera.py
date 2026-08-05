"""Camera vehicle-detection service, backed by the real OpenCV detector.

This wraps `lights.VisualEmergencyDetector` (repo root `lights.py` — the
newer, more refined sibling of `bestattempt.py`; both are near-identical
blue-emergency-light detectors, but `lights.py` adds the alert-zone /
sky-false-positive checks) behind the same interface the previous simulated
service exposed, so `services/fusion.py` and `websocket.py` do not need to
change. The detector algorithm itself is untouched — this module only owns
capture lifecycle (open once, read continuously on a background thread,
release on shutdown) and translates its output dict into a `CameraReading`.

A background thread is required rather than reading frames inline inside
`generate_reading()`: that method is called synchronously from within
`DetectionBroadcaster`'s asyncio loop (see websocket.py), and a blocking
`cv2.VideoCapture.read()` there would stall every WebSocket client.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

# lights.py lives at the repo root and stays there, unmodified — the
# detector algorithm must not be rewritten, only imported and adapted.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lights  # noqa: E402  (sys.path must be patched before this import)

# The real audio service is a random simulation, not a live siren detector,
# so boosting visual sensitivity off it would be misleading. `lights.py`
# exposes this as a public module-level tunable for exactly this kind of
# external override — live_camera.py does the same thing to bestattempt.py.
lights.AUDIO_SIREN_DETECTED = False

from config import settings  # noqa: E402
from models import BoundingBox, CameraReading, Direction, VehicleType  # noqa: E402

logger = logging.getLogger(__name__)

# lights.VisualEmergencyDetector.get_position_label() buckets the detected
# blob into thirds of the frame width; map those onto the existing Direction
# enum rather than adding new schema.
_POSITION_TO_DIRECTION = {
    "left": Direction.FRONT_LEFT,
    "centre": Direction.FRONT,
    "right": Direction.FRONT_RIGHT,
}

_DEMO_DIR = _BACKEND_DIR / "demo"
_DEMO_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _resolve_camera_source() -> "int | str":
    """CAMERA_SOURCE=0 -> webcam index 0. A non-numeric value is a video file
    path — CAMERA_SOURCE=backend/demo/x.mp4 and CAMERA_SOURCE=demo/x.mp4 both
    resolve, checked against the repo root and the backend/ directory so it
    works regardless of which one the process was launched from. Absolute
    paths, and anything that doesn't match either, are passed straight to
    cv2.VideoCapture (relative to the process's own cwd) unchanged.
    """
    raw = settings.camera_source.strip()

    if raw.isdigit():
        return int(raw)

    candidate = Path(raw)
    if not candidate.is_absolute():
        for base in (_REPO_ROOT, _BACKEND_DIR):
            resolved = base / raw
            if resolved.exists():
                return str(resolved)

    return raw


def _find_demo_video() -> Optional[Path]:
    """First video file found in backend/demo/, if any (sorted for determinism)."""
    if not _DEMO_DIR.is_dir():
        return None

    candidates = sorted(
        path
        for path in _DEMO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in _DEMO_VIDEO_EXTENSIONS
    )
    return candidates[0] if candidates else None


class CameraDetectionService:
    """Runs the OpenCV detector continuously against a VideoCapture on a
    background thread and hands out the latest CameraReading on demand.
    """

    def __init__(self, detection_probability: float = 0.5):
        # Unused by the real detector; kept only so the constructor still
        # accepts the argument websocket.py already passes it.
        del detection_probability

        self._source = _resolve_camera_source()
        self._detector = lights.VisualEmergencyDetector()
        self._capture: Optional[cv2.VideoCapture] = None

        self._latest_reading = CameraReading(
            cameraDetected=False,
            vehicleConfidence=0.0,
            vehicleType=VehicleType.UNKNOWN,
            direction=None,
            boundingBox=None,
            frameWidth=lights.FRAME_WIDTH,
            frameHeight=lights.FRAME_HEIGHT,
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Opens the VideoCapture once and starts the continuous frame loop.

        Never raises: a camera that can't be opened (or any other startup
        failure) leaves the service running with no live detections rather
        than taking down the FastAPI process it's part of.
        """
        if self._thread is not None:
            return

        try:
            capture = self._open_working_capture()
        except Exception:
            logger.exception("Unexpected error starting camera detection; continuing without it")
            return

        if capture is None:
            logger.warning(
                "Camera detection disabled — no working video source. "
                "The API keeps running; cameraDetected will stay false."
            )
            return

        self._capture = capture
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="camera-detection-loop", daemon=True
        )
        self._thread.start()
        logger.info("Camera detection loop started (source=%r)", self._source)

    def _open_working_capture(self) -> Optional[cv2.VideoCapture]:
        """Tries the configured source, then falls back to a demo video."""
        capture = self._try_open(self._source)
        if capture is not None:
            return capture

        logger.warning("Configured camera source %r could not be opened.", self._source)

        demo_video = _find_demo_video()
        if demo_video is None:
            logger.warning(
                "No demo video found in %s to fall back to.", _DEMO_DIR
            )
            return None

        if not settings.camera_auto_fallback:
            logger.warning(
                "A demo video is available at %s — set CAMERA_SOURCE=%s to use it "
                "directly, or CAMERA_AUTO_FALLBACK=true to switch to it automatically "
                "whenever the configured source fails.",
                demo_video,
                demo_video.relative_to(_REPO_ROOT),
            )
            return None

        logger.warning("CAMERA_AUTO_FALLBACK is enabled — switching to demo video: %s", demo_video)
        capture = self._try_open(str(demo_video))
        if capture is None:
            logger.warning("Demo video %s could not be opened either.", demo_video)
            return None

        self._source = str(demo_video)
        return capture

    @staticmethod
    def _try_open(source: "int | str") -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            return None
        return capture

    def stop(self) -> None:
        """Stops the frame loop and cleanly releases the camera."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        logger.info("Camera released")

    def generate_reading(self) -> CameraReading:
        """Returns the latest reading produced by the background frame loop.

        Named to match the simulated service it replaces so FusionService
        (services/fusion.py) needs no changes.
        """
        with self._lock:
            return self._latest_reading

    def _run(self) -> None:
        is_file_source = isinstance(self._source, str)

        while not self._stop_event.is_set():
            ok, frame = self._capture.read()

            if not ok:
                if is_file_source:
                    # Loop demo clips so the stream never runs dry.
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Camera read failed; retrying (source=%r)", self._source)
                time.sleep(0.5)
                continue

            output_data, _display, _blue_mask, _blue_change_mask = self._detector.process_frame(frame)
            reading = self._to_camera_reading(output_data)

            with self._lock:
                self._latest_reading = reading

    @staticmethod
    def _to_camera_reading(output_data: dict) -> CameraReading:
        state = output_data["visual_state"]

        direction = _POSITION_TO_DIRECTION.get(output_data.get("best_position"))

        bbox = output_data.get("best_bbox")
        bounding_box = BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3]) if bbox else None

        # The detector only confirms *that* a flashing blue light is present,
        # not which kind of emergency vehicle it belongs to — that
        # classification is future TensorFlow-model territory.
        return CameraReading(
            cameraDetected=state == "detected",
            vehicleConfidence=output_data["visual_confidence"],
            vehicleType=VehicleType.UNKNOWN,
            direction=direction,
            boundingBox=bounding_box,
            frameWidth=lights.FRAME_WIDTH,
            frameHeight=lights.FRAME_HEIGHT,
        )
