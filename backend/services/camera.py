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
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

# lights.py lives at the repo root and stays there, unmodified — the
# detector algorithm must not be rewritten, only imported and adapted.
_REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _resolve_camera_source() -> "int | str":
    """CAMERA_SOURCE=0 -> webcam index 0. CAMERA_SOURCE=demo.mp4 -> a video
    file, resolved against the repo root if a same-named file lives there."""
    raw = settings.camera_source.strip()

    if raw.isdigit():
        return int(raw)

    candidate = Path(raw)
    if not candidate.is_absolute():
        repo_candidate = _REPO_ROOT / raw
        if repo_candidate.exists():
            return str(repo_candidate)

    return raw


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
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Opens the VideoCapture once and starts the continuous frame loop."""
        if self._thread is not None:
            return

        self._capture = cv2.VideoCapture(self._source)
        if not self._capture.isOpened():
            logger.error("Could not open camera source: %r", self._source)
            self._capture = None
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="camera-detection-loop", daemon=True
        )
        self._thread.start()
        logger.info("Camera detection loop started (source=%r)", self._source)

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
        )
