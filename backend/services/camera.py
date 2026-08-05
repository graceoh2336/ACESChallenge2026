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
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

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

# OPENCV_DEBUG-only: where per-detection evidence frames are saved, and how
# many frame-sets (4 files each) to keep before deleting the oldest — an
# unattended debug session against a looping video would otherwise grow this
# directory without bound.
_DEBUG_DIR = _BACKEND_DIR / "debug_output"
_DEBUG_MAX_SAVED_FRAME_SETS = 200


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


def _build_debug_composite(frame, display, blue_mask, blue_change_mask):
    """2x2 grid: original | detector's own annotated display, over the two
    intermediate masks it made its decision from. Purely our own compositing
    for the debug view — none of lights.py's images or logic are touched."""
    blue_mask_bgr = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    change_mask_bgr = cv2.cvtColor(blue_change_mask, cv2.COLOR_GRAY2BGR)

    tiles = []
    for image, label in (
        (frame, "ORIGINAL"),
        (display, "DETECTOR DISPLAY"),
        (blue_mask_bgr, "BLUE MASK"),
        (change_mask_bgr, "BLUE CHANGE MASK"),
    ):
        tile = image.copy()
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    top = cv2.hconcat([tiles[0], tiles[1]])
    bottom = cv2.hconcat([tiles[2], tiles[3]])
    return cv2.vconcat([top, bottom])


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
        # JPEG bytes of the exact frame the detector last analyzed (no
        # burned-in boxes/text) — the frontend draws its own overlay on top,
        # so this stays the plain feed to avoid double annotations.
        self._latest_frame_jpeg: Optional[bytes] = None

        # OPENCV_DEBUG-only state: the live 4-way composite for
        # /api/camera/debug-stream, and the rolling set of saved evidence
        # frames on disk (see _save_debug_frames).
        self._latest_debug_frame_jpeg: Optional[bytes] = None
        self._debug_saved_frame_sets: Deque[List[Path]] = deque()

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

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        """Returns the most recent plain (unannotated) frame as JPEG bytes,
        for the MJPEG live-preview stream (see routes/camera.py)."""
        with self._lock:
            return self._latest_frame_jpeg

    def get_latest_debug_frame_jpeg(self) -> Optional[bytes]:
        """OPENCV_DEBUG-only: the latest 4-way composite (original, detector
        display, blue mask, blue-change mask), or None if debug mode is off
        or no frame has been processed yet."""
        with self._lock:
            return self._latest_debug_frame_jpeg

    def _run(self) -> None:
        is_file_source = isinstance(self._source, str)

        # A live webcam is naturally paced by its own hardware/driver — read()
        # blocks until the next frame exists. A video file has no such limit:
        # cv2 decodes as fast as the CPU allows, which is faster than the
        # clip's own frame rate, so without pacing it here the file plays back
        # sped up (and speeds up further under light CPU load). Pace file
        # reads to the source's own FPS so playback matches the real clip.
        frame_interval = None
        if is_file_source:
            source_fps = self._capture.get(cv2.CAP_PROP_FPS)
            if source_fps and source_fps > 0:
                frame_interval = 1.0 / source_fps
        next_frame_due = time.monotonic()

        while not self._stop_event.is_set():
            if frame_interval is not None:
                sleep_for = next_frame_due - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                next_frame_due += frame_interval

            ok, frame = self._capture.read()

            if not ok:
                if is_file_source:
                    # Loop demo clips so the stream never runs dry.
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Camera read failed; retrying (source=%r)", self._source)
                time.sleep(0.5)
                continue

            # Resize up front so the frame we cache for streaming is pixel-
            # for-pixel what process_frame() analyzed (it resizes internally
            # too, but 640x480 -> 640x480 there is a no-op) — this keeps the
            # frontend's percentage-space overlay aligned with the image.
            frame = cv2.resize(frame, (lights.FRAME_WIDTH, lights.FRAME_HEIGHT))

            output_data, display, blue_mask, blue_change_mask = self._detector.process_frame(frame)
            reading = self._to_camera_reading(output_data)

            encoded_ok, jpeg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_jpeg = jpeg_buffer.tobytes() if encoded_ok else None

            with self._lock:
                self._latest_reading = reading
                self._latest_frame_jpeg = frame_jpeg

            if settings.opencv_debug:
                self._handle_debug(output_data, frame, display, blue_mask, blue_change_mask)

    @staticmethod
    def _to_camera_reading(output_data: dict) -> CameraReading:
        state = output_data["visual_state"]
        detected = state == "detected"

        direction = _POSITION_TO_DIRECTION.get(output_data.get("best_position"))

        # Only surface a bounding box for a confirmed detection. lights.py's
        # own is_alert_candidate check already requires several confirmed
        # flash cycles, adequate source quality, and a low background score
        # before it calls a track "detected" — gating on that here means
        # every box we emit corresponds to an actual active flashing region,
        # never a bare/weak "possible" candidate. It also means stale boxes
        # can't linger: this method recomputes fresh from this frame's
        # output_data every time, so the instant a track drops out of
        # "detected" the very next CameraReading has boundingBox=None again.
        bbox = output_data.get("best_bbox")
        bounding_box = (
            BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
            if detected and bbox
            else None
        )

        # The detector only confirms *that* a flashing blue light is present,
        # not which kind of emergency vehicle it belongs to — that
        # classification is future TensorFlow-model territory.
        return CameraReading(
            cameraDetected=detected,
            vehicleConfidence=output_data["visual_confidence"],
            vehicleType=VehicleType.UNKNOWN,
            direction=direction,
            boundingBox=bounding_box,
            frameWidth=lights.FRAME_WIDTH,
            frameHeight=lights.FRAME_HEIGHT,
        )

    def _handle_debug(self, output_data: dict, frame, display, blue_mask, blue_change_mask) -> None:
        """OPENCV_DEBUG-only side channel — never touches _latest_reading.

        Always refreshes the live 4-way composite (so /debug-stream stays a
        continuous feed even on frames with no candidate). Logging and saved
        evidence frames only happen on frames where the detector found *some*
        candidate blob (best_bbox is set) — this is deliberately broader than
        "detected" (it includes "possible" and weak/background-penalized
        candidates too), because the point of this mode is to see everything
        the tracker considered, not just what survived its own filtering.
        """
        composite = _build_debug_composite(frame, display, blue_mask, blue_change_mask)
        encoded_ok, buffer = cv2.imencode(".jpg", composite, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        with self._lock:
            self._latest_debug_frame_jpeg = buffer.tobytes() if encoded_ok else None

        bbox = output_data.get("best_bbox")
        if bbox is None:
            return

        # "brightness_score" and "blue_area_score" aren't literal field names
        # in lights.py's return dict — it doesn't expose one. These are the
        # closest existing numbers it does return: best_source_quality is
        # already its own blend of brightness+size+shape into a single
        # 0-1 "is this a real light" score, and blue_background_ratio /
        # best_background_score describe how much of the frame (and the area
        # around the blob) reads as diffuse blue vs. an isolated source.
        # Nothing here is invented or requires touching lights.py.
        logger.info(
            "DETECTION frame=%d state=%s confidence=%.3f bbox=%s position=%s "
            "flash_count=%d brightness_score=%.3f blue_area_score=%.3f "
            "background_score=%.3f reflection=%s scene=%s",
            output_data["frame_index"],
            output_data["visual_state"],
            output_data["visual_confidence"],
            bbox,
            output_data.get("best_position"),
            output_data["confirmed_flash_count"],
            output_data["best_source_quality"],
            output_data["blue_background_ratio"],
            output_data["best_background_score"],
            output_data["best_is_reflection"],
            output_data["scene_mode"],
        )

        self._save_debug_frames(
            output_data["frame_index"],
            output_data["visual_state"],
            frame,
            display,
            blue_mask,
            blue_change_mask,
        )

    def _save_debug_frames(self, frame_index, state, frame, display, blue_mask, blue_change_mask) -> None:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        stem = f"frame_{frame_index:06d}_{state}"
        saved: List[Path] = []
        for image, suffix in (
            (frame, "original"),
            (display, "display"),
            (blue_mask, "blue_mask"),
            (blue_change_mask, "blue_change_mask"),
        ):
            path = _DEBUG_DIR / f"{stem}_{suffix}.jpg"
            cv2.imwrite(str(path), image)
            saved.append(path)

        self._debug_saved_frame_sets.append(saved)
        if len(self._debug_saved_frame_sets) > _DEBUG_MAX_SAVED_FRAME_SETS:
            for stale_path in self._debug_saved_frame_sets.popleft():
                stale_path.unlink(missing_ok=True)
