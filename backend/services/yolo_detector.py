"""Ultralytics YOLO vehicle detection, loaded once and reused for every frame.

Mirrors the process-wide singleton pattern services/tensorflow_audio.py uses
for YAMNet: the model is heavy to load (~1-2s) and must never be constructed
per-frame or per-request, so it's loaded lazily on first use and cached at
module scope. `ultralytics`/`torch` are only imported when USE_YOLO=true (see
config.settings.use_yolo) — a desktop/CI run with USE_YOLO=false never pays
that import cost, same reasoning as tensorflow_hub's lazy import there.

This module only knows about "vehicles" as YOLO's pretrained COCO classes
(car/truck/bus/motorcycle) — it has no concept of emergency vehicles, police
liveries, or lights. Associating vehicle boxes with lights.py's flashing-blue
evidence is services/vision_fusion.py's job, not this one's.
"""

import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]

# Where a bare model name (e.g. "yolo11n.pt") is auto-downloaded to and
# loaded from — keeps weights out of the repo root regardless of the
# process's cwd, and out of git (backend/.cache/ is already gitignored,
# same as .tfhub_cache/ for YAMNet).
_YOLO_CACHE_DIR = _BACKEND_DIR / ".cache" / "yolo"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VehicleDetection:
    """One detected vehicle in the frame YOLO was given.

    bbox is (x, y, width, height) in that same frame's pixel space — callers
    resize their frame to a known size *before* calling detect_vehicles() if
    they need the coordinates to line up with something else (services/
    camera.py resizes to lights.FRAME_WIDTH/HEIGHT first, for exactly this
    reason).
    """

    label: str
    confidence: float
    bbox: tuple  # (x, y, width, height), all ints


def _resolve_model_path(model_name: str) -> str:
    """A path that already exists (absolute, or relative to the repo root/
    backend dir) is used as-is. Otherwise treat it as a bare weights filename
    Ultralytics knows how to fetch, and pin the download location to
    _YOLO_CACHE_DIR instead of leaving it to land in cwd."""
    candidate = Path(model_name)
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate)

    repo_root = _BACKEND_DIR.parent
    for base in (repo_root, _BACKEND_DIR):
        resolved = base / model_name
        if resolved.is_file():
            return str(resolved)

    _YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(_YOLO_CACHE_DIR / candidate.name)


class YoloVehicleDetector:
    """Loads a YOLO model exactly once and runs inference on demand.

    Not thread-safe for concurrent predict() calls from multiple threads at
    once, but services/camera.py only ever calls it from its single
    background frame-loop thread, so that's not a concern here.
    """

    def __init__(self, model_name: str, confidence_threshold: float, vehicle_classes: Sequence[str]):
        model_path = _resolve_model_path(model_name)

        logger.info("Loading YOLO model (%s)...", model_path)
        import time

        t0 = time.monotonic()

        # Imported here, not at module level: torch/ultralytics are a heavy,
        # optional dependency only needed when USE_YOLO=true.
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # YOLO's pretrained COCO names are index -> label; invert + filter
        # down to just the labels this deployment cares about ("car",
        # "truck", ...) so predict() can pass class indices straight through
        # instead of filtering every result by string label afterwards.
        name_to_index = {name: index for index, name in self._model.names.items()}
        self._vehicle_class_indices = [
            name_to_index[name] for name in vehicle_classes if name in name_to_index
        ]
        unknown_classes = [name for name in vehicle_classes if name not in name_to_index]
        if unknown_classes:
            logger.warning(
                "YOLO_VEHICLE_CLASSES contains names this model doesn't have: %s (known: %s)",
                unknown_classes,
                sorted(name_to_index),
            )

        logger.info(
            "YOLO model loaded in %.2fs (vehicle classes=%s)",
            time.monotonic() - t0,
            vehicle_classes,
        )

    def detect_vehicles(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Runs one inference pass and returns only the configured vehicle
        classes above the confidence threshold. Never raises — a bad frame
        or an inference hiccup should degrade to "no vehicles this frame",
        not take down the camera loop it's called from."""
        if not self._vehicle_class_indices:
            return []

        try:
            results = self._model.predict(
                frame,
                classes=self._vehicle_class_indices,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception:
            logger.exception("YOLO inference failed on a frame; skipping")
            return []

        detections: List[VehicleDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box_xyxy, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                x1, y1, x2, y2 = box_xyxy
                label = self._model.names[int(cls)]
                detections.append(
                    VehicleDetection(
                        label=label,
                        confidence=float(conf),
                        bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    )
                )

        return detections


_shared_lock = threading.Lock()
_shared_detector: "YoloVehicleDetector | None" = None


def get_shared_detector(
    model_name: str, confidence_threshold: float, vehicle_classes: Sequence[str]
) -> YoloVehicleDetector:
    """Process-wide YOLO singleton, loaded once and reused by every caller —
    same pattern as services/tensorflow_audio.py's _get_shared_classifier()."""
    global _shared_detector
    with _shared_lock:
        if _shared_detector is None:
            _shared_detector = YoloVehicleDetector(model_name, confidence_threshold, vehicle_classes)
        else:
            _shared_detector.confidence_threshold = confidence_threshold
        return _shared_detector
