"""Vehicle-first visual emergency detection: combines YOLO vehicle boxes with
lights.py's flashing-blue-light evidence so a flashing light only counts as
emergency-vehicle evidence when it's actually on/near a detected vehicle.

This is deliberately a thin layer on top of two existing, working detectors —
it does not re-implement vehicle detection (yolo_detector.py) or flash-timing
analysis (lights.py's own confirmed_flash_count/track confidence). All it
adds is:

1. Spatial association: does lights.py's best flashing-blob bbox this frame
   fall inside (a padded) YOLO vehicle box?
2. Temporal stability *of that association* (Phase 4): a light that overlaps
   a vehicle for one frame is noise; one that keeps doing it across a rolling
   window is a real vehicle-mounted light.
3. A single explainable combined confidence + one of five internal visual
   states (Phase 6) for CameraDetectionService to translate into a
   CameraReading.

VisualState.CONFIRMED is the only state that should ever set
CameraReading.cameraDetected=True — every other state existed in one form or
another before this upgrade (lights.py's own "clear"/"possible"/"detected"
already gated cameraDetected the same way), so fusion.py's existing
audio+camera alert-level policy needs no changes.
"""

from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Deque, List, Optional, Sequence

from services.yolo_detector import VehicleDetection

BBox = tuple  # (x, y, width, height)


class VisualState(str, Enum):
    NO_VEHICLE = "no_vehicle"
    VEHICLE_DETECTED = "vehicle_detected"
    POSSIBLE_EMERGENCY_VEHICLE = "possible_emergency_vehicle"
    EMERGENCY_VEHICLE_CONFIRMED = "emergency_vehicle_confirmed"


@dataclass(frozen=True)
class VisionResult:
    """Everything CameraDetectionService needs to build a CameraReading, plus
    the evidence breakdown OPENCV_DEBUG wants to display (Phase 10/11)."""

    state: VisualState
    confidence: float  # combined visual emergency confidence, 0-1
    display_bbox: Optional[BBox]  # what the dashboard should draw — vehicle box when one is associated
    vehicle_bbox: Optional[BBox]
    vehicle_label: Optional[str]
    vehicle_confidence: float
    light_bbox: Optional[BBox]
    light_confidence: float
    associated: bool  # light evidence overlapped a vehicle *this frame*
    association_ratio: float  # fraction of the recent window that was associated
    temporal_confirmation: str  # e.g. "4/15" — human-readable for debug overlay


def _pad_box(box: BBox, padding_ratio: float) -> BBox:
    x, y, w, h = box
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    return (x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y)


def _point_in_box(px: float, py: float, box: BBox) -> bool:
    x, y, w, h = box
    return x <= px <= x + w and y <= py <= y + h


def _box_center(box: BBox):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _light_associated_with_vehicle(light_box: BBox, vehicle_box: BBox, padding_ratio: float) -> bool:
    """A light "belongs" to a vehicle if either its centre sits inside the
    vehicle's padded ROI, or the two boxes plainly overlap — covers both a
    small light blob nested inside a loose vehicle box, and a light blob
    that's larger than (or offset from) a tight vehicle box, e.g. a wide
    lightbar glow next to a car detected end-on."""
    padded_vehicle = _pad_box(vehicle_box, padding_ratio)
    light_cx, light_cy = _box_center(light_box)
    return _point_in_box(light_cx, light_cy, padded_vehicle) or _boxes_overlap(light_box, padded_vehicle)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class VehicleLightFusion:
    """Holds the rolling association-stability window across frames. One
    instance per detection session — services/camera.py creates a fresh one
    alongside its fresh lights.VisualEmergencyDetector() on every demo
    restart, so stale association history from a previous playback position
    can't leak into a new session (mirrors lights.py's own per-session
    tracker reset)."""

    def __init__(
        self,
        roi_padding: float,
        association_window: int,
        association_ratio_required: float,
    ):
        self._roi_padding = roi_padding
        self._association_ratio_required = association_ratio_required
        self._association_history: Deque[bool] = deque(maxlen=max(1, association_window))
        # A real flashing light is only "on" part of the time by nature — a
        # window this small could otherwise reach a (technically correct)
        # 100% ratio off a single lucky sample and get treated as "stable".
        # Confirmed status additionally requires at least this many *sampled*
        # (light-candidate) frames, not just a high ratio over very few.
        self._min_samples_for_confirmation = max(3, association_window // 3)

    @staticmethod
    def _best_vehicle(vehicles: Sequence[VehicleDetection]) -> Optional[VehicleDetection]:
        if not vehicles:
            return None
        return max(vehicles, key=lambda v: v.confidence)

    def _best_associated_vehicle(
        self, light_box: BBox, vehicles: Sequence[VehicleDetection]
    ) -> Optional[VehicleDetection]:
        associated = [
            vehicle
            for vehicle in vehicles
            if _light_associated_with_vehicle(light_box, vehicle.bbox, self._roi_padding)
        ]
        return self._best_vehicle(associated)

    def update(
        self,
        light_state: str,
        light_confidence: float,
        light_bbox: Optional[BBox],
        vehicles: Sequence[VehicleDetection],
    ) -> VisionResult:
        """One call per analyzed video frame.

        light_state/light_confidence/light_bbox come straight from
        lights.py's own output_data ("clear"/"possible"/"detected",
        visual_confidence, best_bbox) — this function never second-guesses
        lights.py's own flash-quality judgement, only *where* that judgement
        is allowed to matter.
        """
        light_is_candidate = light_state in ("possible", "detected") and light_bbox is not None

        associated_vehicle: Optional[VehicleDetection] = None
        if light_is_candidate and vehicles:
            associated_vehicle = self._best_associated_vehicle(light_bbox, vehicles)

        associated_this_frame = associated_vehicle is not None

        # Only sample the association-stability window on frames where the
        # light actually gave us something to test. A flashing light is off
        # roughly half the time by its very nature (that's what "flashing"
        # means) — counting those off-phase frames as "not associated" would
        # punish a perfectly genuine, consistently vehicle-mounted light for
        # blinking the way it's supposed to. This way the ratio answers "of
        # the times this light was on, was it on the vehicle?", which is the
        # question that actually matters and is robust to flash duty cycle,
        # source frame rate, and lights.py's own detection cadence.
        if light_is_candidate:
            self._association_history.append(associated_this_frame)

        sampled = len(self._association_history)
        association_ratio = (sum(self._association_history) / sampled) if sampled else 0.0
        temporal_confirmation = f"{sum(self._association_history)}/{sampled}"

        best_vehicle = associated_vehicle or self._best_vehicle(vehicles)

        # --- No vehicle in frame at all -----------------------------------
        if best_vehicle is None:
            return VisionResult(
                state=VisualState.NO_VEHICLE,
                confidence=0.0,
                display_bbox=None,
                vehicle_bbox=None,
                vehicle_label=None,
                vehicle_confidence=0.0,
                light_bbox=light_bbox,
                light_confidence=light_confidence,
                associated=False,
                association_ratio=association_ratio,
                temporal_confirmation=temporal_confirmation,
            )

        # --- Vehicle present, but no qualifying/associated light ----------
        if not associated_this_frame:
            # A light candidate exists somewhere in the frame but not on any
            # detected vehicle (a traffic light, a reflection, distant
            # scenery) — explicitly rejected as emergency evidence per Phase
            # 3/12, though its (attenuated) confidence is still surfaced for
            # debugging rather than silently dropped to exactly zero.
            residual_confidence = _clamp01(light_confidence * 0.25) if light_is_candidate else 0.0
            return VisionResult(
                state=VisualState.VEHICLE_DETECTED,
                confidence=residual_confidence,
                display_bbox=best_vehicle.bbox,
                vehicle_bbox=best_vehicle.bbox,
                vehicle_label=best_vehicle.label,
                vehicle_confidence=best_vehicle.confidence,
                light_bbox=light_bbox,
                light_confidence=light_confidence,
                associated=False,
                association_ratio=association_ratio,
                temporal_confirmation=temporal_confirmation,
            )

        # --- Vehicle + associated light evidence ---------------------------
        combined_confidence = _clamp01(
            0.70 * light_confidence
            + 0.15 * associated_vehicle.confidence
            + 0.15 * association_ratio
        )

        stable_enough = (
            sampled >= self._min_samples_for_confirmation
            and association_ratio >= self._association_ratio_required
        )
        if light_state == "detected" and stable_enough:
            state = VisualState.EMERGENCY_VEHICLE_CONFIRMED
        else:
            state = VisualState.POSSIBLE_EMERGENCY_VEHICLE

        return VisionResult(
            state=state,
            confidence=combined_confidence,
            display_bbox=associated_vehicle.bbox,
            vehicle_bbox=associated_vehicle.bbox,
            vehicle_label=associated_vehicle.label,
            vehicle_confidence=associated_vehicle.confidence,
            light_bbox=light_bbox,
            light_confidence=light_confidence,
            associated=True,
            association_ratio=association_ratio,
            temporal_confirmation=temporal_confirmation,
        )
