"""Simulated camera vehicle-detection service.

Stands in for a future OpenCV/vision pipeline. Produces randomized-but-plausible
readings so the rest of the system can be built and tested independently of the
real computer-vision model.
"""

import random

from models import CameraReading, Direction, VehicleType

_VEHICLE_TYPES = (VehicleType.AMBULANCE, VehicleType.FIRE_TRUCK, VehicleType.POLICE)
_DIRECTIONS = tuple(Direction)


class CameraDetectionService:
    def __init__(self, detection_probability: float = 0.5):
        self._detection_probability = detection_probability

    def generate_reading(self) -> CameraReading:
        detected = random.random() < self._detection_probability

        if detected:
            return CameraReading(
                cameraDetected=True,
                vehicleConfidence=round(random.uniform(0.60, 0.99), 2),
                vehicleType=random.choice(_VEHICLE_TYPES),
                direction=random.choice(_DIRECTIONS),
            )

        return CameraReading(
            cameraDetected=False,
            vehicleConfidence=round(random.uniform(0.0, 0.40), 2),
            vehicleType=VehicleType.UNKNOWN,
            direction=None,
        )
