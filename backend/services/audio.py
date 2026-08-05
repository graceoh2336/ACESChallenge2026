"""Simulated audio siren-detection service.

Stands in for a future TensorFlow audio-classification pipeline. Produces
randomized-but-plausible readings so the rest of the system (fusion logic,
WebSocket streaming, frontend) can be built and tested independently of the
real ML model.
"""

import random

from models import AudioReading, SirenType

_SIREN_TYPES = (SirenType.AMBULANCE, SirenType.FIRE_TRUCK, SirenType.POLICE)


class AudioDetectionService:
    def __init__(self, detection_probability: float = 0.55):
        self._detection_probability = detection_probability

    def generate_reading(self) -> AudioReading:
        detected = random.random() < self._detection_probability

        if detected:
            return AudioReading(
                audioDetected=True,
                audioConfidence=round(random.uniform(0.60, 0.99), 2),
                sirenType=random.choice(_SIREN_TYPES),
            )

        return AudioReading(
            audioDetected=False,
            audioConfidence=round(random.uniform(0.0, 0.40), 2),
            sirenType=SirenType.NONE,
        )
