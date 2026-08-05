"""Shared data models exchanged between services, routes, and the WebSocket layer."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SirenType(str, Enum):
    AMBULANCE = "Ambulance"
    FIRE_TRUCK = "Fire Truck"
    POLICE = "Police"
    NONE = "None"


class VehicleType(str, Enum):
    AMBULANCE = "Ambulance"
    FIRE_TRUCK = "Fire Truck"
    POLICE = "Police"
    UNKNOWN = "Unknown"


class Direction(str, Enum):
    FRONT = "Front"
    FRONT_LEFT = "Front Left"
    FRONT_RIGHT = "Front Right"
    REAR = "Rear"
    REAR_LEFT = "Rear Left"
    REAR_RIGHT = "Rear Right"
    LEFT = "Left"
    RIGHT = "Right"


class AlertLevel(str, Enum):
    CRITICAL = "Critical"
    WARNING = "Warning"
    INFO = "Info"
    NONE = "None"


class AudioReading(BaseModel):
    """Raw output of the (simulated) audio siren-detection service."""

    audioDetected: bool
    audioConfidence: float = Field(ge=0.0, le=1.0)
    sirenType: SirenType


class BoundingBox(BaseModel):
    """Pixel-space box in the camera detector's working frame (640x480)."""

    x: int
    y: int
    width: int
    height: int


class CameraReading(BaseModel):
    """Raw output of the camera vehicle-detection service."""

    cameraDetected: bool
    vehicleConfidence: float = Field(ge=0.0, le=1.0)
    vehicleType: VehicleType
    direction: Optional[Direction] = None
    boundingBox: Optional[BoundingBox] = None
    # Pixel dimensions of the frame boundingBox's coordinates are expressed
    # in, so consumers can convert to percentages of their own render size.
    frameWidth: Optional[int] = None
    frameHeight: Optional[int] = None


class DetectionEvent(BaseModel):
    """Fused audio + camera detection result broadcast to WebSocket clients."""

    audioDetected: bool
    audioConfidence: float = Field(ge=0.0, le=1.0)
    sirenType: SirenType
    cameraDetected: bool
    vehicleConfidence: float = Field(ge=0.0, le=1.0)
    vehicleType: VehicleType
    direction: Optional[Direction] = None
    boundingBox: Optional[BoundingBox] = None
    frameWidth: Optional[int] = None
    frameHeight: Optional[int] = None
    alertLevel: AlertLevel
    timestamp: str
