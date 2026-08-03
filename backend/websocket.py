"""WebSocket connection management and the detection broadcast loop.

Rather than having each client connection run its own independent simulation
loop, a single background task ticks once per second, generates one fused
DetectionEvent, and fans it out to every connected client. This keeps all
clients in sync and mirrors how a real single-sensor-rig backend would behave.
"""

import asyncio
import logging
from typing import Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from config import settings
from models import DetectionEvent
from services.audio import AudioDetectionService
from services.camera import CameraDetectionService
from services.fusion import FusionService

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Tracks active WebSocket clients and broadcasts messages to all of them."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("Client connected (active=%d)", self.connection_count)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("Client disconnected (active=%d)", self.connection_count)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            targets = list(self._connections)

        dead: Set[WebSocket] = set()
        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception:
                dead.add(connection)

        if dead:
            async with self._lock:
                self._connections -= dead


class DetectionBroadcaster:
    """Periodically generates a simulated DetectionEvent and broadcasts it."""

    def __init__(
        self,
        manager: ConnectionManager,
        fusion_service: FusionService,
        interval_seconds: float,
    ):
        self._manager = manager
        self._fusion_service = fusion_service
        self._interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self.latest_event: Optional[DetectionEvent] = None

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            logger.info("Detection broadcaster started (interval=%.2fs)", self._interval_seconds)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("Detection broadcaster stopped")

    async def _run(self) -> None:
        while True:
            self.latest_event = self._fusion_service.generate_event()
            await self._manager.broadcast(self.latest_event.model_dump(mode="json"))
            await asyncio.sleep(self._interval_seconds)


manager = ConnectionManager()

fusion_service = FusionService(
    audio_service=AudioDetectionService(settings.audio_detection_probability),
    camera_service=CameraDetectionService(settings.camera_detection_probability),
)

broadcaster = DetectionBroadcaster(
    manager=manager,
    fusion_service=fusion_service,
    interval_seconds=settings.broadcast_interval_seconds,
)


async def detection_websocket_endpoint(websocket: WebSocket) -> None:
    """Handles a single client connection on /ws/detection.

    On connect, immediately sends the most recent event (if any) so the
    client isn't left waiting up to a full interval for its first update.
    Afterwards it just listens for disconnects; all data flows out via the
    shared DetectionBroadcaster.
    """
    await manager.connect(websocket)
    try:
        if broadcaster.latest_event is not None:
            await websocket.send_json(broadcaster.latest_event.model_dump(mode="json"))

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
