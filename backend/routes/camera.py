"""MJPEG live-preview stream for the camera panel.

Serves the plain frames CameraDetectionService caches (see
services/camera.py) — no detector-drawn boxes/text, since the frontend
already renders its own bounding-box overlay on top; this just supplies the
pixels underneath it. Independent of, and unrelated to, the WebSocket
detection stream — a client can watch the video without ever opening a
WebSocket connection, and vice versa.
"""

import asyncio
import logging
from typing import Callable, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from websocket import camera_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/camera", tags=["camera"])

_BOUNDARY = "frame"
_STREAM_FPS = 10


async def _mjpeg_frames(get_frame: Callable[[], Optional[bytes]]):
    """Shared MJPEG generator — both /stream and /debug-stream just plug in
    a different frame source (CameraDetectionService already owns the only
    encode of each; this only reads the cached bytes) rather than each
    duplicating this loop."""
    interval = 1 / _STREAM_FPS
    while True:
        frame = get_frame()
        if frame is not None:
            yield (
                b"--" + _BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame + b"\r\n"
            )
        await asyncio.sleep(interval)


@router.get("/stream")
async def camera_stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_frames(camera_service.get_latest_frame_jpeg),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )


@router.get("/debug-stream")
async def camera_debug_stream():
    """2x2 grid: original frame, detector's own annotated display, blue
    mask, blue-change mask — only populated while OPENCV_DEBUG is enabled
    (see services/camera.py's _handle_debug)."""
    if not settings.opencv_debug:
        return JSONResponse(
            status_code=503,
            content={"detail": "OPENCV_DEBUG is not enabled. Set OPENCV_DEBUG=true and restart to use this endpoint."},
        )

    return StreamingResponse(
        _mjpeg_frames(camera_service.get_latest_debug_frame_jpeg),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )
