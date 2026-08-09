"""Demo-session endpoints.

Serves whatever video file CAMERA_SOURCE actually resolves to (the single
source for both OpenCV's frames and YAMNet's audio — see services/camera.py
and services/tensorflow_audio.py) and resets both services' playback
position together, so a press of "Start Demo" on the frontend brings video,
vision, and audio detection back to (approximately) the same starting point.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from websocket import audio_service, camera_service

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/video")
async def get_demo_video() -> FileResponse:
    """Serves the raw MP4 file (with Range support, via Starlette's
    FileResponse) for direct playback in an HTML5 <video> element.

    Reads camera_service.source_path rather than a hardcoded backend/demo/
    path — that property reflects the actual, currently-open CAMERA_SOURCE
    (including any auto-fallback), so this always serves exactly the file
    OpenCV is reading frames from and services/tensorflow_audio.py extracts
    audio from. A hardcoded path would silently drift out of sync with
    CAMERA_SOURCE the moment someone points it at a different file.
    """
    source_path = camera_service.source_path
    if source_path is None or not source_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "No demo video file available — camera_service is reading from a live "
                "source (e.g. CAMERA_SOURCE=0) rather than a video file."
            ),
        )
    return FileResponse(
        source_path,
        media_type="video/mp4",
        # This URL's actual content changes whenever CAMERA_SOURCE points at
        # a different file, but the URL itself never does — without this,
        # browsers cache the video by URL (FileResponse otherwise sends only
        # ETag/Last-Modified, which some caches treat as heuristically
        # freshness-cacheable) and keep serving stale bytes from a
        # previously configured video after a restart with a new source.
        headers={"Cache-Control": "no-store"},
    )


@router.post("/start")
async def start_demo() -> dict:
    """Resets the camera and audio services' playback position to the
    start. The frontend calls this at the same moment it seeks its own
    <video> element back to t=0 and plays — see the Start Demo / Restart
    controls in CameraPanel.tsx.
    """
    camera_service.restart()
    audio_service.restart()
    return {"status": "started", "videoUrl": "/api/demo/video"}
