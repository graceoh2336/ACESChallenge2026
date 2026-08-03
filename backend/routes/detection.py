"""REST endpoints that complement the /ws/detection WebSocket stream.

Useful for the frontend to fetch the current state on initial page load
(before the WebSocket connects) or for debugging/monitoring outside a
WebSocket client.
"""

from fastapi import APIRouter

from websocket import broadcaster, manager

router = APIRouter(prefix="/api/detection", tags=["detection"])


@router.get("/latest")
async def get_latest_detection() -> dict:
    if broadcaster.latest_event is None:
        return {"message": "No detection events generated yet."}
    return broadcaster.latest_event.model_dump(mode="json")


@router.get("/status")
async def get_status() -> dict:
    return {
        "activeConnections": manager.connection_count,
        "broadcastIntervalSeconds": broadcaster.interval_seconds,
    }
