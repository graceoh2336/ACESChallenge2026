"""Aggregates all REST routers into a single APIRouter for main.py to mount."""

from fastapi import APIRouter

from routes.camera import router as camera_router
from routes.detection import router as detection_router
from routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(detection_router)
api_router.include_router(camera_router)
