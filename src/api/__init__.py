"""
API package: FastAPI application exposing research functionality.

Run locally with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

from .app import AppComponents, app, create_app

__all__ = [
    "AppComponents",
    "app",
    "create_app",
]
