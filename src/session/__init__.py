"""
Session package: persistent multi-provider research sessions.

BrowserSessionManager owns the long-lived browser and one live tab per AI
provider, keeping conversations alive across requests and repairing failed
tabs individually.
"""

from .browser_session_manager import (
    BrowserSessionManager,
    ProviderStatus,
    SessionStats,
)

__all__ = [
    "BrowserSessionManager",
    "ProviderStatus",
    "SessionStats",
]
