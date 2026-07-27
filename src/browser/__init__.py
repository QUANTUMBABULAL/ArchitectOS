"""
Reusable browser operating system built on Playwright Async API.

The browser package provides lifecycle management, Chrome profile support,
tab operations, keyboard and mouse controllers, continuous observation, and
generic extraction utilities. It intentionally contains no site-specific or
AI-provider-specific automation logic.
"""

from .browser_connection import BrowserConnection, BrowserConnectionConfig
from .browser_factory import BrowserFactory, BrowserKind, BrowserLaunchConfig
from .browser_manager import BrowserHealth, BrowserManager, BrowserManagerConfig
from .browser_session import BrowserSession, BrowserSessionState
from .chrome_profile import ChromeProfile, ChromeProfileConfig
from .extractor import CodeBlock, Extractor, ExtractorConfig
from .keyboard_controller import KeyboardController, KeyboardControllerConfig
from .launch_diagnostics import (
    LaunchDiagnosis,
    LaunchFailureKind,
    classify_launch_failure,
    detect_handoff,
)
from .mouse_controller import MouseController, MouseControllerConfig, Point
from .profile_lock import LockOwner, ProfileLock
from .observer import (
    DialogAction,
    DialogObservation,
    Observer,
    ObserverConfig,
    ObserverSnapshot,
    PageObservation,
)
from .tab_manager import TabInfo, TabManager, TabTarget

__all__ = [
    "BrowserConnection",
    "BrowserConnectionConfig",
    "BrowserFactory",
    "BrowserHealth",
    "BrowserKind",
    "BrowserLaunchConfig",
    "BrowserManager",
    "BrowserManagerConfig",
    "BrowserSession",
    "BrowserSessionState",
    "ChromeProfile",
    "ChromeProfileConfig",
    "CodeBlock",
    "DialogAction",
    "DialogObservation",
    "Extractor",
    "ExtractorConfig",
    "KeyboardController",
    "KeyboardControllerConfig",
    "LaunchDiagnosis",
    "LaunchFailureKind",
    "LockOwner",
    "ProfileLock",
    "classify_launch_failure",
    "detect_handoff",
    "MouseController",
    "MouseControllerConfig",
    "Observer",
    "ObserverConfig",
    "ObserverSnapshot",
    "PageObservation",
    "Point",
    "TabInfo",
    "TabManager",
    "TabTarget",
]
