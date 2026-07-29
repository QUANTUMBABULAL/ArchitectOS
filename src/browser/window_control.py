"""
Visibility control for the automation browser's OS windows.

Chrome is ArchitectOS's execution engine, not its interface: Mission
Control is what the user watches. This module hides and restores the
automation Chrome windows — automation continues untouched, because
Playwright drives pages over CDP and never needs the window on screen.

Targeting is deliberately narrow. Only top-level windows of Chrome
processes launched with THIS profile's ``--user-data-dir`` are affected,
so the user's personal Chrome is never hidden. Everything degrades to a
no-op off Windows or when psutil is unavailable.
"""

from __future__ import annotations

import sys
from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)

_SW_HIDE = 0
_SW_SHOWNOACTIVATE = 4
_SW_RESTORE = 9
_CHROME_WINDOW_CLASS = "Chrome_WidgetWin_1"


def _automation_pids(user_data_dir: str) -> set[int]:
    """
    Find Chrome processes launched with the automation profile.

    Args:
        user_data_dir: Profile directory passed at launch; its presence in
            a process command line is what marks the process as ours.

    Returns:
        Matching process IDs (empty on any failure).
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil is not installed; cannot control windows")
        return set()

    marker = str(user_data_dir).lower()
    pids: set[int] = set()
    try:
        for process in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (process.info.get("name") or "").lower()
                if "chrome" not in name and "chromium" not in name:
                    continue
                cmdline = " ".join(process.info.get("cmdline") or []).lower()
                if marker and marker in cmdline:
                    pids.add(process.pid)
            except Exception:
                continue
    except Exception as exc:
        logger.warning("Process scan failed: %s", exc)
    return pids


def _find_windows(pids: set[int]) -> list[int]:
    """
    Enumerate top-level Chrome windows belonging to the given processes.

    Args:
        pids: Browser process IDs.

    Returns:
        Window handles (empty off Windows or on failure).
    """
    if sys.platform != "win32" or not pids:
        return []

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    handles: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_callback(hwnd: int, _lparam: int) -> bool:
        try:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value not in pids:
                return True

            class_buffer = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, class_buffer, 64)
            if class_buffer.value != _CHROME_WINDOW_CLASS:
                return True

            # Owned windows (tooltips, menus) are skipped: hiding them is
            # pointless and showing them out of context looks broken.
            if user32.GetWindow(hwnd, 4):  # GW_OWNER
                return True

            handles.append(hwnd)
        except Exception:
            pass
        return True

    try:
        user32.EnumWindows(enum_callback, 0)
    except Exception as exc:
        logger.warning("Window enumeration failed: %s", exc)
    return handles


def _apply(user_data_dir: Optional[str], command: int) -> int:
    """
    Apply a ShowWindow command to every automation Chrome window.

    Args:
        user_data_dir: Automation profile directory.
        command: Win32 ShowWindow command.

    Returns:
        Number of windows affected.
    """
    if sys.platform != "win32":
        logger.info("Window control is only implemented on Windows")
        return 0
    if not user_data_dir:
        return 0

    handles = _find_windows(_automation_pids(user_data_dir))
    if not handles:
        return 0

    import ctypes

    affected = 0
    for hwnd in handles:
        try:
            ctypes.windll.user32.ShowWindow(hwnd, command)
            affected += 1
        except Exception:
            continue
    return affected


def hide_browser_windows(user_data_dir: Optional[str]) -> int:
    """
    Hide every automation Chrome window.

    Automation continues: CDP does not require a visible window.

    Args:
        user_data_dir: Automation profile directory identifying our
            Chrome processes.

    Returns:
        Number of windows hidden.
    """
    count = _apply(user_data_dir, _SW_HIDE)
    if count:
        logger.info("Hid %d automation browser window(s)", count)
    return count


def show_browser_windows(user_data_dir: Optional[str]) -> int:
    """
    Restore every automation Chrome window.

    Windows are restored without stealing focus, then the last one is
    brought forward so the user can actually find it.

    Args:
        user_data_dir: Automation profile directory identifying our
            Chrome processes.

    Returns:
        Number of windows restored.
    """
    if sys.platform != "win32" or not user_data_dir:
        return 0

    handles = _find_windows(_automation_pids(user_data_dir))
    if not handles:
        return 0

    import ctypes

    user32 = ctypes.windll.user32
    affected = 0
    for hwnd in handles:
        try:
            user32.ShowWindow(hwnd, _SW_SHOWNOACTIVATE)
            user32.ShowWindow(hwnd, _SW_RESTORE)
            affected += 1
        except Exception:
            continue

    try:
        user32.SetForegroundWindow(handles[-1])
    except Exception:
        pass

    if affected:
        logger.info("Restored %d automation browser window(s)", affected)
    return affected


__all__ = [
    "hide_browser_windows",
    "show_browser_windows",
]
