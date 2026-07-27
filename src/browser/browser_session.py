"""
Browser session models for asynchronous Playwright automation.

This module contains the state container used by the browser operating
system. A session is intentionally small: it owns references to the
Playwright browser/context objects, tracks active pages, and records
lifecycle state without knowing anything about the work being performed
inside the browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from playwright.async_api import Browser, BrowserContext, Page

from src.exceptions import BrowserError


class BrowserSessionState(str, Enum):
    """
    Lifecycle states for a browser session.

    States are explicit so higher-level automation can make conservative
    decisions about whether the browser is ready for work, shutting down,
    stopped, or suspected to have crashed.
    """

    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CRASHED = "crashed"


@dataclass(slots=True)
class BrowserSession:
    """
    Represents one active browser automation session.

    The session stores the Playwright browser object when one is available,
    the active browser context, all known open pages, the current active
    page, lifecycle state, and caller-defined metadata. Persistent Chrome
    contexts may not expose a separate ``Browser`` instance, so ``browser``
    is optional while ``context`` is always required.

    Attributes:
        browser: Playwright browser instance, if exposed by the context.
        context: Playwright browser context used for all pages in the session.
        session_id: Stable identifier for this session.
        state: Current lifecycle state.
        metadata: Free-form session metadata for diagnostics.
        owns_browser: Whether this session owns the browser process. Launched
            sessions own their browser and close it on shutdown. Attached
            sessions (CDP) do not own the browser; closing them disconnects
            without touching the user's tabs or contexts.
        active_pages: Pages currently known to be open.
        active_page: Page considered active by the automation layer.
        created_at: UTC timestamp when the session was created.
        updated_at: UTC timestamp when the session state last changed.
    """

    browser: Optional[Browser]
    context: BrowserContext
    session_id: str
    state: BrowserSessionState = BrowserSessionState.INITIALIZING
    metadata: dict[str, Any] = field(default_factory=dict)
    owns_browser: bool = True
    active_pages: list[Page] = field(default_factory=list)
    active_page: Optional[Page] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        """Initialize page tracking from the Playwright context."""
        self.sync_pages()
        if self.active_page is None and self.active_pages:
            self.active_page = self.active_pages[0]

    @property
    def page_count(self) -> int:
        """
        Return the number of currently open pages.

        Returns:
            Number of pages that are known and not closed.
        """
        self.sync_pages()
        return len(self.active_pages)

    def mark_running(self) -> None:
        """Mark the session as ready for browser automation."""
        self._set_state(BrowserSessionState.RUNNING)

    def mark_stopping(self) -> None:
        """Mark the session as shutting down."""
        self._set_state(BrowserSessionState.STOPPING)

    def mark_stopped(self) -> None:
        """Mark the session as fully stopped."""
        self._set_state(BrowserSessionState.STOPPED)

    def mark_crashed(self, error: Optional[str] = None) -> None:
        """
        Mark the session as crashed and record the optional error message.

        Args:
            error: Optional diagnostic message from Playwright or observer.
        """
        if error:
            self.metadata["last_error"] = error
        self._set_state(BrowserSessionState.CRASHED)

    def sync_pages(self) -> list[Page]:
        """
        Synchronize tracked pages with the Playwright context.

        Returns:
            List of currently open pages.

        Raises:
            BrowserError: If Playwright cannot read the context pages.
        """
        try:
            self.active_pages = [
                page for page in self.context.pages if not page.is_closed()
            ]
        except Exception as exc:
            raise BrowserError(
                f"Unable to synchronize browser pages: {exc}",
                code="BROWSER_SESSION_SYNC_FAILED",
            ) from exc

        if self.active_page is not None and self.active_page.is_closed():
            self.active_page = None

        if self.active_page is None and self.active_pages:
            self.active_page = self.active_pages[0]

        self._touch()
        return self.active_pages

    def add_page(self, page: Page, make_active: bool = True) -> None:
        """
        Track a page in this session.

        Args:
            page: Playwright page to add.
            make_active: Whether the added page should become active.
        """
        if page not in self.active_pages and not page.is_closed():
            self.active_pages.append(page)

        if make_active:
            self.active_page = page

        self._touch()

    def remove_page(self, page: Page) -> None:
        """
        Remove a page from this session's tracking state.

        Args:
            page: Playwright page to remove.
        """
        self.active_pages = [
            tracked_page for tracked_page in self.active_pages
            if tracked_page is not page and not tracked_page.is_closed()
        ]

        if self.active_page is page:
            self.active_page = (
                self.active_pages[0] if self.active_pages else None
            )

        self._touch()

    def set_active_page(self, page: Page) -> None:
        """
        Set the active page for this session.

        Args:
            page: Page to mark active.

        Raises:
            BrowserError: If the page is closed or not part of the context.
        """
        if page.is_closed():
            raise BrowserError(
                "Cannot activate a closed page",
                code="BROWSER_PAGE_CLOSED",
            )

        self.sync_pages()
        if page not in self.active_pages:
            raise BrowserError(
                "Cannot activate a page outside this browser session",
                code="BROWSER_PAGE_NOT_IN_SESSION",
            )

        self.active_page = page
        self._touch()

    def get_active_page(self) -> Page:
        """
        Return the active page.

        Returns:
            Active Playwright page.

        Raises:
            BrowserError: If no active page is available.
        """
        self.sync_pages()
        if self.active_page is None:
            raise BrowserError(
                "No active page is available in this browser session",
                code="BROWSER_NO_ACTIVE_PAGE",
            )
        return self.active_page

    def is_browser_alive(self) -> bool:
        """
        Check whether the browser context is still usable.

        Returns:
            True when Playwright can inspect the browser/context; otherwise
            False.
        """
        try:
            if self.browser is not None and not self.browser.is_connected():
                return False
            self.sync_pages()
            return self.state not in {
                BrowserSessionState.STOPPING,
                BrowserSessionState.STOPPED,
                BrowserSessionState.CRASHED,
            }
        except Exception:
            return False

    async def close(self) -> None:
        """
        Close the session's context and browser gracefully.

        Launched sessions close their context and browser. Attached (CDP)
        sessions only disconnect, leaving the user's browser untouched.

        Raises:
            BrowserError: If Playwright fails during shutdown.
        """
        if self.state == BrowserSessionState.STOPPED:
            return

        self.mark_stopping()

        try:
            if self.owns_browser:
                await self.context.close()
                if self.browser is not None and self.browser.is_connected():
                    await self.browser.close()
            elif self.browser is not None and self.browser.is_connected():
                # Browser.close() on a browser obtained via connect_over_cdp
                # only disconnects; the user's browser keeps running.
                await self.browser.close()
        except Exception as exc:
            self.mark_crashed(str(exc))
            raise BrowserError(
                f"Failed to close browser session: {exc}",
                code="BROWSER_SESSION_CLOSE_FAILED",
            ) from exc
        finally:
            self.active_pages = []
            self.active_page = None
            if self.state != BrowserSessionState.CRASHED:
                self.mark_stopped()

    def _set_state(self, state: BrowserSessionState) -> None:
        """Set session state and update the modification timestamp."""
        self.state = state
        self._touch()

    def _touch(self) -> None:
        """Refresh the session modification timestamp."""
        self.updated_at = datetime.now(timezone.utc)


__all__ = [
    "BrowserSession",
    "BrowserSessionState",
]
