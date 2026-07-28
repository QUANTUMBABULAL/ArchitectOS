"""
Continuous browser and page observation.

Observer monitors a BrowserSession for liveness, page loading state, network
idle state, timeouts, crashes, and unexpected dialogs. It can run as a
background task or perform one-shot health checks for BrowserManager.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from playwright.async_api import Dialog, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel, Field

from src.logger import get_logger

from .browser_session import BrowserSession


DialogAction = Literal["dismiss", "accept", "ignore"]


class ObserverConfig(BaseModel):
    """
    Configuration for browser observation.

    Attributes:
        check_interval_seconds: Delay between background observation loops.
        page_check_timeout_seconds: Maximum time for page readiness checks.
        network_idle_timeout_seconds: Maximum time to wait for network idle.
        dialog_action: Action taken when unexpected dialogs appear.
        max_dialog_history: Number of dialog observations retained.
    """

    check_interval_seconds: float = Field(default=1.0, gt=0)
    page_check_timeout_seconds: float = Field(default=2.0, gt=0)
    network_idle_timeout_seconds: float = Field(default=1.0, gt=0)
    dialog_action: DialogAction = Field(default="dismiss")
    max_dialog_history: int = Field(default=20, ge=1)


@dataclass(frozen=True, slots=True)
class DialogObservation:
    """
    Snapshot of an unexpected browser dialog.

    Attributes:
        message: Dialog message text.
        dialog_type: Playwright dialog type.
        page_url: URL of the page that produced the dialog.
        handled: Whether the observer accepted or dismissed the dialog.
        action: Observer action applied to the dialog.
        observed_at: UTC timestamp when the dialog was observed.
    """

    message: str
    dialog_type: str
    page_url: str
    handled: bool
    action: DialogAction
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class PageObservation:
    """
    Snapshot of one page's observed state.

    Attributes:
        url: Current page URL.
        title: Current page title when available.
        alive: Whether the page is open and responsive.
        loading_state: Browser document ready state when available.
        network_idle: Whether Playwright observed network idle.
        crashed: Whether a crash event was observed for the page.
        timed_out: Whether a readiness check exceeded its timeout.
        last_error: Last observation error for this page.
    """

    url: str
    title: str
    alive: bool
    loading_state: str
    network_idle: bool
    crashed: bool
    timed_out: bool
    last_error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ObserverSnapshot:
    """
    Current observer view of a browser session.

    Attributes:
        session_id: Browser session identifier.
        browser_alive: Whether the browser context is usable.
        active_page_alive: Whether the session's active page is alive.
        page_count: Number of open pages.
        pages: Per-page observations.
        dialogs: Recent dialog observations.
        last_error: Last observer-level error, if any.
        observed_at: UTC timestamp for this snapshot.
    """

    session_id: str
    browser_alive: bool
    active_page_alive: bool
    page_count: int
    pages: list[PageObservation] = field(default_factory=list)
    dialogs: list[DialogObservation] = field(default_factory=list)
    last_error: Optional[str] = None
    observed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Observer:
    """
    Monitors browser sessions for liveness and unexpected conditions.

    Observer attaches Playwright event handlers to pages and also polls page
    state. Dialog handling is configurable so automation can choose to dismiss,
    accept, or simply record dialogs.
    """

    def __init__(
        self,
        config: Optional[ObserverConfig] = None,
    ) -> None:
        """
        Initialize the observer.

        Args:
            config: Optional observation configuration.
        """
        self._config = config or ObserverConfig()
        self._logger = get_logger(__name__)
        self._session: Optional[BrowserSession] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._snapshot: Optional[ObserverSnapshot] = None
        self._dialogs: list[DialogObservation] = []
        self._attached_page_ids: set[int] = set()
        self._crashed_page_ids: set[int] = set()
        self._dialog_tasks: set[asyncio.Task[None]] = set()

    @property
    def snapshot(self) -> Optional[ObserverSnapshot]:
        """
        Return the latest observer snapshot.

        Returns:
            Latest snapshot or None when observation has not run.
        """
        return self._snapshot

    @property
    def is_running(self) -> bool:
        """
        Return whether background observation is active.

        Returns:
            True when the observer task is running.
        """
        return self._task is not None and not self._task.done()

    async def start(self, session: BrowserSession) -> None:
        """
        Start continuous observation for a session.

        Args:
            session: Browser session to observe.
        """
        await self.stop()
        self._session = session
        self._attach_page_events(session)
        self._snapshot = await self.check_once(session)
        self._task = asyncio.create_task(self._monitor_loop())
        self._logger.debug("Browser observer started")

    async def stop(self) -> None:
        """
        Stop continuous observation.

        Cancels background checks and waits for any dialog handlers to finish.
        """
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        if self._dialog_tasks:
            await asyncio.gather(*self._dialog_tasks, return_exceptions=True)
            self._dialog_tasks.clear()

        self._session = None
        self._logger.debug("Browser observer stopped")

    async def check_once(
        self,
        session: Optional[BrowserSession] = None,
    ) -> ObserverSnapshot:
        """
        Perform one observation pass.

        Args:
            session: Optional session override. Defaults to the observed
                session.

        Returns:
            Observer snapshot.
        """
        observed_session = session or self._session
        if observed_session is None:
            raise ValueError("Observer.check_once requires a browser session")

        last_error = None
        pages: list[PageObservation] = []

        try:
            self._attach_page_events(observed_session)
            open_pages = observed_session.sync_pages()
            browser_alive = observed_session.is_browser_alive()
        except Exception as exc:
            browser_alive = False
            open_pages = []
            last_error = str(exc)

        for page in open_pages:
            pages.append(await self._observe_page(page))

        active_page = observed_session.active_page
        active_page_alive = (
            active_page is not None
            and not active_page.is_closed()
            and id(active_page) not in self._crashed_page_ids
        )

        snapshot = ObserverSnapshot(
            session_id=observed_session.session_id,
            browser_alive=browser_alive,
            active_page_alive=active_page_alive,
            page_count=len(open_pages),
            pages=pages,
            dialogs=list(self._dialogs),
            last_error=last_error,
        )
        self._snapshot = snapshot
        return snapshot

    async def _monitor_loop(self) -> None:
        """Run background observation until cancelled."""
        while True:
            if self._session is not None:
                try:
                    await self.check_once(self._session)
                except Exception as exc:
                    self._logger.warning("Browser observation failed: %s", exc)
            await asyncio.sleep(self._config.check_interval_seconds)

    def _attach_page_events(self, session: BrowserSession) -> None:
        """Attach dialog and crash handlers to pages not seen before."""
        for page in session.context.pages:
            page_id = id(page)
            if page_id in self._attached_page_ids:
                continue

            page.on("dialog", self._on_dialog)
            page.on(
                "crash",
                lambda *args, _page=page: self._mark_crashed(_page),
            )
            page.on(
                "close",
                lambda *args, _page=page: self._mark_closed(_page),
            )
            self._attached_page_ids.add(page_id)

    def _on_dialog(self, dialog: Dialog) -> None:
        """Schedule asynchronous handling for a browser dialog."""
        task = asyncio.create_task(self._handle_dialog(dialog))
        self._dialog_tasks.add(task)
        task.add_done_callback(self._dialog_tasks.discard)

    async def _handle_dialog(self, dialog: Dialog) -> None:
        """Record and optionally handle an unexpected browser dialog."""
        handled = False
        page_url = ""

        with suppress(Exception):
            page = dialog.page
            if page is not None:
                page_url = page.url

        try:
            if self._config.dialog_action == "dismiss":
                await dialog.dismiss()
                handled = True
            elif self._config.dialog_action == "accept":
                await dialog.accept()
                handled = True
        except Exception as exc:
            self._logger.warning("Failed to handle browser dialog: %s", exc)

        observation = DialogObservation(
            message=dialog.message,
            dialog_type=dialog.type,
            page_url=page_url,
            handled=handled,
            action=self._config.dialog_action,
            observed_at=datetime.now(timezone.utc),
        )
        self._dialogs.append(observation)
        self._dialogs = self._dialogs[-self._config.max_dialog_history:]

    def _mark_crashed(self, page: Page) -> None:
        """Record a page crash event."""
        self._crashed_page_ids.add(id(page))
        self._logger.warning("Observed browser page crash: %s", page.url)

    def _mark_closed(self, page: Page) -> None:
        """Forget event state for a closed page."""
        page_id = id(page)
        self._crashed_page_ids.discard(page_id)

    async def _observe_page(self, page: Page) -> PageObservation:
        """Observe one page's liveness, loading, and network-idle state."""
        page_url = page.url
        page_title = ""
        loading_state = "unknown"
        network_idle = False
        timed_out = False
        last_error = None
        crashed = id(page) in self._crashed_page_ids

        if page.is_closed():
            return PageObservation(
                url=page_url,
                title=page_title,
                alive=False,
                loading_state="closed",
                network_idle=False,
                crashed=crashed,
                timed_out=False,
                last_error=None,
            )

        try:
            page_title = await asyncio.wait_for(
                page.title(),
                timeout=self._config.page_check_timeout_seconds,
            )
            loading_state = await asyncio.wait_for(
                page.evaluate("() => document.readyState"),
                timeout=self._config.page_check_timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
            last_error = "Page readiness check timed out"
        except Exception as exc:
            last_error = str(exc)
            if "crash" in last_error.lower():
                crashed = True

        try:
            await page.wait_for_load_state(
                "networkidle",
                timeout=self._config.network_idle_timeout_seconds * 1000,
            )
            network_idle = True
        except PlaywrightTimeoutError:
            timed_out = True
            network_idle = False
        except Exception as exc:
            if last_error is None:
                last_error = str(exc)
            if "crash" in str(exc).lower():
                crashed = True

        return PageObservation(
            url=page_url,
            title=page_title,
            alive=not page.is_closed() and not crashed,
            loading_state=loading_state,
            network_idle=network_idle,
            crashed=crashed,
            timed_out=timed_out,
            last_error=last_error,
        )


__all__ = [
    "DialogAction",
    "DialogObservation",
    "Observer",
    "ObserverConfig",
    "ObserverSnapshot",
    "PageObservation",
]
