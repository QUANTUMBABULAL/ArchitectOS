"""
High-level browser lifecycle manager.

BrowserManager owns browser startup, shutdown, restart, and health checks.
It delegates browser creation to BrowserFactory and continuous observation
to Observer so callers can inject substitutes during testing or future
runtime integrations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.exceptions import BrowserError
from src.logger import get_logger

from .browser_connection import BrowserConnection, BrowserConnectionConfig
from .browser_factory import BrowserFactory, BrowserLaunchConfig
from .browser_session import BrowserSession, BrowserSessionState
from .observer import Observer, ObserverConfig
from .profile_lock import ProfileLock


class BrowserManagerConfig(BaseModel):
    """
    Configuration for BrowserManager lifecycle behavior.

    Attributes:
        launch_config: Default browser launch configuration.
        connection_config: Connection-resolution configuration deciding
            between attaching to a running Chrome and launching one.
        observer_config: Configuration for continuous browser observation.
        restart_delay_seconds: Pause between stop and start during restart.
    """

    launch_config: BrowserLaunchConfig = Field(
        default_factory=BrowserLaunchConfig,
        description="Default browser launch configuration.",
    )
    connection_config: BrowserConnectionConfig = Field(
        default_factory=BrowserConnectionConfig,
        description="Browser connection resolution configuration.",
    )
    observer_config: ObserverConfig = Field(
        default_factory=ObserverConfig,
        description="Observer configuration.",
    )
    restart_delay_seconds: float = Field(default=0.25, ge=0)


@dataclass(frozen=True, slots=True)
class BrowserHealth:
    """
    Health-check result for the managed browser session.

    Attributes:
        session_id: Current session identifier, if one exists.
        healthy: True when browser and active page are alive.
        browser_alive: True when the browser context is usable.
        active_page_alive: True when the active page is usable.
        page_count: Number of open pages in the session.
        state: Current browser session state.
        current_url: URL of the active page, when available.
        last_error: Last observed browser error, when available.
    """

    session_id: Optional[str]
    healthy: bool
    browser_alive: bool
    active_page_alive: bool
    page_count: int
    state: BrowserSessionState
    current_url: Optional[str] = None
    last_error: Optional[str] = None


class BrowserManager:
    """
    Owns the lifecycle of one active browser session.

    BrowserManager starts, stops, restarts, and health-checks the browser.
    It uses dependency injection for settings, connection, and observer so
    the browser operating system can be reused by future workers without
    those workers needing to own Playwright lifecycle details.

    Session acquisition is delegated to BrowserConnection, which applies
    the configured policy:

    * ``attach``  — connect to a Chrome already exposing remote debugging
    * ``launch``  — always start a dedicated automation-profile Chrome
    * ``auto``    — attach when a remote-debug Chrome is reachable,
      otherwise launch

    The user's Default Chrome profile is therefore never required, and an
    already-open Chrome never blocks startup.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        factory: Optional[BrowserFactory] = None,
        observer: Optional[Observer] = None,
        config: Optional[BrowserManagerConfig] = None,
        connection: Optional[BrowserConnection] = None,
    ) -> None:
        """
        Initialize the browser manager.

        Args:
            settings: Optional application settings instance.
            factory: Optional browser factory used for launch mode.
            observer: Optional browser observer dependency.
            config: Optional manager configuration.
            connection: Optional connection resolver. When omitted, one is
                built from settings and wraps ``factory``.
        """
        self._logger = get_logger(__name__)
        self._settings = settings or get_settings()
        self._config = config or BrowserManagerConfig(
            launch_config=BrowserLaunchConfig.from_settings(self._settings),
            connection_config=BrowserConnectionConfig.from_settings(
                self._settings
            ),
        )
        self._factory = factory or BrowserFactory()
        self._connection = connection or BrowserConnection(
            config=self._config.connection_config,
            settings=self._settings,
            factory=self._factory,
        )
        self._observer = observer or Observer(self._config.observer_config)
        self._session: Optional[BrowserSession] = None
        self._profile_lock: Optional[ProfileLock] = None

    @property
    def session(self) -> Optional[BrowserSession]:
        """
        Return the current browser session, if one exists.

        Returns:
            Current session or None.
        """
        return self._session

    @property
    def is_running(self) -> bool:
        """
        Return whether the manager has a running session.

        Returns:
            True when a session exists and reports itself alive.
        """
        return (
            self._session is not None
            and self._session.state == BrowserSessionState.RUNNING
            and self._session.is_browser_alive()
        )

    async def start(
        self,
        launch_config: Optional[BrowserLaunchConfig] = None,
    ) -> BrowserSession:
        """
        Start the managed browser session.

        The session is resolved through BrowserConnection, so an existing
        remote-debug Chrome is reused when the mode allows it and a
        dedicated automation profile is launched otherwise.

        Args:
            launch_config: Optional launch configuration override.

        Returns:
            Running browser session.

        Raises:
            BrowserError: If startup fails.
        """
        if self.is_running and self._session is not None:
            return self._session

        config = launch_config or self._config.launch_config
        self._logger.info(
            "Starting browser session (mode=%s)",
            self._config.connection_config.mode,
        )

        self._acquire_profile_lock(config)

        try:
            self._session = await self._connection.create_session(config)
            await self._observer.start(self._session)
            self._logger.info(
                "Browser session %s ready via %s mode",
                self._session.session_id,
                self._session.metadata.get("connection_mode", "launch"),
            )
            return self._session
        except Exception as exc:
            self._session = None
            self._release_profile_lock()
            if isinstance(exc, BrowserError):
                raise
            raise BrowserError(
                f"Failed to start browser manager: {exc}",
                code="BROWSER_MANAGER_START_FAILED",
            ) from exc

    def _acquire_profile_lock(
        self,
        config: BrowserLaunchConfig,
    ) -> None:
        """
        Take the advisory lock on the automation user data directory.

        Guards against two ArchitectOS instances driving the same
        automation profile, which the dedicated-directory strategy alone
        does not prevent. Skipped when no explicit user data directory is
        configured, because there is then no directory this process can
        claim ownership of.

        Args:
            config: Launch configuration for the pending session.

        Raises:
            ProfileLockedError: If another instance holds the directory.
        """
        if config.user_data_dir is None:
            return

        lock = ProfileLock(config.user_data_dir)
        lock.acquire()
        self._profile_lock = lock

    def _release_profile_lock(self) -> None:
        """
        Release the advisory profile lock if this manager holds one.

        Never raises; safe to call from shutdown and failure paths.
        """
        if self._profile_lock is None:
            return
        self._profile_lock.release()
        self._profile_lock = None

    async def stop(self) -> None:
        """
        Stop the managed browser session and Playwright runtime.

        Raises:
            BrowserError: If shutdown fails.
        """
        self._logger.info("Stopping browser session")
        shutdown_error: Optional[Exception] = None

        try:
            await self._observer.stop()
        except Exception as exc:
            shutdown_error = exc

        if self._session is not None:
            try:
                await self._session.close()
            except Exception as exc:
                shutdown_error = exc
            finally:
                self._session = None

        try:
            await self._connection.stop()
        except Exception as exc:
            shutdown_error = exc

        # Released last: the lock must outlive the browser it guards, so
        # no other instance can claim the directory mid-shutdown.
        self._release_profile_lock()

        if shutdown_error is not None:
            if isinstance(shutdown_error, BrowserError):
                raise shutdown_error
            raise BrowserError(
                f"Failed to stop browser manager: {shutdown_error}",
                code="BROWSER_MANAGER_STOP_FAILED",
            ) from shutdown_error

    async def restart(
        self,
        launch_config: Optional[BrowserLaunchConfig] = None,
    ) -> BrowserSession:
        """
        Restart the managed browser session.

        Args:
            launch_config: Optional launch configuration override.

        Returns:
            Newly started browser session.
        """
        self._logger.info("Restarting browser session")
        await self.stop()
        if self._config.restart_delay_seconds:
            await asyncio.sleep(self._config.restart_delay_seconds)
        return await self.start(launch_config)

    async def health_check(self) -> BrowserHealth:
        """
        Perform a browser health check.

        Returns:
            Browser health result describing browser and active page state.
        """
        if self._session is None:
            return BrowserHealth(
                session_id=None,
                healthy=False,
                browser_alive=False,
                active_page_alive=False,
                page_count=0,
                state=BrowserSessionState.STOPPED,
                last_error="No browser session is active",
            )

        snapshot = await self._observer.check_once(self._session)
        active_page = self._session.active_page
        current_url = None
        if active_page is not None and not active_page.is_closed():
            current_url = active_page.url

        healthy = (
            snapshot.browser_alive
            and snapshot.active_page_alive
            and self._session.state == BrowserSessionState.RUNNING
        )

        return BrowserHealth(
            session_id=self._session.session_id,
            healthy=healthy,
            browser_alive=snapshot.browser_alive,
            active_page_alive=snapshot.active_page_alive,
            page_count=snapshot.page_count,
            state=self._session.state,
            current_url=current_url,
            last_error=snapshot.last_error,
        )

    def get_session(self) -> BrowserSession:
        """
        Return the current browser session.

        Returns:
            Active browser session.

        Raises:
            BrowserError: If no session is active.
        """
        if self._session is None:
            raise BrowserError(
                "No browser session is active",
                code="BROWSER_SESSION_NOT_STARTED",
            )
        return self._session


__all__ = [
    "BrowserHealth",
    "BrowserManager",
    "BrowserManagerConfig",
]
