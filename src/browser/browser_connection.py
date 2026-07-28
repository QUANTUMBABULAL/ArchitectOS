"""
Connection resolution for browser sessions.

BrowserConnection decides how the browser operating system obtains a
browser: by attaching to an already-running Chrome exposed through the
Chrome DevTools Protocol (CDP), by launching a new Chrome through
BrowserFactory, or automatically (attach when available, launch otherwise).

Attaching is how the system reuses a browser the user has already signed
into manually. This module never automates login; it only connects to
what the user has prepared.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from uuid import uuid4

import requests
from playwright.async_api import Playwright, async_playwright
from pydantic import BaseModel, Field, field_validator

from src.config import Settings, get_settings
from src.exceptions import (
    BrowserAttachError,
    BrowserError,
    RemoteDebugUnavailableError,
)
from src.logger import get_logger

from .browser_factory import BrowserFactory, BrowserLaunchConfig
from .browser_session import BrowserSession


class BrowserConnectionConfig(BaseModel):
    """
    Configuration for resolving a browser connection.

    Attributes:
        mode: Connection mode. ``attach`` requires a remote-debug Chrome,
            ``launch`` always starts a new browser, and ``auto`` prefers
            attach with launch as fallback.
        host: Remote debugging host.
        port: Chrome remote debugging port.
        attach_timeout_seconds: Timeout for endpoint probing and CDP attach.
        operation_timeout_seconds: Default Playwright operation timeout
            applied to attached contexts.
    """

    mode: str = Field(default="auto")
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=9222, gt=0, le=65535)
    attach_timeout_seconds: float = Field(default=5.0, gt=0)
    operation_timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        """
        Validate and normalize the connection mode.

        Args:
            value: Raw connection mode.

        Returns:
            Normalized connection mode.

        Raises:
            ValueError: If the mode is not auto, launch, or attach.
        """
        mode = value.lower().strip()
        valid_modes = {"auto", "launch", "attach"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid connection mode '{value}'. "
                f"Must be one of: {valid_modes}"
            )
        return mode

    @property
    def endpoint_url(self) -> str:
        """
        Return the CDP HTTP endpoint URL.

        Returns:
            Base URL of the remote debugging endpoint.
        """
        return f"http://{self.host}:{self.port}"

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        **overrides: object,
    ) -> "BrowserConnectionConfig":
        """
        Build connection configuration from application settings.

        Args:
            settings: Existing application settings object.
            **overrides: Explicit values replacing setting defaults.

        Returns:
            Browser connection configuration.
        """
        values: dict[str, object] = {
            "mode": settings.browser_mode,
            "port": settings.remote_debug_port,
            "attach_timeout_seconds": settings.attach_timeout,
            "operation_timeout_seconds": settings.browser_timeout,
        }
        values.update(overrides)
        return cls(**values)


class BrowserConnection:
    """
    Resolves browser sessions through attach, launch, or auto policy.

    BrowserConnection exposes the same ``create_session``/``stop`` surface
    as BrowserFactory, so it can be injected wherever a factory is expected
    (for example into BrowserManager) without changing lifecycle code.
    Launch requests are delegated to a real BrowserFactory; attach requests
    are handled here through Playwright's CDP client.
    """

    def __init__(
        self,
        config: Optional[BrowserConnectionConfig] = None,
        settings: Optional[Settings] = None,
        factory: Optional[BrowserFactory] = None,
        playwright: Optional[Playwright] = None,
    ) -> None:
        """
        Initialize the browser connection resolver.

        Args:
            config: Optional connection configuration.
            settings: Optional application settings used when config is
                absent.
            factory: Optional browser factory used for launch mode.
            playwright: Optional externally managed Playwright runtime.
        """
        self._logger = get_logger(__name__)
        self._settings = settings or get_settings()
        self._config = config or BrowserConnectionConfig.from_settings(
            self._settings
        )
        self._factory = factory or BrowserFactory(playwright=playwright)
        self._playwright = playwright
        self._owns_playwright = playwright is None

    @property
    def config(self) -> BrowserConnectionConfig:
        """
        Return the active connection configuration.

        Returns:
            Connection configuration.
        """
        return self._config

    async def start(self) -> None:
        """
        Start the Playwright runtime if this connection owns it.

        Raises:
            BrowserError: If Playwright cannot be started.
        """
        if self._playwright is not None:
            return

        try:
            self._playwright = await async_playwright().start()
            self._owns_playwright = True
            self._logger.debug("Playwright runtime started for attach mode")
        except Exception as exc:
            raise BrowserError(
                f"Failed to start Playwright: {exc}",
                code="PLAYWRIGHT_START_FAILED",
            ) from exc

    async def stop(self) -> None:
        """
        Stop owned runtimes.

        Stops the delegated factory and the Playwright runtime this
        connection owns. Attached browsers themselves are never terminated.

        Raises:
            BrowserError: If shutdown fails.
        """
        shutdown_error: Optional[Exception] = None

        try:
            await self._factory.stop()
        except Exception as exc:
            shutdown_error = exc

        if self._playwright is not None and self._owns_playwright:
            try:
                await self._playwright.stop()
                self._logger.debug("Playwright runtime stopped")
            except Exception as exc:
                shutdown_error = exc
            finally:
                self._playwright = None

        if shutdown_error is not None:
            if isinstance(shutdown_error, BrowserError):
                raise shutdown_error
            raise BrowserError(
                f"Failed to stop browser connection: {shutdown_error}",
                code="BROWSER_CONNECTION_STOP_FAILED",
            ) from shutdown_error

    async def is_remote_debug_available(self) -> bool:
        """
        Check whether the remote debugging endpoint is reachable.

        Returns:
            True when the CDP endpoint answers with version metadata.
        """
        try:
            await self.describe_endpoint()
            return True
        except RemoteDebugUnavailableError:
            return False

    async def describe_endpoint(self) -> dict[str, Any]:
        """
        Fetch CDP version metadata from the remote debugging endpoint.

        Returns:
            Parsed ``/json/version`` payload from Chrome.

        Raises:
            RemoteDebugUnavailableError: If the endpoint is unreachable or
                does not return valid CDP metadata.
        """
        url = f"{self._config.endpoint_url}/json/version"

        def probe() -> dict[str, Any]:
            response = requests.get(
                url,
                timeout=self._config.attach_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("CDP metadata payload is not an object")
            return payload

        try:
            payload = await asyncio.to_thread(probe)
        except Exception as exc:
            raise RemoteDebugUnavailableError(
                f"Chrome remote debugging endpoint is unavailable at "
                f"{url}: {exc}",
                code="REMOTE_DEBUG_UNAVAILABLE",
            ) from exc

        if "webSocketDebuggerUrl" not in payload:
            raise RemoteDebugUnavailableError(
                f"Endpoint at {url} did not expose a webSocketDebuggerUrl; "
                "it does not look like a Chrome DevTools endpoint",
                code="REMOTE_DEBUG_INVALID",
            )
        return payload

    async def attach(self) -> BrowserSession:
        """
        Attach to a running Chrome over CDP.

        The session reuses the browser's default context, so cookies and
        login state created manually by the user are available. The session
        is marked as not owning the browser, so closing it only disconnects.

        Returns:
            Attached browser session.

        Raises:
            RemoteDebugUnavailableError: If the endpoint is unreachable.
            BrowserAttachError: If the CDP connection fails.
        """
        metadata = await self.describe_endpoint()
        await self.start()

        if self._playwright is None:
            raise BrowserAttachError(
                "Playwright runtime is unavailable",
                code="PLAYWRIGHT_UNAVAILABLE",
            )

        try:
            browser = await self._playwright.chromium.connect_over_cdp(
                self._config.endpoint_url,
                timeout=self._config.attach_timeout_seconds * 1000,
            )
        except Exception as exc:
            raise BrowserAttachError(
                f"Failed to attach to Chrome at "
                f"{self._config.endpoint_url}: {exc}",
                code="BROWSER_ATTACH_FAILED",
            ) from exc

        try:
            if browser.contexts:
                context = browser.contexts[0]
            else:
                context = await browser.new_context()

            timeout_ms = self._config.operation_timeout_seconds * 1000
            context.set_default_timeout(timeout_ms)
            context.set_default_navigation_timeout(timeout_ms)

            session = BrowserSession(
                browser=browser,
                context=context,
                session_id=uuid4().hex,
                owns_browser=False,
                metadata={
                    "connection_mode": "attach",
                    "endpoint_url": self._config.endpoint_url,
                    "browser_version": metadata.get("Browser", "unknown"),
                    "persistent_context": True,
                },
            )
            session.mark_running()
            self._logger.info(
                "Attached to Chrome %s at %s (session %s)",
                metadata.get("Browser", "unknown"),
                self._config.endpoint_url,
                session.session_id,
            )
            return session
        except Exception as exc:
            try:
                await browser.close()
            except Exception:
                self._logger.warning(
                    "Failed to disconnect after attach error"
                )
            if isinstance(exc, BrowserError):
                raise
            raise BrowserAttachError(
                f"Failed to initialize attached browser session: {exc}",
                code="BROWSER_ATTACH_SESSION_FAILED",
            ) from exc

    async def create_session(
        self,
        config: Optional[BrowserLaunchConfig] = None,
    ) -> BrowserSession:
        """
        Create a browser session according to the configured mode.

        Args:
            config: Optional launch configuration used for launch mode and
                for auto-mode fallback.

        Returns:
            Browser session obtained by attach or launch.

        Raises:
            RemoteDebugUnavailableError: In attach mode when the endpoint
                is unreachable.
            BrowserAttachError: In attach mode when connecting fails.
            BrowserError: If launching fails.
        """
        mode = self._config.mode

        if mode == "attach":
            return await self.attach()

        if mode == "launch":
            return await self._factory.create_session(config)

        if await self.is_remote_debug_available():
            try:
                return await self.attach()
            except BrowserError as exc:
                self._logger.warning(
                    "Attach failed in auto mode, falling back to launch: %s",
                    exc,
                )

        self._logger.info(
            "Remote debugging unavailable at %s; launching a new browser",
            self._config.endpoint_url,
        )
        return await self._factory.create_session(config)


__all__ = [
    "BrowserConnection",
    "BrowserConnectionConfig",
]
