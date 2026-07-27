"""
Factory for creating asynchronous Playwright browser sessions.

The factory is responsible for translating browser configuration into
Playwright launch calls. It currently supports Google Chrome through
Playwright's Chromium engine and is structured so Edge or Firefox support
can be added without changing BrowserManager or the controllers.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from playwright.async_api import BrowserType as PlaywrightBrowserType
from playwright.async_api import Playwright, async_playwright
from pydantic import BaseModel, Field, field_validator

from src.config import Settings
from src.constants import (
    BROWSER_LAUNCH_TIMEOUT,
    BROWSER_TIMEOUT,
    BROWSER_VIEWPORT_HEIGHT,
    BROWSER_VIEWPORT_WIDTH,
)
from src.exceptions import (
    BrowserError,
    BrowserLaunchError,
    ChromeNotFoundError,
    ProfileLockedError,
)
from src.logger import get_logger

from .browser_session import BrowserSession
from .chrome_profile import (
    ARCHITECTOS_PROFILE_NAME,
    ChromeProfile,
    ChromeProfileConfig,
)
from .launch_diagnostics import LaunchFailureKind, classify_launch_failure


class BrowserKind(str, Enum):
    """
    Browser families understood by the factory.

    Only Google Chrome is implemented today. The enum makes unsupported
    browser requests explicit and keeps the public contract stable for
    future Edge or Firefox implementations.
    """

    CHROME = "chrome"
    EDGE = "edge"
    FIREFOX = "firefox"


class BrowserLaunchConfig(BaseModel):
    """
    Browser launch settings for a Playwright session.

    Attributes:
        browser_kind: Browser family to launch.
        headless: Whether the browser should run without a visible window.
        persistent_context: Whether to use a persistent user profile.
        executable_path: Optional explicit browser executable path.
        user_data_dir: Optional persistent Chrome user data root directory.
        profile_directory: Chrome profile folder within the user data root.
        viewport_width: Browser viewport width in pixels.
        viewport_height: Browser viewport height in pixels.
        timeout_seconds: Browser launch timeout in seconds.
        operation_timeout_seconds: Default Playwright operation timeout.
        create_profile_if_missing: Whether a missing profile directory
            should be created rather than treated as an error.
        maximized: Whether to open the window maximized with no fixed
            viewport, matching an ordinary Chrome session.
        launch_args: Extra browser launch arguments.
        slow_mo_ms: Playwright slow-motion delay in milliseconds.
    """

    browser_kind: BrowserKind = Field(default=BrowserKind.CHROME)
    headless: bool = Field(default=True)
    persistent_context: bool = Field(default=True)
    executable_path: Optional[Path] = Field(default=None)
    user_data_dir: Optional[Path] = Field(default=None)
    profile_directory: str = Field(default="Default", min_length=1)
    create_profile_if_missing: bool = Field(default=False)
    maximized: bool = Field(default=True)
    viewport_width: int = Field(default=BROWSER_VIEWPORT_WIDTH, gt=0)
    viewport_height: int = Field(default=BROWSER_VIEWPORT_HEIGHT, gt=0)
    timeout_seconds: float = Field(default=BROWSER_LAUNCH_TIMEOUT, gt=0)
    operation_timeout_seconds: float = Field(default=BROWSER_TIMEOUT, gt=0)
    launch_args: tuple[str, ...] = Field(default_factory=tuple)
    slow_mo_ms: int = Field(default=0, ge=0)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        **overrides: object,
    ) -> "BrowserLaunchConfig":
        """
        Build browser launch configuration from application settings.

        The configured automation profile is mapped to a dedicated Chrome
        user data root under the application data directory. Chrome locks
        an entire user data root while it is open, so reusing the user's
        default root would fail whenever their normal Chrome is running.
        Launching into a dedicated root avoids that lock entirely and
        never requires the Default profile.

        Args:
            settings: Existing application settings object.
            **overrides: Explicit values that should replace setting defaults.

        Returns:
            Browser launch configuration.
        """
        values: dict[str, object] = {
            "headless": (
                settings.headless
                if settings.headless is not None
                else settings.browser_headless
            ),
            "viewport_width": settings.browser_viewport_width,
            "viewport_height": settings.browser_viewport_height,
            "timeout_seconds": settings.browser_launch_timeout,
            "operation_timeout_seconds": settings.browser_timeout,
        }

        if settings.chrome_path:
            values["executable_path"] = settings.chrome_path

        values["maximized"] = settings.browser_maximized

        automation_profile = (
            settings.automation_profile or ARCHITECTOS_PROFILE_NAME
        ).strip() or ARCHITECTOS_PROFILE_NAME

        # Preserve any signed-in sessions from an earlier profile layout
        # before deciding where to launch from.
        ChromeProfile.migrate_legacy_profile(
            base_dir=Path(settings.data_dir),
            profile_directory=automation_profile,
        )

        values["user_data_dir"] = ChromeProfile.automation_user_data_dir(
            profile_name=automation_profile,
            base_dir=Path(settings.data_dir),
        )
        values["profile_directory"] = automation_profile
        values["create_profile_if_missing"] = True

        values.update(overrides)
        return cls(**values)

    @field_validator("executable_path", "user_data_dir", mode="before")
    @classmethod
    def expand_optional_path(cls, value: object) -> object:
        """
        Expand user markers in optional path fields.

        Args:
            value: Raw field value supplied to Pydantic.

        Returns:
            Expanded path value or the original value when absent.
        """
        if value is None:
            return value
        return Path(value).expanduser()

    @field_validator("profile_directory")
    @classmethod
    def validate_profile_directory(cls, value: str) -> str:
        """
        Validate a non-empty profile directory name.

        Args:
            value: Profile directory name.

        Returns:
            Trimmed profile directory name.
        """
        profile_directory = value.strip()
        if not profile_directory:
            raise ValueError("profile_directory cannot be empty")
        return profile_directory


class BrowserFactory:
    """
    Creates Playwright browser sessions from validated configuration.

    BrowserFactory owns the Playwright runtime it starts unless a Playwright
    instance is injected. This keeps BrowserManager focused on lifecycle
    policy while the factory handles browser-specific launch details.
    """

    def __init__(
        self,
        playwright: Optional[Playwright] = None,
    ) -> None:
        """
        Initialize the browser factory.

        Args:
            playwright: Optional externally managed Playwright runtime.
        """
        self._logger = get_logger(__name__)
        self._playwright = playwright
        self._owns_playwright = playwright is None

    @property
    def is_started(self) -> bool:
        """
        Return whether a Playwright runtime is available.

        Returns:
            True when the factory can create sessions.
        """
        return self._playwright is not None

    async def start(self) -> None:
        """
        Start the Playwright runtime if this factory owns it.

        Raises:
            BrowserError: If Playwright cannot be started.
        """
        if self._playwright is not None:
            return

        try:
            self._playwright = await async_playwright().start()
            self._owns_playwright = True
            self._logger.debug("Playwright runtime started")
        except Exception as exc:
            raise BrowserError(
                f"Failed to start Playwright: {exc}",
                code="PLAYWRIGHT_START_FAILED",
            ) from exc

    async def stop(self) -> None:
        """
        Stop the Playwright runtime if this factory owns it.

        Raises:
            BrowserError: If Playwright fails during shutdown.
        """
        if self._playwright is None or not self._owns_playwright:
            return

        try:
            await self._playwright.stop()
            self._logger.debug("Playwright runtime stopped")
        except Exception as exc:
            raise BrowserError(
                f"Failed to stop Playwright: {exc}",
                code="PLAYWRIGHT_STOP_FAILED",
            ) from exc
        finally:
            self._playwright = None

    async def create_session(
        self,
        config: Optional[BrowserLaunchConfig] = None,
    ) -> BrowserSession:
        """
        Create a new browser session.

        Args:
            config: Optional launch configuration. Defaults to Chrome with a
                persistent profile.

        Returns:
            New browser session.

        Raises:
            BrowserError: If the browser kind is unsupported or launch fails.
        """
        launch_config = config or BrowserLaunchConfig()
        await self.start()

        if self._playwright is None:
            raise BrowserError(
                "Playwright runtime is unavailable",
                code="PLAYWRIGHT_UNAVAILABLE",
            )

        if launch_config.browser_kind != BrowserKind.CHROME:
            raise BrowserError(
                f"Browser kind '{launch_config.browser_kind.value}' is not "
                "implemented by BrowserFactory yet",
                code="BROWSER_KIND_UNSUPPORTED",
            )

        if launch_config.persistent_context:
            return await self._create_persistent_chrome_session(
                launch_config=launch_config,
            )

        return await self._create_ephemeral_chrome_session(
            launch_config=launch_config,
        )

    async def _create_persistent_chrome_session(
        self,
        launch_config: BrowserLaunchConfig,
    ) -> BrowserSession:
        """Create a Chrome session backed by an existing user profile."""
        if self._playwright is None:
            raise BrowserError(
                "Playwright runtime is unavailable",
                code="PLAYWRIGHT_UNAVAILABLE",
            )

        profile = ChromeProfile.resolve(
            ChromeProfileConfig(
                executable_path=launch_config.executable_path,
                user_data_dir=launch_config.user_data_dir,
                profile_directory=launch_config.profile_directory,
                create_if_missing=launch_config.create_profile_if_missing,
            )
        )

        args = tuple(
            profile.launch_args(maximized=launch_config.maximized)
        ) + launch_config.launch_args

        # A maximized window must not also carry a fixed viewport:
        # Playwright would resize the page back to that size, defeating
        # --start-maximized. no_viewport lets the page track the real
        # window, which is also how an ordinary Chrome session behaves.
        viewport_kwargs: dict[str, object]
        if launch_config.maximized and not launch_config.headless:
            viewport_kwargs = {"no_viewport": True}
        else:
            viewport_kwargs = {
                "viewport": {
                    "width": launch_config.viewport_width,
                    "height": launch_config.viewport_height,
                }
            }

        try:
            context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile.user_data_dir),
                executable_path=str(profile.executable_path),
                headless=launch_config.headless,
                timeout=self._seconds_to_ms(launch_config.timeout_seconds),
                slow_mo=launch_config.slow_mo_ms,
                args=list(args),
                # Drop Playwright's own automation switch so Chrome does
                # not advertise itself as automated. Sign-in flows that
                # refuse automated browsers are less likely to block.
                ignore_default_args=["--enable-automation"],
                **viewport_kwargs,
            )
            context.set_default_timeout(
                self._seconds_to_ms(
                    launch_config.operation_timeout_seconds
                )
            )
            context.set_default_navigation_timeout(
                self._seconds_to_ms(
                    launch_config.operation_timeout_seconds
                )
            )

            session = BrowserSession(
                browser=context.browser,
                context=context,
                session_id=uuid4().hex,
                metadata={
                    "browser_kind": launch_config.browser_kind.value,
                    "persistent_context": True,
                    "executable_path": str(profile.executable_path),
                    "user_data_dir": str(profile.user_data_dir),
                    "profile_directory": profile.profile_directory,
                    "profile_path": str(profile.profile_path),
                },
            )
            session.mark_running()
            self._logger.info(
                "Started persistent Chrome session %s with profile %s",
                session.session_id,
                profile.profile_path,
            )
            return session
        except BrowserError:
            raise
        except Exception as exc:
            raise self._diagnose_launch_failure(
                exc, str(profile.user_data_dir)
            ) from exc

    @staticmethod
    def _diagnose_launch_failure(
        exc: Exception,
        user_data_dir: str,
    ) -> BrowserError:
        """
        Convert a raw launch exception into a precise, actionable error.

        Playwright embeds the browser's own output in its exception text,
        so several unrelated root causes arrive as one opaque error. The
        text is classified here and mapped to the narrowest exception
        type available, with operator guidance attached.

        The singleton hand-off case matters most: Chromium exits with a
        NORMAL status code after deferring to an existing instance, so
        without explicit detection the failure reads as a success with a
        missing browser.

        Args:
            exc: Original exception raised by the launch call.
            user_data_dir: User data directory used for the launch.

        Returns:
            A BrowserError subclass carrying the diagnosis. Returned
            rather than raised so callers keep exception chaining.
        """
        diagnosis = classify_launch_failure(str(exc), user_data_dir)
        message = f"{diagnosis.summary}. {diagnosis.remedy}"

        if diagnosis.kind in {
            LaunchFailureKind.SINGLETON_HANDOFF,
            LaunchFailureKind.PROFILE_IN_USE,
        }:
            return ProfileLockedError(
                message,
                code=f"CHROME_{diagnosis.kind.value.upper()}",
            )

        if diagnosis.kind is LaunchFailureKind.EXECUTABLE_NOT_FOUND:
            return ChromeNotFoundError(
                message,
                code="CHROME_EXECUTABLE_NOT_FOUND",
            )

        return BrowserLaunchError(
            f"{message} Original error: {exc}",
            code=f"CHROME_{diagnosis.kind.value.upper()}",
        )

    async def _create_ephemeral_chrome_session(
        self,
        launch_config: BrowserLaunchConfig,
    ) -> BrowserSession:
        """Create a Chrome session with a fresh browser context."""
        if self._playwright is None:
            raise BrowserError(
                "Playwright runtime is unavailable",
                code="PLAYWRIGHT_UNAVAILABLE",
            )

        browser_type = self._select_browser_type(launch_config.browser_kind)

        executable_path = (
            launch_config.executable_path
            or ChromeProfile.locate_executable()
        )

        launch_kwargs: dict[str, object] = {
            "headless": launch_config.headless,
            "timeout": self._seconds_to_ms(launch_config.timeout_seconds),
            "slow_mo": launch_config.slow_mo_ms,
            "args": list(launch_config.launch_args),
            "executable_path": str(executable_path),
        }

        try:
            browser = await browser_type.launch(**launch_kwargs)
            context = await browser.new_context(
                viewport={
                    "width": launch_config.viewport_width,
                    "height": launch_config.viewport_height,
                },
            )
            context.set_default_timeout(
                self._seconds_to_ms(
                    launch_config.operation_timeout_seconds
                )
            )
            context.set_default_navigation_timeout(
                self._seconds_to_ms(
                    launch_config.operation_timeout_seconds
                )
            )

            session = BrowserSession(
                browser=browser,
                context=context,
                session_id=uuid4().hex,
                metadata={
                    "browser_kind": launch_config.browser_kind.value,
                    "persistent_context": False,
                    "executable_path": str(executable_path),
                },
            )
            session.mark_running()
            self._logger.info(
                "Started ephemeral Chrome session %s",
                session.session_id,
            )
            return session
        except Exception as exc:
            raise BrowserError(
                f"Failed to launch Chrome session: {exc}",
                code="CHROME_LAUNCH_FAILED",
            ) from exc

    def _select_browser_type(
        self,
        browser_kind: BrowserKind,
    ) -> PlaywrightBrowserType:
        """Return the Playwright browser type for a supported browser kind."""
        if self._playwright is None:
            raise BrowserError(
                "Playwright runtime is unavailable",
                code="PLAYWRIGHT_UNAVAILABLE",
            )

        if browser_kind == BrowserKind.CHROME:
            return self._playwright.chromium

        raise BrowserError(
            f"Browser kind '{browser_kind.value}' is not supported",
            code="BROWSER_KIND_UNSUPPORTED",
        )

    @staticmethod
    def _seconds_to_ms(seconds: float) -> float:
        """Convert seconds to Playwright milliseconds."""
        return seconds * 1000


__all__ = [
    "BrowserFactory",
    "BrowserKind",
    "BrowserLaunchConfig",
]
