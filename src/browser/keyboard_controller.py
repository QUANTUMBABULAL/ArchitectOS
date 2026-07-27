"""
Keyboard automation utilities for Playwright pages.

KeyboardController centralizes text entry, key presses, shortcuts, textbox
clearing, and clipboard paste behavior. It is generic browser input code
and contains no assumptions about any destination website.
"""

from __future__ import annotations

import sys
from typing import Optional

from playwright.async_api import Page
from pydantic import BaseModel, Field

from src.exceptions import BrowserError
from src.logger import get_logger


class KeyboardControllerConfig(BaseModel):
    """
    Configuration for keyboard automation timing.

    Attributes:
        typing_delay_ms: Delay between typed characters.
        shortcut_delay_ms: Delay before pressing keyboard shortcuts.
        clear_delay_ms: Delay used while clearing text boxes.
    """

    typing_delay_ms: int = Field(default=40, ge=0)
    shortcut_delay_ms: int = Field(default=20, ge=0)
    clear_delay_ms: int = Field(default=10, ge=0)


class KeyboardController:
    """
    Controls keyboard input on Playwright pages.

    The controller uses Playwright's async keyboard API and optionally focuses
    a selector before performing input. Typing delays are configurable so
    callers can use more human-like input when sites react poorly to instant
    text insertion.
    """

    def __init__(
        self,
        config: Optional[KeyboardControllerConfig] = None,
    ) -> None:
        """
        Initialize the keyboard controller.

        Args:
            config: Optional keyboard timing configuration.
        """
        self._config = config or KeyboardControllerConfig()
        self._logger = get_logger(__name__)

    async def type_text(
        self,
        page: Page,
        text: str,
        selector: Optional[str] = None,
        delay_ms: Optional[int] = None,
        clear_first: bool = False,
    ) -> None:
        """
        Type text naturally into the current focus or a selector.

        Args:
            page: Playwright page receiving input.
            text: Text to type.
            selector: Optional selector to focus before typing.
            delay_ms: Optional per-character delay override.
            clear_first: Whether to clear the target before typing.

        Raises:
            BrowserError: If typing fails.
        """
        try:
            if clear_first:
                await self.clear_textbox(page=page, selector=selector)
            else:
                await self._focus_selector(page=page, selector=selector)

            await page.keyboard.type(
                text,
                delay=(
                    self._config.typing_delay_ms
                    if delay_ms is None
                    else delay_ms
                ),
            )
            self._logger.debug("Typed %d characters", len(text))
        except Exception as exc:
            raise BrowserError(
                f"Failed to type text: {exc}",
                code="BROWSER_KEYBOARD_TYPE_FAILED",
            ) from exc

    async def press_key(
        self,
        page: Page,
        key: str,
        selector: Optional[str] = None,
    ) -> None:
        """
        Press one keyboard key.

        Args:
            page: Playwright page receiving input.
            key: Playwright key name, such as ``Enter`` or ``Escape``.
            selector: Optional selector to focus first.

        Raises:
            BrowserError: If the key press fails.
        """
        try:
            await self._focus_selector(page=page, selector=selector)
            await page.keyboard.press(key)
        except Exception as exc:
            raise BrowserError(
                f"Failed to press key '{key}': {exc}",
                code="BROWSER_KEYBOARD_PRESS_FAILED",
            ) from exc

    async def keyboard_shortcut(
        self,
        page: Page,
        *keys: str,
        selector: Optional[str] = None,
    ) -> None:
        """
        Press a keyboard shortcut.

        Args:
            page: Playwright page receiving input.
            *keys: Keys composing the shortcut, such as ``Control`` and ``A``.
            selector: Optional selector to focus first.

        Raises:
            BrowserError: If the shortcut is empty or cannot be pressed.
        """
        if not keys:
            raise BrowserError(
                "Keyboard shortcut requires at least one key",
                code="BROWSER_KEYBOARD_SHORTCUT_EMPTY",
            )

        shortcut = "+".join(keys)
        try:
            await self._focus_selector(page=page, selector=selector)
            await page.wait_for_timeout(self._config.shortcut_delay_ms)
            await page.keyboard.press(shortcut)
        except Exception as exc:
            raise BrowserError(
                f"Failed to press keyboard shortcut '{shortcut}': {exc}",
                code="BROWSER_KEYBOARD_SHORTCUT_FAILED",
            ) from exc

    async def clear_textbox(
        self,
        page: Page,
        selector: Optional[str] = None,
    ) -> None:
        """
        Clear the focused textbox or a textbox matched by selector.

        Args:
            page: Playwright page receiving input.
            selector: Optional selector to focus before clearing.

        Raises:
            BrowserError: If clearing fails.
        """
        try:
            await self._focus_selector(page=page, selector=selector)
            await page.keyboard.press(f"{self._modifier_key()}+A")
            await page.wait_for_timeout(self._config.clear_delay_ms)
            await page.keyboard.press("Backspace")
        except Exception as exc:
            raise BrowserError(
                f"Failed to clear textbox: {exc}",
                code="BROWSER_KEYBOARD_CLEAR_FAILED",
            ) from exc

    async def paste_clipboard(
        self,
        page: Page,
        selector: Optional[str] = None,
    ) -> None:
        """
        Paste the current system clipboard into the page.

        Args:
            page: Playwright page receiving input.
            selector: Optional selector to focus before pasting.

        Raises:
            BrowserError: If the paste shortcut fails.
        """
        try:
            await self._focus_selector(page=page, selector=selector)
            await page.keyboard.press(f"{self._modifier_key()}+V")
        except Exception as exc:
            raise BrowserError(
                f"Failed to paste clipboard: {exc}",
                code="BROWSER_KEYBOARD_PASTE_FAILED",
            ) from exc

    async def paste_text(
        self,
        page: Page,
        text: str,
        selector: Optional[str] = None,
    ) -> None:
        """
        Write text to the browser clipboard and paste it into the page.

        Args:
            page: Playwright page receiving input.
            text: Text to place on the browser clipboard.
            selector: Optional selector to focus before pasting.

        Raises:
            BrowserError: If clipboard paste and fallback insertion fail.
        """
        try:
            await self._focus_selector(page=page, selector=selector)
            await page.evaluate(
                """
                async (value) => {
                    if (!navigator.clipboard) {
                        throw new Error("Clipboard API unavailable");
                    }
                    await navigator.clipboard.writeText(value);
                }
                """,
                text,
            )
            await self.paste_clipboard(page=page)
        except Exception:
            try:
                await page.keyboard.insert_text(text)
            except Exception as exc:
                raise BrowserError(
                    f"Failed to paste text: {exc}",
                    code="BROWSER_KEYBOARD_PASTE_TEXT_FAILED",
                ) from exc

    async def _focus_selector(
        self,
        page: Page,
        selector: Optional[str],
    ) -> None:
        """Focus a selector when one is provided."""
        if selector is not None:
            await page.locator(selector).focus()

    @staticmethod
    def _modifier_key() -> str:
        """Return the platform shortcut modifier key."""
        return "Meta" if sys.platform == "darwin" else "Control"


__all__ = [
    "KeyboardController",
    "KeyboardControllerConfig",
]
