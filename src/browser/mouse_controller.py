"""
Mouse automation utilities for Playwright pages.

MouseController provides generic pointer movement and mouse actions:
clicking, double-clicking, right-clicking, hovering, dragging, scrolling,
and smooth cursor movement between coordinates or element centers.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from playwright.async_api import Page
from pydantic import BaseModel, Field

from src.exceptions import BrowserError
from src.logger import get_logger


Point = tuple[float, float]


class MouseControllerConfig(BaseModel):
    """
    Configuration for mouse movement.

    Attributes:
        movement_steps: Number of interpolation steps for smooth movement.
        movement_delay_ms: Delay between movement steps.
        scroll_steps: Number of increments used for smooth scrolling.
    """

    movement_steps: int = Field(default=12, ge=1)
    movement_delay_ms: int = Field(default=8, ge=0)
    scroll_steps: int = Field(default=8, ge=1)


class MouseController:
    """
    Controls pointer and mouse input on Playwright pages.

    The controller can target explicit coordinates or selector centers. It
    keeps the last pointer position as instance-local state so future moves
    can be smoothed without relying on globals.
    """

    def __init__(
        self,
        config: Optional[MouseControllerConfig] = None,
    ) -> None:
        """
        Initialize the mouse controller.

        Args:
            config: Optional movement configuration.
        """
        self._config = config or MouseControllerConfig()
        self._last_position: Optional[Point] = None
        self._logger = get_logger(__name__)

    async def click(
        self,
        page: Page,
        selector: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """
        Click an element center or coordinate.

        Args:
            page: Playwright page receiving input.
            selector: Optional selector whose center should be clicked.
            x: Optional x coordinate.
            y: Optional y coordinate.

        Raises:
            BrowserError: If the target cannot be resolved or clicked.
        """
        await self._click(page=page, selector=selector, x=x, y=y, click_count=1)

    async def double_click(
        self,
        page: Page,
        selector: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """
        Double-click an element center or coordinate.

        Args:
            page: Playwright page receiving input.
            selector: Optional selector whose center should be double-clicked.
            x: Optional x coordinate.
            y: Optional y coordinate.
        """
        await self._click(page=page, selector=selector, x=x, y=y, click_count=2)

    async def right_click(
        self,
        page: Page,
        selector: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """
        Right-click an element center or coordinate.

        Args:
            page: Playwright page receiving input.
            selector: Optional selector whose center should be right-clicked.
            x: Optional x coordinate.
            y: Optional y coordinate.
        """
        await self._click(
            page=page,
            selector=selector,
            x=x,
            y=y,
            click_count=1,
            button="right",
        )

    async def hover(
        self,
        page: Page,
        selector: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """
        Move the cursor over an element center or coordinate.

        Args:
            page: Playwright page receiving input.
            selector: Optional selector whose center should be hovered.
            x: Optional x coordinate.
            y: Optional y coordinate.

        Raises:
            BrowserError: If the hover target cannot be resolved.
        """
        try:
            point = await self._resolve_point(page=page, selector=selector, x=x, y=y)
            await self.move_smoothly(page=page, x=point[0], y=point[1])
        except Exception as exc:
            raise BrowserError(
                f"Failed to hover mouse: {exc}",
                code="BROWSER_MOUSE_HOVER_FAILED",
            ) from exc

    async def drag(
        self,
        page: Page,
        source_selector: Optional[str] = None,
        target_selector: Optional[str] = None,
        source_x: Optional[float] = None,
        source_y: Optional[float] = None,
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
    ) -> None:
        """
        Drag from a source element or coordinate to a target.

        Args:
            page: Playwright page receiving input.
            source_selector: Optional source selector.
            target_selector: Optional target selector.
            source_x: Optional source x coordinate.
            source_y: Optional source y coordinate.
            target_x: Optional target x coordinate.
            target_y: Optional target y coordinate.

        Raises:
            BrowserError: If drag targets cannot be resolved or moved.
        """
        try:
            source = await self._resolve_point(
                page=page,
                selector=source_selector,
                x=source_x,
                y=source_y,
            )
            target = await self._resolve_point(
                page=page,
                selector=target_selector,
                x=target_x,
                y=target_y,
            )

            await self.move_smoothly(page=page, x=source[0], y=source[1])
            await page.mouse.down()
            await self.move_smoothly(
                page=page,
                x=target[0],
                y=target[1],
                steps=max(self._config.movement_steps, 2),
            )
            await page.mouse.up()
        except Exception as exc:
            raise BrowserError(
                f"Failed to drag mouse: {exc}",
                code="BROWSER_MOUSE_DRAG_FAILED",
            ) from exc

    async def scroll(
        self,
        page: Page,
        delta_x: float = 0,
        delta_y: float = 0,
        x: Optional[float] = None,
        y: Optional[float] = None,
        smooth: bool = True,
    ) -> None:
        """
        Scroll the page at the current or selected cursor position.

        Args:
            page: Playwright page receiving input.
            delta_x: Horizontal scroll amount.
            delta_y: Vertical scroll amount.
            x: Optional x coordinate to move to before scrolling.
            y: Optional y coordinate to move to before scrolling.
            smooth: Whether to split the scroll into increments.

        Raises:
            BrowserError: If scrolling fails.
        """
        try:
            if x is not None or y is not None:
                if x is None or y is None:
                    raise BrowserError(
                        "Both x and y are required when selecting a scroll point",
                        code="BROWSER_MOUSE_POINT_INCOMPLETE",
                    )
                await self.move_smoothly(page=page, x=x, y=y)

            steps = self._config.scroll_steps if smooth else 1
            step_x = delta_x / steps
            step_y = delta_y / steps
            for _ in range(steps):
                await page.mouse.wheel(step_x, step_y)
                if smooth and self._config.movement_delay_ms:
                    await asyncio.sleep(self._config.movement_delay_ms / 1000)
        except Exception as exc:
            raise BrowserError(
                f"Failed to scroll mouse: {exc}",
                code="BROWSER_MOUSE_SCROLL_FAILED",
            ) from exc

    async def move_smoothly(
        self,
        page: Page,
        x: float,
        y: float,
        start: Optional[Point] = None,
        steps: Optional[int] = None,
    ) -> None:
        """
        Move the cursor smoothly to a coordinate.

        Args:
            page: Playwright page receiving input.
            x: Destination x coordinate.
            y: Destination y coordinate.
            start: Optional explicit start coordinate.
            steps: Optional interpolation step count.

        Raises:
            BrowserError: If movement fails.
        """
        try:
            start_point = start or self._last_position or (x, y)
            step_count = steps or self._config.movement_steps

            for index in range(1, step_count + 1):
                ratio = index / step_count
                next_x = start_point[0] + (x - start_point[0]) * ratio
                next_y = start_point[1] + (y - start_point[1]) * ratio
                await page.mouse.move(next_x, next_y)
                if self._config.movement_delay_ms:
                    await asyncio.sleep(self._config.movement_delay_ms / 1000)

            self._last_position = (x, y)
        except Exception as exc:
            raise BrowserError(
                f"Failed to move mouse: {exc}",
                code="BROWSER_MOUSE_MOVE_FAILED",
            ) from exc

    async def _click(
        self,
        page: Page,
        selector: Optional[str],
        x: Optional[float],
        y: Optional[float],
        click_count: int,
        button: str = "left",
    ) -> None:
        """Shared click implementation."""
        try:
            point = await self._resolve_point(page=page, selector=selector, x=x, y=y)
            await self.move_smoothly(page=page, x=point[0], y=point[1])
            await page.mouse.click(
                point[0],
                point[1],
                button=button,
                click_count=click_count,
            )
        except Exception as exc:
            raise BrowserError(
                f"Failed to click mouse: {exc}",
                code="BROWSER_MOUSE_CLICK_FAILED",
            ) from exc

    async def _resolve_point(
        self,
        page: Page,
        selector: Optional[str],
        x: Optional[float],
        y: Optional[float],
    ) -> Point:
        """Resolve a selector center or explicit coordinates to a point."""
        if selector is not None:
            box = await page.locator(selector).bounding_box()
            if box is None:
                raise BrowserError(
                    f"Element has no visible bounding box: {selector}",
                    code="BROWSER_MOUSE_TARGET_NOT_VISIBLE",
                )
            return (
                box["x"] + box["width"] / 2,
                box["y"] + box["height"] / 2,
            )

        if x is None or y is None:
            raise BrowserError(
                "Mouse action requires either selector or both x and y",
                code="BROWSER_MOUSE_TARGET_REQUIRED",
            )

        return (x, y)


__all__ = [
    "MouseController",
    "MouseControllerConfig",
    "Point",
]
