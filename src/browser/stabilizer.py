"""
Waits for a provider page to stop moving before anything touches it.

This is the component that fixes the reload-loop failure. Nothing looks
for a prompt input until :class:`PageStabilizer` reports INPUT_READY, so a
selector search can never race a redirect chain.

The order of checks matters and is deliberate:

1. Detect a reload loop first. A looping page will satisfy no other
   condition, and reporting the loop is more useful than reporting the
   timeout it would otherwise produce.
2. Detect sign-in and verification pages next. Both are *settled* pages —
   they are not failures to load, and reloading them destroys the page the
   user needs.
3. Only then look for an input, trying every configured selector.

Reloading is never performed here. A page that needs a human is left
exactly as it is.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Sequence

from playwright.async_api import Page

from src.logger import get_logger

from .page_state import (
    NavigationTracker,
    PageState,
    StabilizationFailure,
    StabilizationResult,
    compile_patterns,
    matches_any,
)


class PageStabilizer:
    """
    Drives a page to a settled, classified state.

    Configured with selector and URL-pattern sets rather than
    provider-specific logic, so one instance serves every current and
    future provider.
    """

    def __init__(
        self,
        input_selectors: Sequence[str],
        login_selectors: str = "",
        challenge_selectors: str = "",
        login_url_patterns: Sequence[str] = (),
        challenge_url_patterns: Sequence[str] = (),
        settle_seconds: float = 2.0,
        poll_interval_seconds: float = 0.35,
        max_url_changes: int = 8,
        network_idle_seconds: float = 5.0,
        logger_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the stabilizer.

        Args:
            input_selectors: Prompt input candidates, tried in order.
            login_selectors: Selector matching sign-in indicators.
            challenge_selectors: Selector matching verification widgets.
            login_url_patterns: URL patterns indicating a sign-in flow.
            challenge_url_patterns: URL patterns indicating a challenge.
            settle_seconds: How long the URL must hold steady.
            poll_interval_seconds: Delay between observations.
            max_url_changes: Navigations tolerated before declaring a loop.
            network_idle_seconds: How long to allow for network idle. A
                page with long-lived connections never goes idle, so this
                is best-effort rather than required.
            logger_name: Optional logger suffix identifying the provider.
        """
        self._input_selectors = tuple(input_selectors)
        self._login_selectors = login_selectors
        self._challenge_selectors = challenge_selectors
        self._login_urls = compile_patterns(login_url_patterns)
        self._challenge_urls = compile_patterns(challenge_url_patterns)
        self._settle_seconds = settle_seconds
        self._poll = max(0.05, poll_interval_seconds)
        self._max_url_changes = max_url_changes
        self._network_idle = network_idle_seconds
        self._logger = get_logger(
            f"{__name__}.{logger_name}" if logger_name else __name__
        )

    async def stabilize(
        self,
        page: Page,
        timeout_seconds: float,
    ) -> StabilizationResult:
        """
        Wait for the page to settle and classify what it settled on.

        Args:
            page: Page to stabilize.
            timeout_seconds: Overall budget.

        Returns:
            Stabilization result. Never raises for page conditions; the
            caller decides what to do with each outcome.
        """
        started = time.monotonic()
        deadline = started + timeout_seconds
        tracker = NavigationTracker(
            settle_seconds=self._settle_seconds,
            max_url_changes=self._max_url_changes,
        )

        while time.monotonic() < deadline:
            now = time.monotonic()
            url = await self._safe_url(page)
            tracker.record(now, url)

            if tracker.is_looping():
                return self._loop_result(tracker, started, page)

            if not tracker.is_settled(now):
                await asyncio.sleep(self._poll)
                continue

            if not await self._document_complete(page):
                await asyncio.sleep(self._poll)
                continue

            # The page has stopped moving. Classify it before touching it.
            verdict = await self._classify(page, tracker, started)
            if verdict is not None:
                return verdict

            await asyncio.sleep(self._poll)

        return await self._timeout_result(page, tracker, started)

    async def _classify(
        self,
        page: Page,
        tracker: NavigationTracker,
        started: float,
    ) -> Optional[StabilizationResult]:
        """
        Classify a settled page.

        Args:
            page: Page to classify.
            tracker: Navigation history.
            started: Monotonic start time.

        Returns:
            A terminal result, or None to keep waiting for an input to
            appear on an otherwise-loaded chat page.
        """
        url = tracker.current_url

        if await self._challenge_present(page, url):
            return await self._build(
                page,
                tracker,
                started,
                PageState.CHALLENGE_PAGE,
                StabilizationFailure.CHALLENGE,
                "Human verification challenge is displayed",
            )

        if await self._login_present(page, url):
            return await self._build(
                page,
                tracker,
                started,
                PageState.LOGIN_PAGE,
                StabilizationFailure.AUTH_EXPIRED,
                "A sign-in screen is displayed",
            )

        selector = await self._find_input(page)
        if selector is not None:
            result = await self._build(
                page,
                tracker,
                started,
                PageState.INPUT_READY,
                None,
                "Prompt input is available",
            )
            return StabilizationResult(
                state=result.state,
                url=result.url,
                title=result.title,
                failure=None,
                reason=result.reason,
                url_changes=result.url_changes,
                elapsed_seconds=result.elapsed_seconds,
                matched_selector=selector,
            )

        # Settled, authenticated, but no input yet. The interface may
        # still be rendering, so keep waiting rather than failing.
        return None

    async def _find_input(self, page: Page) -> Optional[str]:
        """
        Find a usable prompt input, trying every configured selector.

        Args:
            page: Page to search.

        Returns:
            The first selector that matched a visible, enabled element, or
            None when none matched.
        """
        for selector in self._input_selectors:
            try:
                candidate = page.locator(selector).first
                if not await candidate.is_visible():
                    continue
                if not await candidate.is_enabled():
                    continue
                return selector
            except Exception:
                continue
        return None

    async def _login_present(self, page: Page, url: str) -> bool:
        """
        Report whether a sign-in screen is displayed.

        Args:
            page: Page to inspect.
            url: Current URL.

        Returns:
            True when the URL or the DOM indicates a sign-in flow.
        """
        if matches_any(url, self._login_urls):
            return True
        return await self._selector_visible(page, self._login_selectors)

    async def _challenge_present(self, page: Page, url: str) -> bool:
        """
        Report whether a verification challenge is displayed.

        Args:
            page: Page to inspect.
            url: Current URL.

        Returns:
            True when the URL or the DOM indicates a challenge.
        """
        if matches_any(url, self._challenge_urls):
            return True
        return await self._selector_visible(page, self._challenge_selectors)

    @staticmethod
    async def _selector_visible(page: Page, selector: str) -> bool:
        """
        Report whether a selector matches a visible element.

        Args:
            page: Page to inspect.
            selector: Selector to test. Empty selectors match nothing.

        Returns:
            True when a visible match exists.
        """
        if not selector:
            return False
        try:
            return await page.locator(selector).first.is_visible()
        except Exception:
            return False

    @staticmethod
    async def _document_complete(page: Page) -> bool:
        """
        Report whether the document finished loading.

        Args:
            page: Page to inspect.

        Returns:
            True when readyState is complete.
        """
        try:
            state = await page.evaluate("() => document.readyState")
            return state == "complete"
        except Exception:
            # Evaluation fails while a navigation is in flight, which is
            # itself evidence the page has not settled.
            return False

    async def wait_for_network_idle(self, page: Page) -> bool:
        """
        Best-effort wait for network activity to settle.

        Chat interfaces hold streaming connections open and may never
        reach true idle, so failure here is informational rather than
        fatal.

        Args:
            page: Page to wait on.

        Returns:
            True when the page reported network idle.
        """
        try:
            await page.wait_for_load_state(
                "networkidle", timeout=self._network_idle * 1000
            )
            return True
        except Exception:
            return False

    @staticmethod
    async def _safe_url(page: Page) -> str:
        """
        Read the page URL without raising during navigation.

        Args:
            page: Page to read.

        Returns:
            Current URL, or an empty string when unavailable.
        """
        try:
            return page.url or ""
        except Exception:
            return ""

    @staticmethod
    async def _safe_title(page: Page) -> str:
        """
        Read the page title without raising during navigation.

        Args:
            page: Page to read.

        Returns:
            Page title, or an empty string when unavailable.
        """
        try:
            return await page.title()
        except Exception:
            return ""

    async def _build(
        self,
        page: Page,
        tracker: NavigationTracker,
        started: float,
        state: PageState,
        failure: Optional[StabilizationFailure],
        reason: str,
    ) -> StabilizationResult:
        """
        Assemble a result with page metadata attached.

        Args:
            page: Page being classified.
            tracker: Navigation history.
            started: Monotonic start time.
            state: Final state.
            failure: Structured failure cause, if any.
            reason: Human-readable explanation.

        Returns:
            Stabilization result.
        """
        return StabilizationResult(
            state=state,
            url=tracker.current_url,
            title=await self._safe_title(page),
            failure=failure,
            reason=reason,
            url_changes=tracker.transitions,
            elapsed_seconds=time.monotonic() - started,
        )

    def _loop_result(
        self,
        tracker: NavigationTracker,
        started: float,
        page: Page,
    ) -> StabilizationResult:
        """
        Build the result for a detected reload loop.

        Args:
            tracker: Navigation history.
            started: Monotonic start time.
            page: Page being classified.

        Returns:
            Stabilization result describing the loop.
        """
        url = tracker.current_url
        summary = tracker.loop_summary()

        if matches_any(url, self._login_urls):
            cause = (
                "the session appears to have expired and the page keeps "
                "returning to a sign-in screen"
            )
            failure = StabilizationFailure.AUTH_EXPIRED
        elif matches_any(url, self._challenge_urls):
            cause = "a bot-detection challenge keeps re-issuing"
            failure = StabilizationFailure.CHALLENGE
        else:
            cause = (
                "the page keeps redirecting; this is usually an expired "
                "session, a blocked sign-in, or an unreachable network"
            )
            failure = StabilizationFailure.RELOAD_LOOP

        self._logger.error(
            "Detected reload loop. Reason: %s. %s", cause, summary
        )

        return StabilizationResult(
            state=PageState.RELOAD_LOOP,
            url=url,
            failure=failure,
            reason=f"Detected reload loop — {cause}. {summary}",
            url_changes=tracker.transitions,
            elapsed_seconds=time.monotonic() - started,
        )

    async def _timeout_result(
        self,
        page: Page,
        tracker: NavigationTracker,
        started: float,
    ) -> StabilizationResult:
        """
        Build the result for a page that never became usable.

        Distinguishes "still moving" from "settled but no input", because
        the remedies differ.

        Args:
            page: Page being classified.
            tracker: Navigation history.
            started: Monotonic start time.

        Returns:
            Stabilization result describing the timeout.
        """
        now = time.monotonic()
        settled = tracker.is_settled(now)

        if not settled:
            state = PageState.REDIRECTING
            failure = StabilizationFailure.NAVIGATION_TIMEOUT
            reason = (
                "The page was still navigating when the timeout expired. "
                f"{tracker.loop_summary()}"
            )
        else:
            state = PageState.CHAT_READY
            failure = StabilizationFailure.NO_INPUT
            reason = (
                "The page settled and is signed in, but no prompt input "
                "matched any configured selector. The provider's layout "
                "has probably changed."
            )

        return await self._build(
            page, tracker, started, state, failure, reason
        )


__all__ = [
    "PageStabilizer",
]
