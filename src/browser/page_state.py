"""
Page stabilization and navigation state detection.

The failure this module exists to prevent: a worker begins searching for a
prompt input while the page is still redirecting. Navigating with
``domcontentloaded`` resolves on the *first* document, but a sign-in flow
may then bounce through several more. The selector search races the
redirects, never settles, and eventually reports a misleading "composer
not found" timeout — when the real cause was that the page never stopped
moving.

Stabilization therefore happens before any selector is looked for:

1. ``document.readyState`` reaches ``complete``.
2. The URL stops changing for a configured settle window.
3. Network activity goes idle, when the page allows it.

Only then is the settled page classified, and only an INPUT_READY page may
receive a prompt.

The decision logic lives in :class:`NavigationTracker`, which is pure and
takes URL samples rather than a browser. That keeps loop detection and
settle logic testable without Playwright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Pattern, Sequence


class PageState(str, Enum):
    """
    Where a provider page is in its load lifecycle.

    Attributes:
        UNKNOWN: Not yet inspected.
        LOADING: Document is still loading.
        REDIRECTING: The URL is still changing.
        RELOAD_LOOP: The page is cycling and will not settle.
        LOGIN_PAGE: A settled sign-in screen.
        CHALLENGE_PAGE: A settled human-verification screen.
        CHAT_READY: The chat interface loaded, but no usable input was
            found yet.
        INPUT_READY: A prompt input is present and usable.
        ERROR: Stabilization failed for a reason none of the above covers.
    """

    UNKNOWN = "UNKNOWN"
    LOADING = "LOADING"
    REDIRECTING = "REDIRECTING"
    RELOAD_LOOP = "RELOAD_LOOP"
    LOGIN_PAGE = "LOGIN_PAGE"
    CHALLENGE_PAGE = "CHALLENGE_PAGE"
    CHAT_READY = "CHAT_READY"
    INPUT_READY = "INPUT_READY"
    ERROR = "ERROR"

    @property
    def can_prompt(self) -> bool:
        """
        Return whether a prompt may be submitted in this state.

        Returns:
            True only for INPUT_READY.
        """
        return self is PageState.INPUT_READY

    @property
    def is_settled(self) -> bool:
        """
        Return whether the page has stopped navigating.

        Returns:
            True when the page reached a terminal state.
        """
        return self in {
            PageState.LOGIN_PAGE,
            PageState.CHALLENGE_PAGE,
            PageState.CHAT_READY,
            PageState.INPUT_READY,
            PageState.RELOAD_LOOP,
            PageState.ERROR,
        }

    @property
    def needs_human(self) -> bool:
        """
        Return whether the state requires manual intervention.

        Returns:
            True for sign-in and verification pages.
        """
        return self in {PageState.LOGIN_PAGE, PageState.CHALLENGE_PAGE}

    @property
    def label(self) -> str:
        """
        Return an operator-facing label for the dashboard.

        Returns:
            Short human-readable status.
        """
        return {
            PageState.UNKNOWN: "UNKNOWN",
            PageState.LOADING: "LOADING...",
            PageState.REDIRECTING: "REDIRECTING...",
            PageState.RELOAD_LOOP: "RELOAD LOOP",
            PageState.LOGIN_PAGE: "LOGIN_REQUIRED",
            PageState.CHALLENGE_PAGE: "CAPTCHA_REQUIRED",
            PageState.CHAT_READY: "WAITING FOR INPUT",
            PageState.INPUT_READY: "READY",
            PageState.ERROR: "ERROR",
        }[self]


class StabilizationFailure(str, Enum):
    """
    Why a page failed to reach a usable state.

    Attributes:
        RELOAD_LOOP: The page kept navigating.
        AUTH_EXPIRED: A sign-in screen appeared.
        CHALLENGE: A human-verification screen appeared.
        NO_INPUT: The page settled but exposed no usable prompt input.
        NAVIGATION_TIMEOUT: The page never finished loading.
        NETWORK: The page reported a network-level error.
    """

    RELOAD_LOOP = "reload_loop"
    AUTH_EXPIRED = "authentication_expired"
    CHALLENGE = "human_verification_required"
    NO_INPUT = "no_prompt_input_found"
    NAVIGATION_TIMEOUT = "navigation_timeout"
    NETWORK = "network_error"


@dataclass(frozen=True, slots=True)
class StabilizationResult:
    """
    Outcome of waiting for a page to settle.

    Attributes:
        state: Final page state.
        url: URL at the moment of the verdict.
        title: Page title, when it could be read.
        failure: Structured failure cause, when not usable.
        reason: Human-readable explanation.
        url_changes: How many distinct URLs were observed.
        elapsed_seconds: Time spent stabilizing.
        matched_selector: Selector that located the prompt input.
    """

    state: PageState
    url: str = ""
    title: str = ""
    failure: Optional[StabilizationFailure] = None
    reason: str = ""
    url_changes: int = 0
    elapsed_seconds: float = 0.0
    matched_selector: Optional[str] = None

    @property
    def ok(self) -> bool:
        """
        Return whether the page is ready to receive a prompt.

        Returns:
            True when the state is INPUT_READY.
        """
        return self.state.can_prompt

    def describe(self) -> str:
        """
        Render a one-line summary for logs.

        Returns:
            Human-readable description.
        """
        parts = [
            f"state={self.state.value}",
            f"url={self.url or 'unknown'}",
            f"changes={self.url_changes}",
            f"elapsed={self.elapsed_seconds:.1f}s",
        ]
        if self.matched_selector:
            parts.append(f"input={self.matched_selector!r}")
        if self.failure:
            parts.append(f"failure={self.failure.value}")
        return " ".join(parts)


@dataclass
class NavigationTracker:
    """
    Records URL samples and decides when a page has settled.

    Pure decision logic, deliberately free of any browser dependency so
    settle and loop detection can be tested directly.

    A page is settled once the most recent samples share one URL across
    the settle window. A reload loop is declared when the number of URL
    transitions exceeds the allowed maximum, or when the page keeps
    returning to a URL it already left — the signature of a redirect
    bounce such as an expired session cycling through a sign-in host.

    Attributes:
        settle_seconds: How long the URL must hold steady.
        max_url_changes: Transitions tolerated before declaring a loop.
        max_revisits: How many times a URL may be returned to.
        samples: Recorded (timestamp, url) pairs.
    """

    settle_seconds: float = 2.0
    max_url_changes: int = 8
    max_revisits: int = 3
    samples: list[tuple[float, str]] = field(default_factory=list)

    def record(self, timestamp: float, url: str) -> None:
        """
        Record one observation of the page URL.

        Consecutive identical URLs are kept, since the settle check needs
        their timestamps.

        Args:
            timestamp: Monotonic time of the observation.
            url: Observed URL.
        """
        self.samples.append((timestamp, url))

    @property
    def current_url(self) -> str:
        """
        Return the most recently observed URL.

        Returns:
            Latest URL, or an empty string when nothing was recorded.
        """
        return self.samples[-1][1] if self.samples else ""

    @property
    def transitions(self) -> int:
        """
        Return how many times the URL changed.

        Returns:
            Count of consecutive differing samples.
        """
        return sum(
            1
            for previous, current in zip(self.samples, self.samples[1:])
            if previous[1] != current[1]
        )

    @property
    def distinct_urls(self) -> int:
        """
        Return how many distinct URLs were seen.

        Returns:
            Number of unique URLs.
        """
        return len({url for _, url in self.samples})

    def revisit_count(self) -> int:
        """
        Return the highest number of separate visits to any one URL.

        A URL visited, left, and returned to indicates a bounce rather
        than ordinary forward navigation.

        Returns:
            Maximum visit count across all URLs.
        """
        visits: dict[str, int] = {}
        previous: Optional[str] = None

        for _, url in self.samples:
            if url != previous:
                visits[url] = visits.get(url, 0) + 1
                previous = url

        return max(visits.values(), default=0)

    def is_settled(self, now: float) -> bool:
        """
        Report whether the URL has held steady for the settle window.

        Args:
            now: Current monotonic time.

        Returns:
            True when every sample within the window shares one URL and
            the window is fully covered.
        """
        if not self.samples:
            return False

        current = self.current_url
        window_start = now - self.settle_seconds

        # The window must be covered by observation, otherwise a page
        # sampled once would look settled immediately.
        if self.samples[0][0] > window_start:
            return False

        return all(
            url == current
            for timestamp, url in self.samples
            if timestamp >= window_start
        )

    def is_looping(self) -> bool:
        """
        Report whether the page appears to be cycling.

        Returns:
            True when transitions or revisits exceed their limits.
        """
        return (
            self.transitions > self.max_url_changes
            or self.revisit_count() > self.max_revisits
        )

    def loop_summary(self) -> str:
        """
        Describe the observed navigation pattern.

        Returns:
            Human-readable summary naming the URLs involved.
        """
        ordered: list[str] = []
        for _, url in self.samples:
            if not ordered or ordered[-1] != url:
                ordered.append(url)

        trail = " -> ".join(ordered[-6:])
        return (
            f"{self.transitions} navigation(s) across "
            f"{self.distinct_urls} URL(s); recent trail: {trail}"
        )


def compile_patterns(patterns: Sequence[str]) -> tuple[Pattern[str], ...]:
    """
    Compile URL patterns for case-insensitive matching.

    Args:
        patterns: Regular expression sources.

    Returns:
        Compiled patterns, skipping any that fail to compile.
    """
    compiled: list[Pattern[str]] = []
    for source in patterns:
        try:
            compiled.append(re.compile(source, re.IGNORECASE))
        except re.error:
            continue
    return tuple(compiled)


def matches_any(url: str, patterns: Sequence[Pattern[str]]) -> bool:
    """
    Report whether a URL matches any pattern.

    Args:
        url: URL to test.
        patterns: Compiled patterns.

    Returns:
        True when at least one pattern matches.
    """
    return any(pattern.search(url or "") for pattern in patterns)


# URL fragments that indicate a sign-in flow across common providers.
# Kept generic: provider-specific additions belong in the site config.
DEFAULT_LOGIN_URL_PATTERNS: tuple[str, ...] = (
    r"accounts\.google\.com",
    r"/auth/login",
    r"/login\b",
    r"/signin\b",
    r"/sign-in\b",
    r"/oauth",
    r"ServiceLogin",
)

# URL fragments that indicate a bot-detection interstitial.
DEFAULT_CHALLENGE_URL_PATTERNS: tuple[str, ...] = (
    r"/cdn-cgi/challenge",
    r"challenges\.cloudflare\.com",
    r"/recaptcha/",
    r"/sorry/index",
)

# Prompt input candidates tried in order. Multiple selectors are required
# because provider markup differs and changes without notice; relying on
# one selector turns a cosmetic redesign into an outage.
DEFAULT_INPUT_SELECTORS: tuple[str, ...] = (
    "#prompt-textarea",
    'div[contenteditable="true"][role="textbox"]',
    'div[contenteditable="true"]',
    'textarea[placeholder]',
    "textarea:not([readonly]):not([disabled])",
    '[role="textbox"]',
)


__all__ = [
    "DEFAULT_CHALLENGE_URL_PATTERNS",
    "DEFAULT_INPUT_SELECTORS",
    "DEFAULT_LOGIN_URL_PATTERNS",
    "NavigationTracker",
    "PageState",
    "StabilizationFailure",
    "StabilizationResult",
    "compile_patterns",
    "matches_any",
]
