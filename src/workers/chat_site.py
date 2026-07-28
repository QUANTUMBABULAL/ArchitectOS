"""
Site descriptions for browser-driven AI chat providers.

Every supported provider is a web chat interface with the same shape: a
composer to type into, a send affordance, a streaming indicator, and
assistant message containers. Only the selectors and timings differ.
Capturing those differences as data rather than as code means one worker
implementation serves all providers, and adding a provider is a
configuration change.

Selector stability
------------------
These selectors target third-party web applications that publish no
stability contract and change without notice. They are the single most
likely cause of a provider failing. Each field is therefore overridable
so a broken provider can be repaired without editing code, and the
worker layer treats a provider failure as degradation rather than as an
error (see WorkerManager.dispatch_many).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.browser.page_state import (
    DEFAULT_CHALLENGE_URL_PATTERNS,
    DEFAULT_INPUT_SELECTORS,
    DEFAULT_LOGIN_URL_PATTERNS,
)


class ChatSiteConfig(BaseModel):
    """
    Selectors and timings describing one AI chat website.

    Attributes:
        name: Worker name, used for registration and logging.
        display_name: Human-readable provider name for operator output.
        base_url: URL that opens a fresh conversation.
        composer_selector: Prompt input element.
        send_button_selector: Send affordance. Falls back to pressing
            Enter when absent or not clickable.
        stop_button_selector: Element visible only while the response is
            streaming. Used as the primary completion signal.
        assistant_message_selector: Container matching assistant turns.
        login_wall_selector: Element indicating the provider requires
            sign-in. Detected so the failure is actionable rather than a
            generic timeout.
        challenge_selector: Element indicating a CAPTCHA or bot-detection
            interstitial. Detected separately from a login wall because
            the response differs: the provider is paused for manual
            resolution rather than retried, and research continues with
            the remaining providers.
        capabilities: Capability tags used for consultation routing.
        navigation_timeout_seconds: Timeout for page navigation.
        ready_timeout_seconds: Timeout for the composer to become usable.
        response_timeout_seconds: Maximum wait for one complete answer.
        poll_interval_seconds: Delay between response polling checks.
        stability_checks: Consecutive identical polls required before a
            streamed answer is considered complete.
        paste_threshold_chars: Prompt length above which text is pasted
            rather than typed character by character.
        submit_delay_seconds: Pause after filling the composer before
            submitting, for sites that enable the send button
            asynchronously.
        verified: False for providers whose selectors have not been
            confirmed against the live site. Logged at startup so an
            operator knows which providers are best-effort.
        requires_auth: Whether the provider needs a signed-in session.
            Providers that answer anonymously skip authentication checks
            entirely.
        enabled_by_default: Whether this provider participates unless
            configuration says otherwise. Set False for providers known to
            be unusable through browser automation, so they stay dormant
            without their code being removed.
        disabled_reason: Why the provider ships disabled. Surfaced in
            startup logs so a disabled provider is never a mystery.
    """

    name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    composer_selector: str = Field(min_length=1)
    input_selectors: tuple[str, ...] = Field(default_factory=tuple)
    login_url_patterns: tuple[str, ...] = Field(default_factory=tuple)
    challenge_url_patterns: tuple[str, ...] = Field(default_factory=tuple)
    settle_seconds: float = Field(default=2.0, gt=0)
    max_url_changes: int = Field(default=8, ge=1)

    def prompt_input_selectors(self) -> tuple[str, ...]:
        """
        Return prompt input candidates, most specific first.

        The provider's own composer selector is tried first, then any
        provider-specific alternatives, then generic fallbacks. Relying on
        one selector turns a cosmetic redesign into an outage, so the
        generic tail is always present.

        Returns:
            Ordered, de-duplicated selectors.
        """
        ordered: list[str] = [self.composer_selector]
        ordered.extend(self.input_selectors)
        ordered.extend(DEFAULT_INPUT_SELECTORS)

        seen: list[str] = []
        for selector in ordered:
            if selector and selector not in seen:
                seen.append(selector)
        return tuple(seen)

    def login_urls(self) -> tuple[str, ...]:
        """
        Return URL patterns indicating a sign-in flow.

        Returns:
            Provider-specific patterns followed by shared defaults.
        """
        return tuple(self.login_url_patterns) + DEFAULT_LOGIN_URL_PATTERNS

    def challenge_urls(self) -> tuple[str, ...]:
        """
        Return URL patterns indicating a verification challenge.

        Returns:
            Provider-specific patterns followed by shared defaults.
        """
        return (
            tuple(self.challenge_url_patterns)
            + DEFAULT_CHALLENGE_URL_PATTERNS
        )
    send_button_selector: str = Field(default="")
    stop_button_selector: str = Field(default="")
    assistant_message_selector: str = Field(min_length=1)
    login_wall_selector: str = Field(default="")
    challenge_selector: str = Field(
        default=(
            'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], '
            'iframe[title*="challenge"], iframe[src*="turnstile"], '
            'div#challenge-running, div.cf-challenge, '
            'text=/verify you are human/i'
        )
    )
    capabilities: frozenset[str] = Field(
        default_factory=lambda: frozenset({"general"})
    )
    navigation_timeout_seconds: float = Field(default=45.0, gt=0)
    ready_timeout_seconds: float = Field(default=20.0, gt=0)
    response_timeout_seconds: float = Field(default=180.0, gt=0)
    poll_interval_seconds: float = Field(default=0.5, gt=0)
    stability_checks: int = Field(default=3, ge=1)
    paste_threshold_chars: int = Field(default=400, ge=1)
    submit_delay_seconds: float = Field(default=0.3, ge=0)
    verified: bool = Field(default=False)
    requires_auth: bool = Field(default=True)
    enabled_by_default: bool = Field(default=True)
    disabled_reason: str = Field(default="")


__all__ = [
    "ChatSiteConfig",
]
