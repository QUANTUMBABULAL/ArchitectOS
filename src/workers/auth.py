"""
Authentication state for browser-based providers.

ArchitectOS never handles credentials. It does not read them, store them,
or type them. Authentication lives entirely inside the persistent Chrome
profile on disk, created by the user signing in manually once. This module
models what the system is allowed to know: whether a provider currently
appears signed in, and if not, what kind of intervention is needed.

Authentication state is tracked separately from worker lifecycle state on
purpose. A worker can be perfectly healthy — tab open, page responsive —
while being signed out, and the remedies are opposite. A browser problem
is fixed by reloading; an authentication problem is made *worse* by
reloading, because it discards the page the user was about to sign in on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class AuthState(str, Enum):
    """
    Authentication status of one provider.

    Attributes:
        READY: Signed in and able to accept prompts.
        LOGIN_REQUIRED: A sign-in screen is present. Requires a human;
            never resolved by retrying.
        CAPTCHA_REQUIRED: A human-verification challenge is present.
            Requires a human and must not be retried, since repeated
            attempts can harden the challenge.
        OFFLINE: The tab is gone or the page is unreachable. This is the
            only state a browser-level remedy can fix.
        RECOVERING: A recovery attempt is in progress.
        UNKNOWN: Not yet checked.
    """

    READY = "READY"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"
    CAPTCHA_REQUIRED = "CAPTCHA_REQUIRED"
    OFFLINE = "OFFLINE"
    RECOVERING = "RECOVERING"
    UNKNOWN = "UNKNOWN"

    @property
    def needs_human(self) -> bool:
        """
        Return whether resolving this state requires a person.

        Returns:
            True when no automated action can help.
        """
        return self in {
            AuthState.LOGIN_REQUIRED,
            AuthState.CAPTCHA_REQUIRED,
        }

    @property
    def is_recoverable(self) -> bool:
        """
        Return whether a browser-level restart could resolve this state.

        Returns:
            True only for OFFLINE. Restarting an unauthenticated or
            challenged provider destroys the page the user needs.
        """
        return self is AuthState.OFFLINE

    @property
    def can_dispatch(self) -> bool:
        """
        Return whether the provider may receive a prompt.

        Returns:
            True only when signed in and ready.
        """
        return self is AuthState.READY


@dataclass(frozen=True, slots=True)
class AuthStatus:
    """
    Point-in-time authentication snapshot for one provider.

    Attributes:
        provider: Provider name.
        state: Authentication state.
        checked_at: When the check ran.
        detail: Human-readable explanation, shown to the operator.
        action: What the user should do, when action is needed.
    """

    provider: str
    state: AuthState
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    detail: str = ""
    action: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """
        Return whether the provider is signed in and usable.

        Returns:
            True when the state is READY.
        """
        return self.state is AuthState.READY

    def describe(self) -> str:
        """
        Render a single-line summary for logs and status output.

        Returns:
            Human-readable description.
        """
        suffix = f" — {self.detail}" if self.detail else ""
        return f"{self.provider}: {self.state.value}{suffix}"


def login_prompt(display_name: str, url: str) -> str:
    """
    Build the operator instruction for a first-time sign-in.

    Args:
        display_name: Human-readable provider name.
        url: Provider URL the tab is already showing.

    Returns:
        Instruction text.
    """
    return (
        f"Please log into {display_name} manually.\n"
        f"    A tab is already open at {url} in the automation Chrome "
        f"window.\n"
        f"    Sign in there; ArchitectOS will detect it and continue "
        f"automatically.\n"
        f"    Your credentials are never seen or stored by ArchitectOS — "
        f"the session lives only in the browser profile."
    )


def expiry_notice(display_name: str) -> str:
    """
    Build the operator instruction for an expired session.

    Args:
        display_name: Human-readable provider name.

    Returns:
        Notice text.
    """
    return (
        f"{display_name} session expired.\n"
        f"    Please log in again in the automation Chrome window. Other "
        f"providers are unaffected and research continues without it."
    )


__all__ = [
    "AuthState",
    "AuthStatus",
    "expiry_notice",
    "login_prompt",
]
