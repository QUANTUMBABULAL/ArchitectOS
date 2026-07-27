"""
Tests for provider authentication state.

The property that matters most: an unauthenticated provider must never be
treated as a broken one. Restarting a signed-out provider destroys the
page the user needs in order to sign in, so the state machine has to keep
"needs a human" and "needs a restart" strictly apart.
"""

from __future__ import annotations

import pytest

from src.workers.auth import (
    AuthState,
    AuthStatus,
    expiry_notice,
    login_prompt,
)


class TestStateSemantics:
    """What each state permits."""

    @pytest.mark.parametrize(
        "state",
        [AuthState.LOGIN_REQUIRED, AuthState.CAPTCHA_REQUIRED],
    )
    def test_human_states_need_a_person(self, state: AuthState) -> None:
        """Sign-in and challenge states require manual action."""
        assert state.needs_human is True

    @pytest.mark.parametrize(
        "state",
        [
            AuthState.READY,
            AuthState.OFFLINE,
            AuthState.RECOVERING,
            AuthState.UNKNOWN,
        ],
    )
    def test_other_states_do_not(self, state: AuthState) -> None:
        """Remaining states are not blocked on a person."""
        assert state.needs_human is False

    def test_only_offline_is_recoverable(self) -> None:
        """
        This is the load-bearing rule. If LOGIN_REQUIRED were ever
        recoverable, the health monitor would restart the tab the user is
        signing in on, and sign-in could never complete.
        """
        assert AuthState.OFFLINE.is_recoverable is True
        assert AuthState.LOGIN_REQUIRED.is_recoverable is False
        assert AuthState.CAPTCHA_REQUIRED.is_recoverable is False
        assert AuthState.READY.is_recoverable is False

    def test_only_ready_can_dispatch(self) -> None:
        """No prompt reaches a provider that is not signed in."""
        assert AuthState.READY.can_dispatch is True
        for state in (
            AuthState.LOGIN_REQUIRED,
            AuthState.CAPTCHA_REQUIRED,
            AuthState.OFFLINE,
            AuthState.RECOVERING,
            AuthState.UNKNOWN,
        ):
            assert state.can_dispatch is False

    def test_required_states_all_exist(self) -> None:
        """The five specified states are present and named exactly."""
        names = {state.value for state in AuthState}
        assert {
            "READY",
            "LOGIN_REQUIRED",
            "CAPTCHA_REQUIRED",
            "OFFLINE",
            "RECOVERING",
        } <= names


class TestAuthStatus:
    """The status snapshot."""

    def test_ready_flag_tracks_state(self) -> None:
        """is_ready reflects the underlying state."""
        assert AuthStatus(provider="grok", state=AuthState.READY).is_ready
        assert not AuthStatus(
            provider="grok", state=AuthState.LOGIN_REQUIRED
        ).is_ready

    def test_describe_includes_provider_and_state(self) -> None:
        """The log line identifies both provider and state."""
        described = AuthStatus(
            provider="grok",
            state=AuthState.LOGIN_REQUIRED,
            detail="Sign-in screen displayed",
        ).describe()

        assert "grok" in described
        assert "LOGIN_REQUIRED" in described
        assert "Sign-in screen displayed" in described

    def test_timestamp_is_populated(self) -> None:
        """Every snapshot records when it was taken."""
        assert AuthStatus(provider="grok", state=AuthState.READY).checked_at


class TestOperatorMessages:
    """Text shown to the user."""

    def test_login_prompt_matches_required_wording(self) -> None:
        """The first-time prompt names the provider explicitly."""
        message = login_prompt("Grok", "https://grok.com/")
        assert "Please log into Grok manually." in message
        assert "https://grok.com/" in message

    def test_login_prompt_states_credentials_are_not_stored(self) -> None:
        """
        The prompt reassures the user where their credentials go, which
        matters because the app is asking them to type a password nearby.
        """
        message = login_prompt("Grok", "https://grok.com/")
        assert "never seen or stored" in message
        assert "browser profile" in message

    def test_expiry_notice_matches_required_wording(self) -> None:
        """The expiry notice names the provider and reassures on scope."""
        message = expiry_notice("Grok")
        assert "Grok session expired." in message
        assert "Please log in again" in message
        assert "Other providers are unaffected" in message

    def test_messages_contain_no_credential_fields(self) -> None:
        """
        Neither message ever asks for a credential through ArchitectOS.
        Sign-in happens only in the browser.
        """
        for message in (
            login_prompt("Grok", "https://grok.com/"),
            expiry_notice("Grok"),
        ):
            lowered = message.lower()
            assert "enter your password" not in lowered
            assert "username:" not in lowered
            assert "api key" not in lowered
