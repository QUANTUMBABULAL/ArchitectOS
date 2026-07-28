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
    challenge_prompt,
    expiry_notice,
    login_prompt,
    render_dashboard,
    state_glyph,
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
        """The first-time prompt uses the specified phrasing exactly."""
        message = login_prompt("Claude", "https://claude.ai/new")

        assert "Claude requires authentication." in message
        assert "A browser tab has already been opened." in message
        assert "Please complete login manually." in message
        assert "ArchitectOS is waiting..." in message
        assert "https://claude.ai/new" in message

    def test_challenge_prompt_matches_required_wording(self) -> None:
        """The verification prompt uses the specified phrasing exactly."""
        message = challenge_prompt("Claude")

        assert "Claude requires human verification." in message
        assert "Please complete verification manually." in message
        assert "will automatically continue when" in message

    def test_challenge_prompt_never_offers_to_solve(self) -> None:
        """
        ArchitectOS must not imply it can solve a CAPTCHA. Solving is
        always the user's action.
        """
        lowered = challenge_prompt("Claude").lower()
        assert "solving" not in lowered
        assert "we will solve" not in lowered

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
        No message ever asks for a credential through ArchitectOS.
        Sign-in happens only in the browser.
        """
        for message in (
            login_prompt("Grok", "https://grok.com/"),
            challenge_prompt("Grok"),
            expiry_notice("Grok"),
        ):
            lowered = message.lower()
            assert "enter your password" not in lowered
            assert "username:" not in lowered
            assert "api key" not in lowered


class TestDashboard:
    """The provider status board."""

    @staticmethod
    def _rows() -> list[AuthStatus]:
        """Build the specification's example provider set."""
        return [
            AuthStatus(
                provider="chatgpt",
                state=AuthState.READY,
                display_name="ChatGPT",
            ),
            AuthStatus(
                provider="gemini",
                state=AuthState.READY,
                display_name="Gemini",
            ),
            AuthStatus(
                provider="grok",
                state=AuthState.READY,
                display_name="Grok",
            ),
            AuthStatus(
                provider="claude",
                state=AuthState.LOGIN_REQUIRED,
                display_name="Claude",
            ),
            AuthStatus(
                provider="deepseek",
                state=AuthState.LOGIN_REQUIRED,
                display_name="DeepSeek",
            ),
        ]

    def test_renders_every_provider_with_state(self) -> None:
        """Each provider appears with its state."""
        board = render_dashboard(self._rows())

        assert "Provider Status" in board
        for name in ("ChatGPT", "Gemini", "Grok", "Claude", "DeepSeek"):
            assert name in board
        assert "READY" in board
        assert "LOGIN_REQUIRED" in board

    def test_glyphs_distinguish_ready_from_pending(self) -> None:
        """Ready and pending providers are visually distinct."""
        board = render_dashboard(self._rows())
        assert "🟢 ChatGPT" in board
        assert "🟡 Claude" in board

    def test_glyph_per_state(self) -> None:
        """Every state maps to a glyph, including unknown ones."""
        assert state_glyph(AuthState.READY) == "🟢"
        assert state_glyph(AuthState.LOGIN_REQUIRED) == "🟡"
        assert state_glyph(AuthState.CAPTCHA_REQUIRED) == "🟡"
        assert state_glyph(AuthState.OFFLINE) == "🔴"
        assert state_glyph(AuthState.UNKNOWN) == "⚪"

    def test_handles_empty_provider_set(self) -> None:
        """An empty board renders without raising."""
        assert "no providers" in render_dashboard([]).lower()

    def test_display_name_defaults_to_provider(self) -> None:
        """A snapshot without a display name still renders."""
        board = render_dashboard(
            [AuthStatus(provider="mistral", state=AuthState.READY)]
        )
        assert "mistral" in board
