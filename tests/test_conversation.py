"""
Tests for per-provider conversation state.

Conversation continuity is what makes multi-round debate possible: round
two asks a provider to defend a claim from round one, which only works if
the provider's thread was never restarted. These tests pin the bookkeeping
that tracks it, and the default that preserves it.
"""

from __future__ import annotations

from src.workers.base_worker import WorkerQuery
from src.workers.conversation import ConversationState


class TestTurnTracking:
    """Recording exchanges."""

    def test_starts_fresh(self) -> None:
        """A new conversation has no turns."""
        state = ConversationState(provider="chatgpt")
        assert state.is_fresh is True
        assert state.turns == 0

    def test_records_turns_and_context(self) -> None:
        """Each turn increments the count and accumulates context."""
        state = ConversationState(provider="chatgpt")
        state.record_turn("hello", "hi there")

        assert state.turns == 1
        assert state.is_fresh is False
        assert state.approx_context_chars == len("hello") + len("hi there")
        assert state.last_prompt == "hello"
        assert state.last_answer == "hi there"
        assert state.last_prompt_at is not None

    def test_context_accumulates_across_turns(self) -> None:
        """Context grows monotonically, which is what triggers resets."""
        state = ConversationState(provider="claude")
        state.record_turn("a", "b")
        first = state.approx_context_chars
        state.record_turn("cc", "dd")

        assert state.approx_context_chars > first
        assert state.turns == 2


class TestReset:
    """Clearing history."""

    def test_reset_clears_history_but_counts_resets(self) -> None:
        """History is dropped; the reset counter survives for diagnostics."""
        state = ConversationState(provider="gemini")
        state.record_turn("q", "a")
        state.reset()

        assert state.turns == 0
        assert state.approx_context_chars == 0
        assert state.last_answer is None
        assert state.conversation_id is None
        assert state.resets == 1

    def test_repeated_resets_accumulate(self) -> None:
        """The reset counter is cumulative."""
        state = ConversationState(provider="grok")
        state.reset()
        state.reset()
        assert state.resets == 2


class TestContextLimit:
    """Automatic reset thresholds."""

    def test_needs_reset_when_over_limit(self) -> None:
        """Exceeding the budget requests a reset."""
        state = ConversationState(provider="chatgpt")
        state.record_turn("x" * 60, "y" * 60)
        assert state.needs_reset(100) is True

    def test_does_not_need_reset_under_limit(self) -> None:
        """Staying under the budget preserves the conversation."""
        state = ConversationState(provider="chatgpt")
        state.record_turn("x", "y")
        assert state.needs_reset(100) is False

    def test_zero_limit_disables_check(self) -> None:
        """A non-positive budget never forces a reset."""
        state = ConversationState(provider="chatgpt")
        state.record_turn("x" * 1000, "y" * 1000)
        assert state.needs_reset(0) is False
        assert state.needs_reset(-1) is False


class TestQueryDefault:
    """The default that preserves context."""

    def test_new_conversation_defaults_to_false(self) -> None:
        """
        Persistent sessions must continue existing conversations. If this
        default flips back to True, every request silently reloads the
        provider and debate loses its memory.
        """
        assert WorkerQuery(prompt="q").new_conversation is False

    def test_can_be_requested_explicitly(self) -> None:
        """A fresh thread is still available when needed."""
        assert (
            WorkerQuery(prompt="q", new_conversation=True).new_conversation
            is True
        )
