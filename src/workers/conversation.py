"""
Per-provider conversation state.

A persistent research session keeps one long-running conversation per
provider rather than starting a fresh thread for every request. That is
what allows follow-up prompts to reference earlier answers, which in turn
is what makes multi-round debate possible: round two asks a provider to
defend a claim it made in round one, and the provider must still remember
making it.

This module holds the bookkeeping only. Deciding *when* a conversation
must be reset — an explicit user request, or a context limit — is policy
that lives with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ConversationState:
    """
    Mutable record of one provider's ongoing conversation.

    Attributes:
        provider: Provider name.
        conversation_id: Provider-assigned conversation identifier when it
            can be recovered from the page URL, otherwise None.
        turns: Number of completed prompt/response exchanges.
        started_at: When the conversation began.
        last_prompt_at: When the most recent prompt was submitted.
        last_prompt: Most recent prompt text.
        last_answer: Most recent answer text.
        approx_context_chars: Running total of prompt and answer characters
            exchanged. Used as a cheap proxy for context pressure, since
            provider token counts are not exposed to the browser.
        resets: How many times this conversation has been reset.
    """

    provider: str
    conversation_id: Optional[str] = None
    turns: int = 0
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_prompt_at: Optional[datetime] = None
    last_prompt: Optional[str] = None
    last_answer: Optional[str] = None
    approx_context_chars: int = 0
    resets: int = 0

    @property
    def is_fresh(self) -> bool:
        """
        Return whether the conversation has no completed turns yet.

        Returns:
            True when nothing has been exchanged.
        """
        return self.turns == 0

    def record_turn(self, prompt: str, answer: str) -> None:
        """
        Record one completed exchange.

        Args:
            prompt: Prompt that was submitted.
            answer: Answer that was received.
        """
        self.turns += 1
        self.last_prompt = prompt
        self.last_answer = answer
        self.last_prompt_at = datetime.now(timezone.utc)
        self.approx_context_chars += len(prompt) + len(answer)

    def reset(self) -> None:
        """Clear conversation history, keeping cumulative reset count."""
        self.conversation_id = None
        self.turns = 0
        self.started_at = datetime.now(timezone.utc)
        self.last_prompt_at = None
        self.last_prompt = None
        self.last_answer = None
        self.approx_context_chars = 0
        self.resets += 1

    def needs_reset(self, context_char_limit: int) -> bool:
        """
        Report whether accumulated context has grown past a limit.

        Args:
            context_char_limit: Character budget for the conversation. A
                non-positive value disables the check.

        Returns:
            True when the conversation should be reset before reuse.
        """
        if context_char_limit <= 0:
            return False
        return self.approx_context_chars >= context_char_limit

    def describe(self) -> str:
        """
        Render a single-line summary for logs and status output.

        Returns:
            Human-readable description.
        """
        return (
            f"{self.provider}: turns={self.turns} "
            f"context~{self.approx_context_chars}c resets={self.resets}"
        )


__all__ = [
    "ConversationState",
]
