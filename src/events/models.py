"""
Event vocabulary shared between the engine and any connected interface.

These events are the entire contract between the Python engine and the
desktop UI. The UI never reaches into engine internals and never sees a
browser; it renders what these events describe. That keeps Chrome an
implementation detail and lets the engine change without breaking clients.

Every event serializes to a flat JSON object with a ``type`` discriminator,
so a TypeScript client can narrow on it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class EventType(str, Enum):
    """
    Every event the engine can publish.

    Provider events describe one AI provider's progress. Research events
    describe a whole session. System events describe the engine itself.
    """

    # Engine lifecycle
    ENGINE_READY = "EngineReady"
    ENGINE_SHUTDOWN = "EngineShutdown"
    LOG = "Log"

    # Provider lifecycle
    PROVIDER_REGISTERED = "ProviderRegistered"
    PROVIDER_STARTED = "ProviderStarted"
    PROVIDER_TYPING = "ProviderTyping"
    PROVIDER_WAITING = "ProviderWaiting"
    PROVIDER_STREAMING = "ProviderStreaming"
    PROVIDER_FINISHED = "ProviderFinished"
    PROVIDER_ERROR = "ProviderError"
    PROVIDER_LOGIN_REQUIRED = "ProviderLoginRequired"
    PROVIDER_CAPTCHA_REQUIRED = "ProviderCaptchaRequired"
    PROVIDER_STATE_CHANGED = "ProviderStateChanged"

    # Research lifecycle
    RESEARCH_STARTED = "ResearchStarted"
    RESEARCH_PROGRESS = "ResearchProgress"
    RESEARCH_ROUND_STARTED = "ResearchRoundStarted"
    RESEARCH_FINISHED = "ResearchFinished"
    RESEARCH_FAILED = "ResearchFailed"

    # Consensus
    CONSENSUS_STARTED = "ConsensusStarted"
    CONSENSUS_UPDATED = "ConsensusUpdated"
    CONTRADICTION_DETECTED = "ContradictionDetected"

    # Assistant output
    ASSISTANT_MESSAGE = "AssistantMessage"
    ASSISTANT_TOKEN = "AssistantToken"


class WorkerPhase(str, Enum):
    """
    Coarse phase shown on a worker card.

    Deliberately smaller than the internal state machines: the UI needs a
    label a person can read at a glance, not the full lifecycle.
    """

    IDLE = "Idle"
    WAITING = "Waiting"
    THINKING = "Thinking"
    GENERATING = "Generating"
    FINISHED = "Finished"
    FAILED = "Failed"
    BLOCKED = "Blocked"


@dataclass(frozen=True, slots=True)
class Event:
    """
    One published event.

    Attributes:
        type: Event discriminator.
        payload: Event-specific fields. Always JSON-serializable.
        event_id: Unique identifier, useful for client-side deduplication.
        timestamp: When the event was created, in UTC.
        provider: Provider the event concerns, when applicable.
        research_id: Research session the event belongs to, when
            applicable.
    """

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provider: Optional[str] = None
    research_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the event for transport.

        Returns:
            Flat JSON-compatible dictionary with a ``type`` discriminator.
        """
        body: dict[str, Any] = {
            "type": self.type.value,
            "eventId": self.event_id,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.provider is not None:
            body["provider"] = self.provider
        if self.research_id is not None:
            body["researchId"] = self.research_id
        body.update(self.payload)
        return body


# ---------------------------------------------------------------------------
# Constructors
#
# Helpers rather than subclasses: callers get argument checking and the
# transport keeps one flat shape.
# ---------------------------------------------------------------------------


def provider_event(
    type_: EventType,
    provider: str,
    phase: Optional[WorkerPhase] = None,
    **payload: Any,
) -> Event:
    """
    Build a provider-scoped event.

    Args:
        type_: Event type.
        provider: Provider name.
        phase: Optional coarse phase for the worker card.
        **payload: Additional event fields.

    Returns:
        Constructed event.
    """
    body = dict(payload)
    if phase is not None:
        body["phase"] = phase.value
    return Event(type=type_, provider=provider, payload=body)


def research_event(
    type_: EventType,
    research_id: str,
    **payload: Any,
) -> Event:
    """
    Build a research-scoped event.

    Args:
        type_: Event type.
        research_id: Research session identifier.
        **payload: Additional event fields.

    Returns:
        Constructed event.
    """
    return Event(type=type_, research_id=research_id, payload=dict(payload))


def log_event(level: str, message: str, source: str = "") -> Event:
    """
    Build a log event mirroring a terminal log line.

    The terminal remains the authoritative log destination; this only
    forwards a copy so the UI can show a developer pane.

    Args:
        level: Log level name.
        message: Log message.
        source: Originating logger name.

    Returns:
        Constructed event.
    """
    return Event(
        type=EventType.LOG,
        payload={"level": level, "message": message, "source": source},
    )


__all__ = [
    "Event",
    "EventType",
    "WorkerPhase",
    "log_event",
    "provider_event",
    "research_event",
]
