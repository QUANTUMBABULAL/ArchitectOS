"""
Events package: the seam between the engine and any interface.

The engine publishes events; interfaces subscribe. Nothing in the engine
depends on an interface existing, and the terminal remains the
authoritative log destination whether or not one is attached.
"""

from .bus import (
    EventBus,
    EventEmitter,
    Subscription,
    get_emitter,
    get_event_bus,
)
from .models import (
    Event,
    EventType,
    WorkerPhase,
    log_event,
    provider_event,
    research_event,
)

__all__ = [
    "Event",
    "EventBus",
    "EventEmitter",
    "EventType",
    "Subscription",
    "WorkerPhase",
    "get_emitter",
    "get_event_bus",
    "log_event",
    "provider_event",
    "research_event",
]
