"""
In-process publish/subscribe bus for engine events.

The bus is the seam between the engine and any interface attached to it.
Three properties matter for correctness:

* **Publishing never blocks.** A slow or stalled UI client must not be
  able to delay research. Each subscriber has a bounded queue and the
  slowest ones drop their oldest events rather than applying backpressure
  to the engine.
* **Publishing never raises.** A missing or broken subscriber is not an
  engine failure. Delivery problems are logged and swallowed.
* **The bus is optional.** Components take an emitter that defaults to a
  no-op, so the engine behaves identically with no interface attached and
  the terminal remains the authoritative log destination.

A replay buffer lets a client that connects mid-session reconstruct
current state instead of waiting for the next event.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncIterator, Callable, Optional

from src.logger import get_logger

from .models import Event, EventType, WorkerPhase, provider_event

DEFAULT_QUEUE_SIZE = 512
DEFAULT_REPLAY_SIZE = 300


class Subscription:
    """
    One subscriber's bounded view of the event stream.

    Backed by a fixed-size queue. When a subscriber cannot keep up its
    oldest events are discarded and a counter is incremented, so slow
    clients degrade rather than stalling the engine.
    """

    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        """
        Initialize the subscription.

        Args:
            maxsize: Maximum queued events before dropping the oldest.
        """
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self.dropped = 0

    @property
    def is_closed(self) -> bool:
        """
        Return whether the subscription has been closed.

        Returns:
            True when closed.
        """
        return self._closed

    def offer(self, event: Event) -> None:
        """
        Enqueue an event, dropping the oldest when full.

        Args:
            event: Event to deliver.
        """
        if self._closed:
            return

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
                self.dropped += 1
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                self.dropped += 1

    async def __aiter__(self) -> AsyncIterator[Event]:
        """
        Iterate delivered events until the subscription closes.

        Yields:
            Events in delivery order.
        """
        while not self._closed:
            try:
                yield await self._queue.get()
            except asyncio.CancelledError:
                return

    def close(self) -> None:
        """Close the subscription and release any waiting consumer."""
        self._closed = True


class EventBus:
    """
    Fan-out event bus with a bounded replay buffer.

    One instance is shared by the engine and every attached interface.
    """

    def __init__(
        self,
        replay_size: int = DEFAULT_REPLAY_SIZE,
        queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        """
        Initialize the bus.

        Args:
            replay_size: Events retained for late subscribers.
            queue_size: Per-subscriber queue depth.
        """
        self._subscribers: list[Subscription] = []
        self._replay: deque[Event] = deque(maxlen=replay_size)
        self._queue_size = queue_size
        self._logger = get_logger(__name__)

    @property
    def subscriber_count(self) -> int:
        """
        Return how many interfaces are currently attached.

        Returns:
            Number of open subscriptions.
        """
        return len(self._subscribers)

    def publish(self, event: Event) -> None:
        """
        Publish an event to every subscriber.

        Synchronous, non-blocking, and never raises, so engine code can
        call it from anywhere without defensive handling.

        Args:
            event: Event to publish.
        """
        self._replay.append(event)

        stale: list[Subscription] = []
        for subscription in self._subscribers:
            if subscription.is_closed:
                stale.append(subscription)
                continue
            try:
                subscription.offer(event)
            except Exception as exc:
                self._logger.debug("Event delivery failed: %s", exc)
                stale.append(subscription)

        for subscription in stale:
            self._detach(subscription)

    def subscribe(self, replay: bool = True) -> Subscription:
        """
        Attach a new subscriber.

        Args:
            replay: Whether to seed the subscriber with recent history so a
                client connecting mid-session can rebuild current state.

        Returns:
            Subscription to iterate.
        """
        subscription = Subscription(maxsize=self._queue_size)

        if replay:
            for event in list(self._replay):
                subscription.offer(event)

        self._subscribers.append(subscription)
        self._logger.info(
            "Interface attached to the event bus (%d total)",
            len(self._subscribers),
        )
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """
        Detach a subscriber.

        Args:
            subscription: Subscription to remove.
        """
        subscription.close()
        self._detach(subscription)

    def _detach(self, subscription: Subscription) -> None:
        """
        Remove a subscription from the fan-out list.

        Args:
            subscription: Subscription to remove.
        """
        if subscription in self._subscribers:
            self._subscribers.remove(subscription)
            if subscription.dropped:
                self._logger.warning(
                    "Interface detached after dropping %d event(s); the "
                    "client could not keep up",
                    subscription.dropped,
                )

    def replay(self) -> list[Event]:
        """
        Return the retained event history.

        Returns:
            Recent events, oldest first.
        """
        return list(self._replay)

    def clear_replay(self) -> None:
        """Discard retained history, for example when starting a session."""
        self._replay.clear()


class EventEmitter:
    """
    Narrow, provider-aware facade over the bus.

    Engine components take an emitter rather than the bus so they can be
    constructed with no interface attached: the default emitter publishes
    nowhere and costs a branch. This is what keeps the engine unchanged in
    behaviour whether or not a UI is running.
    """

    def __init__(self, bus: Optional[EventBus] = None) -> None:
        """
        Initialize the emitter.

        Args:
            bus: Optional bus. When omitted, emission is a no-op.
        """
        self._bus = bus

    @property
    def enabled(self) -> bool:
        """
        Return whether events are being published.

        Returns:
            True when a bus is attached.
        """
        return self._bus is not None

    def emit(self, event: Event) -> None:
        """
        Publish an event when a bus is attached.

        Args:
            event: Event to publish.
        """
        if self._bus is not None:
            self._bus.publish(event)

    def provider(
        self,
        type_: EventType,
        provider: str,
        phase: Optional[WorkerPhase] = None,
        **payload: object,
    ) -> None:
        """
        Publish a provider-scoped event.

        Args:
            type_: Event type.
            provider: Provider name.
            phase: Optional coarse worker phase.
            **payload: Additional event fields.
        """
        if self._bus is None:
            return
        self._bus.publish(
            provider_event(type_, provider, phase=phase, **payload)
        )


# A module-level bus is provided for convenience so the server and the
# application can share one instance without threading it through every
# constructor. Components should still accept an emitter for testability.
_GLOBAL_BUS: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """
    Return the process-wide event bus, creating it on first use.

    Returns:
        Shared event bus.
    """
    global _GLOBAL_BUS
    if _GLOBAL_BUS is None:
        _GLOBAL_BUS = EventBus()
    return _GLOBAL_BUS


def get_emitter() -> EventEmitter:
    """
    Return an emitter bound to the process-wide bus.

    Returns:
        Emitter publishing to the shared bus.
    """
    return EventEmitter(get_event_bus())


__all__ = [
    "DEFAULT_QUEUE_SIZE",
    "DEFAULT_REPLAY_SIZE",
    "EventBus",
    "EventEmitter",
    "Subscription",
    "get_emitter",
    "get_event_bus",
]
