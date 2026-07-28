"""
Tests for the engine event bus.

The bus sits between the engine and any attached interface, so its
failure modes are the dangerous kind: a slow UI client must never be able
to delay or break a research run. These tests pin that isolation.
"""

from __future__ import annotations

import asyncio

from src.events import (
    Event,
    EventBus,
    EventEmitter,
    EventType,
    WorkerPhase,
    provider_event,
    research_event,
)


class TestSerialization:
    """Wire format shared with the TypeScript client."""

    def test_flat_shape_with_discriminator(self) -> None:
        """Events serialize flat with a `type` field for narrowing."""
        body = provider_event(
            EventType.PROVIDER_STARTED,
            "chatgpt",
            phase=WorkerPhase.THINKING,
            promptChars=42,
        ).to_dict()

        assert body["type"] == "ProviderStarted"
        assert body["provider"] == "chatgpt"
        assert body["phase"] == "Thinking"
        assert body["promptChars"] == 42
        assert "eventId" in body
        assert "timestamp" in body

    def test_research_scope(self) -> None:
        """Research events carry their session id."""
        body = research_event(
            EventType.RESEARCH_STARTED, "abc123", question="why?"
        ).to_dict()

        assert body["researchId"] == "abc123"
        assert body["question"] == "why?"

    def test_absent_scope_is_omitted(self) -> None:
        """Unset scope fields do not appear, keeping frames small."""
        body = Event(type=EventType.ENGINE_READY).to_dict()
        assert "provider" not in body
        assert "researchId" not in body


class TestFanOut:
    """Delivery to attached interfaces."""

    def test_every_subscriber_receives(self) -> None:
        """All subscribers see a published event."""

        async def scenario() -> list[int]:
            bus = EventBus()
            first = bus.subscribe(replay=False)
            second = bus.subscribe(replay=False)
            bus.publish(Event(type=EventType.ENGINE_READY))

            async def take(sub) -> int:
                async for _ in sub:
                    return 1
                return 0

            # gather returns a list, not a tuple.
            return await asyncio.gather(take(first), take(second))

        assert asyncio.run(scenario()) == [1, 1]

    def test_publish_with_no_subscribers_is_safe(self) -> None:
        """The engine runs identically with no interface attached."""
        bus = EventBus()
        bus.publish(Event(type=EventType.ENGINE_READY))
        assert bus.subscriber_count == 0

    def test_unsubscribe_detaches(self) -> None:
        """A detached subscriber stops counting."""
        bus = EventBus()
        subscription = bus.subscribe(replay=False)
        assert bus.subscriber_count == 1

        bus.unsubscribe(subscription)
        assert bus.subscriber_count == 0
        assert subscription.is_closed is True


class TestSlowClientIsolation:
    """A slow interface must not affect the engine."""

    def test_publish_never_blocks_when_queue_is_full(self) -> None:
        """
        The load-bearing property. A subscriber that never reads must not
        apply backpressure: publishing stays synchronous and fast, and the
        subscriber loses its oldest events instead.
        """
        bus = EventBus(queue_size=4)
        subscription = bus.subscribe(replay=False)

        for _ in range(50):
            bus.publish(Event(type=EventType.LOG))

        assert subscription.dropped > 0
        assert bus.subscriber_count == 1

    def test_publish_survives_a_broken_subscriber(self) -> None:
        """A closed subscription is reaped rather than raising."""
        bus = EventBus()
        subscription = bus.subscribe(replay=False)
        subscription.close()

        bus.publish(Event(type=EventType.LOG))
        assert bus.subscriber_count == 0


class TestReplay:
    """History for interfaces that connect mid-session."""

    def test_late_subscriber_receives_history(self) -> None:
        """A client attaching mid-run can rebuild current state."""
        bus = EventBus()
        bus.publish(Event(type=EventType.ENGINE_READY))
        bus.publish(Event(type=EventType.RESEARCH_STARTED))

        subscription = bus.subscribe(replay=True)

        async def drain() -> int:
            count = 0
            async for _ in subscription:
                count += 1
                if count == 2:
                    return count
            return count

        assert asyncio.run(drain()) == 2

    def test_replay_can_be_skipped(self) -> None:
        """Replay is opt-out for clients that only want new events."""
        bus = EventBus()
        bus.publish(Event(type=EventType.ENGINE_READY))
        assert bus.subscribe(replay=False)._queue.empty()  # type: ignore[attr-defined]

    def test_replay_is_bounded(self) -> None:
        """History cannot grow without limit."""
        bus = EventBus(replay_size=5)
        for _ in range(20):
            bus.publish(Event(type=EventType.LOG))
        assert len(bus.replay()) == 5

    def test_clear_replay(self) -> None:
        """History can be dropped between sessions."""
        bus = EventBus()
        bus.publish(Event(type=EventType.LOG))
        bus.clear_replay()
        assert bus.replay() == []


class TestEmitter:
    """The facade engine components depend on."""

    def test_disabled_emitter_is_a_noop(self) -> None:
        """
        With no bus, emission does nothing. This is what allows the engine
        to be constructed and behave identically with no UI attached.
        """
        emitter = EventEmitter(None)
        assert emitter.enabled is False
        emitter.provider(EventType.PROVIDER_STARTED, "chatgpt")
        emitter.emit(Event(type=EventType.LOG))

    def test_enabled_emitter_publishes(self) -> None:
        """With a bus attached, events reach it."""
        bus = EventBus()
        emitter = EventEmitter(bus)

        assert emitter.enabled is True
        emitter.provider(
            EventType.PROVIDER_FINISHED,
            "gemini",
            phase=WorkerPhase.FINISHED,
        )

        published = bus.replay()
        assert len(published) == 1
        assert published[0].provider == "gemini"
        assert published[0].payload["phase"] == "Finished"


class TestPhases:
    """Worker phases shown on cards."""

    def test_all_phases_present(self) -> None:
        """The UI relies on this exact set."""
        assert {phase.value for phase in WorkerPhase} == {
            "Idle",
            "Waiting",
            "Thinking",
            "Generating",
            "Finished",
            "Failed",
            "Blocked",
        }
