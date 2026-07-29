"""
WebSocket and REST bridge between the engine and the desktop interface.

The desktop UI is a *client*, not a replacement. This server exposes the
engine over a socket and nothing more: it owns no research logic, no
browser knowledge, and no provider selectors. It forwards commands to the
existing application object and streams the engine's events back.

Two consequences follow deliberately from that split:

* Terminal logging is untouched. The engine keeps printing to stdout; the
  UI receives a *copy* of log records through the bus.
* Running without the UI is unchanged. Nothing here is required for the
  terminal REPL to work.

Transport is a single WebSocket carrying JSON. Clients receive every
engine event and may send a small command set.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.application import ResearchOperatorApp
from src.events import (
    Event,
    EventBus,
    EventType,
    get_event_bus,
    log_event,
)
from src.logger import get_logger
from src.system import MetricsSampler

_METRICS_INTERVAL_SECONDS = 2.5


class BusLogHandler(logging.Handler):
    """
    Mirrors terminal log records onto the event bus.

    A handler rather than a replacement: records still reach every existing
    handler, so the terminal output required for debugging is unchanged.
    Only a copy is forwarded to attached interfaces.
    """

    def __init__(self, bus: EventBus) -> None:
        """
        Initialize the handler.

        Args:
            bus: Bus to publish log copies onto.
        """
        super().__init__()
        self._bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        """
        Forward one record to the bus.

        Args:
            record: Log record to forward.
        """
        try:
            self._bus.publish(
                log_event(
                    level=record.levelname,
                    message=record.getMessage(),
                    source=record.name,
                )
            )
        except Exception:
            # A logging handler must never raise into the logging call.
            pass


class EngineServer:
    """
    Serves the engine to desktop clients.

    Holds the application instance, publishes its events, and executes
    commands received over the socket. One research request runs at a time,
    which matches the engine's own model and keeps provider tabs
    predictable.
    """

    def __init__(
        self,
        app: Optional[ResearchOperatorApp] = None,
        bus: Optional[EventBus] = None,
    ) -> None:
        """
        Initialize the server.

        Args:
            app: Optional application instance. Created if omitted.
            bus: Optional event bus. The shared bus is used if omitted.
        """
        self._app = app or ResearchOperatorApp()
        self._bus = bus or get_event_bus()
        self._logger = get_logger(__name__)
        self._research_lock = asyncio.Lock()
        self._current_task: Optional[asyncio.Task[None]] = None
        self._metrics_task: Optional[asyncio.Task[None]] = None
        self._metrics = MetricsSampler()
        self._research_started_at: Optional[float] = None
        self._research_durations: list[float] = []

    @property
    def app(self) -> ResearchOperatorApp:
        """
        Return the application instance being served.

        Returns:
            Application instance.
        """
        return self._app

    async def startup(self) -> None:
        """
        Initialize the engine and begin mirroring logs to the bus.

        Raises:
            AIResearchOperatorError: If engine initialization fails.
        """
        handler = BusLogHandler(self._bus)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)

        await self._app.initialize()
        self._bus.publish(
            Event(
                type=EventType.ENGINE_READY,
                payload={
                    "providers": self._app.session.enabled_providers(),
                    "disabled": self._app.session.disabled_providers(),
                },
            )
        )
        self._metrics_task = asyncio.create_task(self._metrics_loop())

    async def shutdown(self) -> None:
        """Shut the engine down, notifying attached interfaces first."""
        self._bus.publish(Event(type=EventType.ENGINE_SHUTDOWN))

        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()

        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

        await self._app.shutdown()

    async def _metrics_loop(self) -> None:
        """
        Publish one Metrics event per interval while the engine runs.

        System and engine figures travel in one event so the dashboard
        renders a consistent snapshot. A failed tick is skipped, never
        fatal.
        """
        while True:
            await asyncio.sleep(_METRICS_INTERVAL_SECONDS)
            try:
                system = await self._metrics.sample()
                engine = await self._app.runtime_metrics()

                research: dict[str, Any] = {
                    "running": self._research_lock.locked(),
                    "elapsedSeconds": None,
                    "estimatedRemainingSeconds": None,
                    "averageDurationSeconds": (
                        round(
                            sum(self._research_durations)
                            / len(self._research_durations),
                            1,
                        )
                        if self._research_durations
                        else None
                    ),
                }
                if (
                    self._research_lock.locked()
                    and self._research_started_at is not None
                ):
                    elapsed = (
                        asyncio.get_running_loop().time()
                        - self._research_started_at
                    )
                    research["elapsedSeconds"] = round(elapsed, 1)
                    if self._research_durations:
                        average = sum(self._research_durations) / len(
                            self._research_durations
                        )
                        research["estimatedRemainingSeconds"] = round(
                            max(average - elapsed, 0.0), 1
                        )

                self._bus.publish(
                    Event(
                        type=EventType.METRICS,
                        payload={
                            "system": system,
                            "engine": engine,
                            "research": research,
                        },
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.debug("Metrics tick failed: %s", exc)

    async def snapshot(self) -> dict[str, Any]:
        """
        Return current engine state for a newly connected client.

        Lets the UI render immediately rather than waiting for the next
        event to arrive.

        Returns:
            JSON-compatible state snapshot.
        """
        statuses = await self._app.session.login_status()
        return {
            "type": "Snapshot",
            "providers": [
                {
                    "name": name,
                    "displayName": status.display_name,
                    "authState": status.state.value,
                    "detail": status.detail,
                }
                for name, status in statuses.items()
            ],
            "enabled": self._app.session.enabled_providers(),
            "disabled": self._app.session.disabled_providers(),
            "paused": self._app.session.paused_providers,
            "sessionOpen": self._app.session.is_open,
            "browserHidden": self._app.session.browser_hidden,
        }

    async def handle_command(
        self,
        message: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Execute one command from a client.

        Args:
            message: Decoded JSON command with a ``command`` field.

        Returns:
            Optional direct reply. Most effects arrive as bus events
            instead, so every connected client observes them.
        """
        command = str(message.get("command", "")).strip()

        if command == "research":
            query = str(message.get("query", "")).strip()
            if not query:
                return {"type": "Error", "message": "Query cannot be empty"}
            return await self._start_research(
                query,
                new_conversation=bool(message.get("newChat", False)),
            )

        if command == "newChat":
            reset = await self._app.session.reset_conversations()
            return {"type": "ConversationsReset", "providers": reset}

        if command == "showBrowser":
            count = self._app.session.show_browser()
            return {"type": "BrowserVisibility", "hidden": False,
                    "windows": count}

        if command == "hideBrowser":
            count = self._app.session.hide_browser()
            return {"type": "BrowserVisibility", "hidden": count > 0,
                    "windows": count}

        if command == "diagnostics":
            records = await self._app.session.provider_diagnostics()
            return {"type": "Diagnostics", "providers": records}

        if command == "screenshot":
            provider = str(message.get("provider", "")).strip()
            try:
                image = await self._app.session.screenshot_provider(provider)
            except Exception as exc:
                return {"type": "Error", "message": str(exc)}
            return {
                "type": "Screenshot",
                "provider": provider,
                "image": base64.b64encode(image).decode("ascii"),
            }

        if command == "reloadProvider":
            provider = str(message.get("provider", "")).strip()
            try:
                reloaded = await self._app.session.reload_provider(provider)
            except Exception as exc:
                return {"type": "Error", "message": str(exc)}
            return {
                "type": "ProviderReloaded",
                "provider": provider,
                "reloaded": reloaded,
            }

        if command == "focusProvider":
            provider = str(message.get("provider", "")).strip()
            try:
                await self._app.session.focus_provider(provider)
            except Exception as exc:
                return {"type": "Error", "message": str(exc)}
            return {"type": "BrowserVisibility", "hidden": False}

        if command == "openSession":
            ready = await self._app.open_session()
            return {"type": "SessionOpened", "ready": ready}

        if command == "snapshot":
            return await self.snapshot()

        if command == "providers":
            return await self.snapshot()

        if command == "recover":
            recovered = await self._app.session.recover_unhealthy()
            return {"type": "Recovered", "providers": recovered}

        if command == "checkAuth":
            promoted = await self._app.session.poll_pending_auth()
            return {"type": "AuthChecked", "promoted": promoted}

        if command == "resetConversations":
            reset = await self._app.session.reset_conversations()
            return {"type": "ConversationsReset", "providers": reset}

        return {"type": "Error", "message": f"Unknown command: {command!r}"}

    async def _start_research(
        self,
        query: str,
        new_conversation: bool = False,
    ) -> dict[str, Any]:
        """
        Start a research run if one is not already in progress.

        Args:
            query: Research query.
            new_conversation: Whether providers should start fresh
                threads for this run (the explicit New Chat case).

        Returns:
            Acknowledgement describing whether the run was accepted.
        """
        if self._research_lock.locked():
            return {
                "type": "Error",
                "message": (
                    "A research run is already in progress. Wait for it to "
                    "finish before starting another."
                ),
            }

        self._current_task = asyncio.create_task(
            self._run_research(query, new_conversation)
        )
        return {"type": "ResearchAccepted", "query": query}

    async def _run_research(
        self,
        query: str,
        new_conversation: bool = False,
    ) -> None:
        """
        Execute one research run, publishing its outcome.

        Args:
            query: Research query.
            new_conversation: Whether providers start fresh threads.
        """
        async with self._research_lock:
            self._research_started_at = asyncio.get_running_loop().time()
            try:
                result = await self._app.research_now(
                    query,
                    new_conversation=new_conversation or None,
                )
                payload: dict[str, Any] = {
                    "content": result.answer,
                    "route": result.route.target.value,
                }
                # The concise final answer travels with its evidence so
                # the UI can show one recommendation and tuck the raw
                # provider responses behind an expandable section.
                if result.final is not None:
                    payload["final"] = result.final.to_payload()
                if result.research is not None:
                    payload["research"] = result.research.to_payload()
                self._bus.publish(
                    Event(
                        type=EventType.ASSISTANT_MESSAGE,
                        payload=payload,
                    )
                )
                elapsed = (
                    asyncio.get_running_loop().time()
                    - self._research_started_at
                )
                self._research_durations.append(elapsed)
                del self._research_durations[:-10]
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.exception("Research failed")
                self._bus.publish(
                    Event(
                        type=EventType.RESEARCH_FAILED,
                        payload={"error": str(exc)},
                    )
                )
            finally:
                self._research_started_at = None


def create_app(server: Optional[EngineServer] = None) -> FastAPI:
    """
    Build the FastAPI application exposing the engine.

    Args:
        server: Optional engine server. Created if omitted.

    Returns:
        Configured FastAPI application.
    """
    engine = server or EngineServer()
    api = FastAPI(title="ArchitectOS Engine", version="1.0.0")

    # The desktop client is served from a local dev server during
    # development and from the Tauri bundle in production; both are local
    # origins.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.on_event("startup")
    async def _startup() -> None:
        """Initialize the engine when the server boots."""
        await engine.startup()

    @api.on_event("shutdown")
    async def _shutdown() -> None:
        """Shut the engine down with the server."""
        await engine.shutdown()

    @api.get("/health")
    async def health() -> dict[str, Any]:
        """
        Report server and engine readiness.

        Returns:
            Health payload.
        """
        return {
            "status": "ok",
            "initialized": engine.app.is_initialized,
            "subscribers": get_event_bus().subscriber_count,
        }

    @api.get("/state")
    async def state() -> dict[str, Any]:
        """
        Return the current engine snapshot.

        Returns:
            State snapshot.
        """
        return await engine.snapshot()

    @api.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """
        Stream engine events and accept client commands.

        Args:
            websocket: Client connection.
        """
        await websocket.accept()
        bus = get_event_bus()
        subscription = bus.subscribe(replay=True)

        async def pump() -> None:
            """Forward bus events to this client."""
            async for event in subscription:
                await websocket.send_json(event.to_dict())

        forwarder = asyncio.create_task(pump())

        try:
            await websocket.send_json(await engine.snapshot())

            while True:
                message = await websocket.receive_json()
                reply = await engine.handle_command(message)
                if reply is not None:
                    await websocket.send_json(reply)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            get_logger(__name__).warning("WebSocket closed: %s", exc)
        finally:
            forwarder.cancel()
            bus.unsubscribe(subscription)

    return api


__all__ = [
    "BusLogHandler",
    "EngineServer",
    "create_app",
]
