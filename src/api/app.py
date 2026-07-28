"""
FastAPI application exposing the research operating system.

The API layer composes the dependency graph and exposes it over HTTP; it
contains no research logic. Heavy components (browser, workers) start
lazily on the first research request so the API boots instantly and
health checks stay side-effect free.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request

from src.brain import DecisionEngine, OllamaClient
from src.browser import (
    BrowserConnection,
    BrowserManager,
    BrowserSessionState,
)
from src.config import Settings, get_settings
from src.consensus import ConsensusEngine
from src.exceptions import AIResearchOperatorError
from src.logger import get_logger
from src.memory import FeedbackRecord, MemoryStore, new_id
from src.orchestrator import ResearchOrchestrator
from src.planner import Planner
from src.workers import ChatGPTWorker, WorkerManager

from .schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchSummary,
    StoredReport,
    StoredResponse,
)

_logger = get_logger(__name__)


class AppComponents:
    """
    Owns the application dependency graph.

    Light components (memory, brain, planner, consensus) are built at
    startup. The browser stack and workers boot lazily behind a lock on
    the first research request and are torn down at shutdown.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Build the light part of the dependency graph.

        Args:
            settings: Optional application settings.
        """
        self.settings = settings or get_settings()
        self.settings.create_directories()

        self.memory = MemoryStore(settings=self.settings)
        self.ollama = OllamaClient(settings=self.settings)
        self.decision_engine = DecisionEngine(client=self.ollama)
        self.planner = Planner(decision_engine=self.decision_engine)
        self.consensus = ConsensusEngine(
            decision_engine=self.decision_engine,
            settings=self.settings,
        )
        self.worker_manager = WorkerManager()
        self.orchestrator = ResearchOrchestrator(
            planner=self.planner,
            decision_engine=self.decision_engine,
            worker_manager=self.worker_manager,
            consensus_engine=self.consensus,
            memory=self.memory,
        )

        self.browser_manager: Optional[BrowserManager] = None
        self._browser_lock = asyncio.Lock()
        self._logger = get_logger(__name__)

    async def ensure_workers_started(self) -> None:
        """
        Start the browser and workers if they are not running yet.

        Raises:
            AIResearchOperatorError: If the browser or workers fail to
                start.
        """
        async with self._browser_lock:
            if (
                self.browser_manager is not None
                and self.browser_manager.is_running
            ):
                return

            self._logger.info("Starting browser stack")
            connection = BrowserConnection(settings=self.settings)
            self.browser_manager = BrowserManager(
                settings=self.settings,
                factory=connection,
            )
            session = await self.browser_manager.start()

            if "chatgpt" not in self.worker_manager.worker_names:
                self.worker_manager.register(ChatGPTWorker(session=session))
            await self.worker_manager.start_all()

    async def shutdown(self) -> None:
        """Stop workers and the browser stack, tolerating failures."""
        try:
            await self.worker_manager.stop_all()
        except Exception as exc:
            self._logger.warning("Worker shutdown failed: %s", exc)

        if self.browser_manager is not None:
            try:
                await self.browser_manager.stop()
            except Exception as exc:
                self._logger.warning("Browser shutdown failed: %s", exc)
            finally:
                self.browser_manager = None


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        settings: Optional application settings.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        components = AppComponents(settings=settings)
        await components.memory.initialize()
        application.state.components = components
        try:
            yield
        finally:
            await components.shutdown()

    app = FastAPI(
        title="AI Research Operator",
        description=(
            "Local AI research operating system that orchestrates online "
            "AI systems through browser automation"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    def components_of(request: Request) -> AppComponents:
        return request.app.state.components

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Report component health without side effects."""
        components = components_of(request)

        ollama_ok = await components.decision_engine.health_check()

        browser_status = "not started"
        if components.browser_manager is not None:
            browser_health = await components.browser_manager.health_check()
            browser_status = (
                "healthy"
                if browser_health.healthy
                else browser_health.state.value
                if browser_health.state != BrowserSessionState.STOPPED
                else "stopped"
            )

        worker_health = await components.worker_manager.health_check_all()
        workers = {
            name: {
                "state": health_item.state.value,
                "healthy": health_item.healthy,
                "detail": health_item.detail,
            }
            for name, health_item in worker_health.items()
        }

        status = "ok" if ollama_ok else "degraded"
        return HealthResponse(
            status=status,
            ollama=ollama_ok,
            browser=browser_status,
            workers=workers,
        )

    @app.post("/research", response_model=ResearchResponse)
    async def run_research(
        request: Request,
        payload: ResearchRequest,
    ) -> ResearchResponse:
        """Run one research session end to end and return the report."""
        components = components_of(request)

        try:
            await components.ensure_workers_started()
        except AIResearchOperatorError as exc:
            _logger.warning(
                "Browser stack unavailable (%s); research will run with "
                "local answers only",
                exc,
            )

        try:
            outcome = await components.orchestrator.run_research(
                payload.goal
            )
        except AIResearchOperatorError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        steps = [
            {
                "step_id": result.step.step_id,
                "kind": result.step.kind.value,
                "prompt": result.step.prompt,
                "responses": [
                    {
                        "source": response.worker_name,
                        "success": response.success,
                        "error": response.error,
                    }
                    for response in result.responses
                ],
                "consensus": (
                    {
                        "agreement": result.consensus.agreement_score,
                        "confidence": result.consensus.confidence,
                        "reached": result.consensus.consensus_reached,
                    }
                    if result.consensus is not None
                    else None
                ),
            }
            for result in outcome.step_results
        ]

        return ResearchResponse(
            research_id=outcome.research_id,
            goal=outcome.goal,
            report=outcome.report,
            consensus=outcome.consensus_summary,
            steps=steps,
        )

    @app.get("/research", response_model=list[ResearchSummary])
    async def list_research(
        request: Request,
        limit: int = 50,
    ) -> list[ResearchSummary]:
        """List recent research sessions."""
        components = components_of(request)
        records = await components.memory.list_research(limit=limit)
        return [
            ResearchSummary(
                research_id=record.research_id,
                goal=record.goal,
                status=record.status,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            for record in records
        ]

    @app.get(
        "/research/{research_id}",
        response_model=ResearchSummary,
    )
    async def get_research(
        request: Request,
        research_id: str,
    ) -> ResearchSummary:
        """Fetch one research session."""
        components = components_of(request)
        record = await components.memory.get_research(research_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Research session not found: {research_id}",
            )
        return ResearchSummary(
            research_id=record.research_id,
            goal=record.goal,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @app.get(
        "/research/{research_id}/responses",
        response_model=list[StoredResponse],
    )
    async def get_responses(
        request: Request,
        research_id: str,
    ) -> list[StoredResponse]:
        """Fetch all persisted responses for a research session."""
        components = components_of(request)
        records = await components.memory.get_responses(research_id)
        return [
            StoredResponse(
                response_id=record.response_id,
                step_id=record.step_id,
                source=record.source,
                prompt=record.prompt,
                answer=record.answer,
                success=record.success,
                error=record.error,
                attempts=record.attempts,
                elapsed_seconds=record.elapsed_seconds,
                created_at=record.created_at,
            )
            for record in records
        ]

    @app.get(
        "/research/{research_id}/reports",
        response_model=list[StoredReport],
    )
    async def get_reports(
        request: Request,
        research_id: str,
    ) -> list[StoredReport]:
        """Fetch all persisted reports for a research session."""
        components = components_of(request)
        records = await components.memory.get_reports(research_id)
        return [
            StoredReport(
                report_id=record.report_id,
                content=record.content,
                consensus=record.consensus,
                created_at=record.created_at,
            )
            for record in records
        ]

    @app.post(
        "/research/{research_id}/feedback",
        response_model=FeedbackResponse,
    )
    async def submit_feedback(
        request: Request,
        research_id: str,
        payload: FeedbackRequest,
    ) -> FeedbackResponse:
        """Store user feedback for a research session."""
        components = components_of(request)

        record = await components.memory.get_research(research_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Research session not found: {research_id}",
            )

        feedback = FeedbackRecord(
            feedback_id=new_id(),
            research_id=research_id,
            rating=payload.rating,
            comment=payload.comment,
        )
        await components.memory.save_feedback(feedback)
        return FeedbackResponse(
            feedback_id=feedback.feedback_id,
            research_id=research_id,
        )

    return app


app = create_app()


__all__ = [
    "AppComponents",
    "app",
    "create_app",
]
