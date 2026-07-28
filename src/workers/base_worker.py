"""
Worker framework contract for AI-website workers.

BaseWorker defines the interface and shared behavior for every worker that
communicates with an online AI system through the browser operating system.
Workers own website-specific logic (selectors, waiting rules, extraction);
the browser layer stays generic and the orchestration layer stays
site-agnostic by only depending on this contract.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.browser import (
    BrowserSession,
    Extractor,
    KeyboardController,
    MouseController,
    TabManager,
)
from src.constants import BACKOFF_MULTIPLIER, INITIAL_RETRY_DELAY, MAX_RETRIES
from src.exceptions import ProviderChallengeError, WorkerError
from src.logger import get_logger


class WorkerState(str, Enum):
    """
    Lifecycle states for a worker.

    States let WorkerManager and the orchestrator make conservative
    dispatch decisions: only READY workers receive queries.
    """

    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


class WorkerConfig(BaseModel):
    """
    Shared configuration for workers.

    Attributes:
        request_timeout_seconds: Maximum time to wait for one full answer.
        max_retries: Maximum attempts per query before failing.
        retry_delay_seconds: Initial delay between retry attempts.
        retry_backoff_multiplier: Exponential backoff multiplier.
    """

    request_timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=MAX_RETRIES, ge=1)
    retry_delay_seconds: float = Field(default=INITIAL_RETRY_DELAY, gt=0)
    retry_backoff_multiplier: float = Field(
        default=BACKOFF_MULTIPLIER, ge=1.0
    )


@dataclass(frozen=True, slots=True)
class WorkerQuery:
    """
    One question sent to a worker.

    Attributes:
        prompt: Question or instruction text.
        query_id: Stable identifier for tracing the query.
        context: Optional extra context prepended by the worker if useful.
        new_conversation: Whether the worker should start a fresh thread.
            Defaults to False so persistent research sessions continue an
            existing conversation, preserving the context that follow-up
            prompts and multi-round debate depend on. Set True only when
            contamination from earlier turns must be avoided.
        metadata: Free-form caller metadata carried through to the response.
    """

    prompt: str
    query_id: str = field(default_factory=lambda: uuid4().hex)
    context: Optional[str] = None
    new_conversation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    """
    One answer produced by a worker.

    Attributes:
        query_id: Identifier of the originating query.
        worker_name: Name of the worker that produced the answer.
        prompt: Prompt that was asked.
        answer: Extracted answer text.
        success: Whether the worker considers the answer valid.
        error: Error description when success is False.
        attempts: Number of attempts used.
        elapsed_seconds: Wall-clock time spent producing the answer.
        created_at: UTC timestamp when the response was finalized.
        metadata: Diagnostic metadata (URLs, extraction details, etc.).
    """

    query_id: str
    worker_name: str
    prompt: str
    answer: str
    success: bool
    error: Optional[str] = None
    attempts: int = 1
    elapsed_seconds: float = 0.0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerHealth:
    """
    Health-check result for one worker.

    Attributes:
        worker_name: Worker name.
        state: Current worker state.
        healthy: True when the worker can accept queries.
        detail: Human-readable diagnostic detail.
    """

    worker_name: str
    state: WorkerState
    healthy: bool
    detail: Optional[str] = None


class BaseWorker(ABC):
    """
    Abstract base class for AI-website workers.

    Subclasses implement site navigation, prompt submission, response
    waiting, and extraction. BaseWorker provides the shared lifecycle,
    retry loop, timing, and structured response construction so those
    concerns are implemented exactly once.
    """

    def __init__(
        self,
        name: str,
        session: BrowserSession,
        config: Optional[WorkerConfig] = None,
        tab_manager: Optional[TabManager] = None,
        keyboard: Optional[KeyboardController] = None,
        mouse: Optional[MouseController] = None,
        extractor: Optional[Extractor] = None,
    ) -> None:
        """
        Initialize the worker.

        Args:
            name: Unique worker name (for example ``chatgpt``).
            session: Browser session the worker operates in.
            config: Optional worker configuration.
            tab_manager: Optional tab manager dependency.
            keyboard: Optional keyboard controller dependency.
            mouse: Optional mouse controller dependency.
            extractor: Optional extractor dependency.
        """
        if not name or not name.strip():
            raise WorkerError(
                "Worker name cannot be empty",
                code="WORKER_NAME_EMPTY",
            )

        self._name = name.strip().lower()
        self._session = session
        self._config = config or WorkerConfig()
        self._tabs = tab_manager or TabManager(session)
        self._keyboard = keyboard or KeyboardController()
        self._mouse = mouse or MouseController()
        self._extractor = extractor or Extractor()
        self._state = WorkerState.CREATED
        self._lock = asyncio.Lock()
        self._logger = get_logger(f"{__name__}.{self._name}")

    @property
    def name(self) -> str:
        """
        Return the worker name.

        Returns:
            Worker name.
        """
        return self._name

    @property
    def state(self) -> WorkerState:
        """
        Return the current worker state.

        Returns:
            Worker state.
        """
        return self._state

    @property
    def config(self) -> WorkerConfig:
        """
        Return the worker configuration.

        Returns:
            Worker configuration.
        """
        return self._config

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[str]:
        """
        Return capability tags for this worker.

        Returns:
            Capability tags such as ``{"reasoning", "code"}`` used by the
            orchestrator to select consultation targets.
        """

    async def start(self) -> None:
        """
        Prepare the worker for queries.

        Navigates to the target site and verifies readiness.

        Raises:
            WorkerError: If the worker cannot become ready.
        """
        async with self._lock:
            if self._state == WorkerState.READY:
                return

            self._state = WorkerState.STARTING
            try:
                await self._prepare()
                self._state = WorkerState.READY
                self._logger.info("Worker '%s' is ready", self._name)
            except Exception as exc:
                self._state = WorkerState.ERROR
                # A human-verification challenge propagates unchanged so
                # callers can pause this provider rather than treat it as
                # an ordinary startup failure to retry.
                if isinstance(exc, (ProviderChallengeError, WorkerError)):
                    raise
                raise WorkerError(
                    f"Worker '{self._name}' failed to start: {exc}",
                    code="WORKER_START_FAILED",
                ) from exc

    async def stop(self) -> None:
        """
        Stop the worker and release site resources.

        Never raises; shutdown problems are logged so manager shutdown can
        proceed.
        """
        async with self._lock:
            try:
                await self._cleanup()
            except Exception as exc:
                self._logger.warning(
                    "Worker '%s' cleanup failed: %s", self._name, exc
                )
            finally:
                self._state = WorkerState.STOPPED
                self._logger.info("Worker '%s' stopped", self._name)

    async def ask(self, query: WorkerQuery) -> WorkerResponse:
        """
        Ask the worker a question and wait for the complete answer.

        Runs the site-specific implementation inside a retry loop with
        exponential backoff and a hard per-request timeout. Failed attempts
        trigger the worker's recovery hook before the next try.

        Args:
            query: Query to submit.

        Returns:
            Structured worker response. ``success`` is False when all
            attempts failed; the method only raises for invalid usage.

        Raises:
            WorkerError: If the worker has not been started.
        """
        if self._state in {WorkerState.CREATED, WorkerState.STOPPED}:
            raise WorkerError(
                f"Worker '{self._name}' is not started",
                code="WORKER_NOT_STARTED",
            )

        async with self._lock:
            self._state = WorkerState.BUSY
            started = asyncio.get_running_loop().time()
            delay = self._config.retry_delay_seconds
            last_error: Optional[str] = None
            attempts = 0

            try:
                for attempt in range(1, self._config.max_retries + 1):
                    attempts = attempt
                    try:
                        answer = await asyncio.wait_for(
                            self._execute(query),
                            timeout=self._config.request_timeout_seconds,
                        )
                        elapsed = (
                            asyncio.get_running_loop().time() - started
                        )
                        self._state = WorkerState.READY
                        return WorkerResponse(
                            query_id=query.query_id,
                            worker_name=self._name,
                            prompt=query.prompt,
                            answer=answer,
                            success=True,
                            attempts=attempt,
                            elapsed_seconds=elapsed,
                            metadata=dict(query.metadata),
                        )
                    except asyncio.TimeoutError:
                        last_error = (
                            f"Timed out after "
                            f"{self._config.request_timeout_seconds}s"
                        )
                    except ProviderChallengeError:
                        # Retrying a CAPTCHA cannot succeed and may
                        # harden the challenge. Fail out immediately so
                        # the provider can be paused.
                        self._state = WorkerState.ERROR
                        raise
                    except Exception as exc:
                        last_error = str(exc)

                    self._logger.warning(
                        "Worker '%s' attempt %d/%d failed: %s",
                        self._name,
                        attempt,
                        self._config.max_retries,
                        last_error,
                    )

                    if attempt < self._config.max_retries:
                        try:
                            await self._recover()
                        except Exception as recover_exc:
                            self._logger.warning(
                                "Worker '%s' recovery failed: %s",
                                self._name,
                                recover_exc,
                            )
                        await asyncio.sleep(delay)
                        delay *= self._config.retry_backoff_multiplier

                elapsed = asyncio.get_running_loop().time() - started
                self._state = WorkerState.ERROR
                return WorkerResponse(
                    query_id=query.query_id,
                    worker_name=self._name,
                    prompt=query.prompt,
                    answer="",
                    success=False,
                    error=last_error,
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    metadata=dict(query.metadata),
                )
            finally:
                if self._state == WorkerState.BUSY:
                    self._state = WorkerState.READY

    async def health_check(self) -> WorkerHealth:
        """
        Check whether the worker can accept queries.

        Returns:
            Worker health snapshot.
        """
        detail: Optional[str] = None
        healthy = False

        if self._state in {WorkerState.READY, WorkerState.BUSY}:
            try:
                healthy = await self._is_site_ready()
                if not healthy:
                    detail = "Site readiness check failed"
            except Exception as exc:
                detail = str(exc)
        else:
            detail = f"Worker state is {self._state.value}"

        return WorkerHealth(
            worker_name=self._name,
            state=self._state,
            healthy=healthy,
            detail=detail,
        )

    @abstractmethod
    async def _prepare(self) -> None:
        """
        Navigate to the target site and confirm the worker can operate.

        Raises:
            WorkerError: If preparation fails.
        """

    @abstractmethod
    async def _execute(self, query: WorkerQuery) -> str:
        """
        Submit one query and return the complete extracted answer.

        Args:
            query: Query to submit.

        Returns:
            Extracted answer text.

        Raises:
            WorkerError: If submission, waiting, or extraction fails.
        """

    @abstractmethod
    async def _recover(self) -> None:
        """
        Attempt to restore the worker after a failed query attempt.

        Typical implementations reload the page or open a fresh tab.
        """

    @abstractmethod
    async def _is_site_ready(self) -> bool:
        """
        Check that the site is loaded and accepting input.

        Returns:
            True when the site is ready for a query.
        """

    async def _cleanup(self) -> None:
        """
        Release site resources during shutdown.

        The default implementation does nothing; subclasses override when
        they own tabs or other resources.
        """


__all__ = [
    "BaseWorker",
    "WorkerConfig",
    "WorkerHealth",
    "WorkerQuery",
    "WorkerResponse",
    "WorkerState",
]
