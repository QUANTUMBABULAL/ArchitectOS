"""
Registry and dispatcher for AI-website workers.

WorkerManager owns the set of registered workers. The orchestrator talks to
workers exclusively through this manager: it resolves workers by name or
capability, dispatches queries to one or many workers concurrently, and
performs group lifecycle operations. The manager knows nothing about any
specific website.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Optional

from src.exceptions import WorkerError
from src.logger import get_logger

from .base_worker import (
    BaseWorker,
    WorkerHealth,
    WorkerQuery,
    WorkerResponse,
    WorkerState,
)


class WorkerManager:
    """
    Manages registration, lookup, dispatch, and lifecycle of workers.

    Dispatch to multiple workers is concurrent; one failing worker never
    prevents others from answering. All worker access is name-based so the
    orchestration layer stays independent of concrete worker classes.
    """

    def __init__(self) -> None:
        """Initialize an empty worker registry."""
        self._workers: dict[str, BaseWorker] = {}
        self._logger = get_logger(__name__)

    @property
    def worker_names(self) -> list[str]:
        """
        Return the names of all registered workers.

        Returns:
            Sorted worker names.
        """
        return sorted(self._workers)

    def register(self, worker: BaseWorker) -> None:
        """
        Register a worker.

        Args:
            worker: Worker to register.

        Raises:
            WorkerError: If a worker with the same name already exists.
        """
        if worker.name in self._workers:
            raise WorkerError(
                f"Worker '{worker.name}' is already registered",
                code="WORKER_ALREADY_REGISTERED",
            )
        self._workers[worker.name] = worker
        self._logger.info("Registered worker '%s'", worker.name)

    def unregister(self, name: str) -> BaseWorker:
        """
        Remove a worker from the registry.

        Args:
            name: Worker name.

        Returns:
            The removed worker.

        Raises:
            WorkerError: If no worker with that name is registered.
        """
        worker = self._workers.pop(name.strip().lower(), None)
        if worker is None:
            raise WorkerError(
                f"Worker '{name}' is not registered",
                code="WORKER_NOT_REGISTERED",
            )
        self._logger.info("Unregistered worker '%s'", worker.name)
        return worker

    def get(self, name: str) -> BaseWorker:
        """
        Return a registered worker by name.

        Args:
            name: Worker name.

        Returns:
            Registered worker.

        Raises:
            WorkerError: If no worker with that name is registered.
        """
        worker = self._workers.get(name.strip().lower())
        if worker is None:
            raise WorkerError(
                f"Worker '{name}' is not registered",
                code="WORKER_NOT_REGISTERED",
            )
        return worker

    def find_by_capability(self, capability: str) -> list[BaseWorker]:
        """
        Return all workers that advertise a capability.

        Args:
            capability: Capability tag to match.

        Returns:
            Workers with the capability, sorted by name.
        """
        tag = capability.strip().lower()
        return sorted(
            (
                worker
                for worker in self._workers.values()
                if tag in worker.capabilities
            ),
            key=lambda worker: worker.name,
        )

    def ready_workers(self) -> list[BaseWorker]:
        """
        Return all workers currently able to accept queries.

        Returns:
            Workers in READY state, sorted by name.
        """
        return sorted(
            (
                worker
                for worker in self._workers.values()
                if worker.state == WorkerState.READY
            ),
            key=lambda worker: worker.name,
        )

    async def start_all(self) -> None:
        """
        Start every registered worker.

        Workers are started sequentially because they share one browser
        session and must not race during navigation.

        Raises:
            WorkerError: If any worker fails to start.
        """
        for worker in self._workers.values():
            await worker.start()

    async def start_available(self) -> dict[str, Optional[str]]:
        """
        Start every registered worker, tolerating individual failures.

        Used for multi-provider consultation, where one provider being
        logged out or having changed its layout must not prevent the rest
        from answering. Startup remains sequential so tab creation and
        navigation do not race.

        Returns:
            Mapping of worker name to None on success, or the failure
            description on error.
        """
        results: dict[str, Optional[str]] = {}

        for worker in self._workers.values():
            try:
                await worker.start()
                results[worker.name] = None
            except Exception as exc:
                results[worker.name] = str(exc)
                self._logger.warning(
                    "Worker '%s' failed to start and will be skipped: %s",
                    worker.name,
                    exc,
                )

        started = [name for name, error in results.items() if error is None]
        self._logger.info(
            "Started %d/%d worker(s): %s",
            len(started),
            len(results),
            ", ".join(started) or "none",
        )
        return results

    async def stop_all(self) -> None:
        """
        Stop every registered worker.

        Individual stop failures are logged by the workers themselves and
        never abort the group shutdown.
        """
        for worker in self._workers.values():
            await worker.stop()

    async def dispatch(
        self,
        name: str,
        query: WorkerQuery,
    ) -> WorkerResponse:
        """
        Send one query to one worker.

        Args:
            name: Worker name.
            query: Query to submit.

        Returns:
            Worker response.

        Raises:
            WorkerError: If the worker is not registered or not started.
        """
        worker = self.get(name)
        return await worker.ask(query)

    async def dispatch_many(
        self,
        names: Iterable[str],
        query: WorkerQuery,
    ) -> list[WorkerResponse]:
        """
        Send the same query to several workers concurrently.

        Args:
            names: Worker names to consult.
            query: Query to submit to each worker.

        Returns:
            Responses in the same order as ``names``. Workers that raised
            unexpectedly are represented as failed responses rather than
            propagating exceptions.

        Raises:
            WorkerError: If any name is not registered.
        """
        workers = [self.get(name) for name in names]

        results = await asyncio.gather(
            *(worker.ask(query) for worker in workers),
            return_exceptions=True,
        )

        responses: list[WorkerResponse] = []
        for worker, result in zip(workers, results):
            if isinstance(result, WorkerResponse):
                responses.append(result)
                continue

            error = str(result)
            self._logger.error(
                "Worker '%s' raised during dispatch: %s", worker.name, error
            )
            responses.append(
                WorkerResponse(
                    query_id=query.query_id,
                    worker_name=worker.name,
                    prompt=query.prompt,
                    answer="",
                    success=False,
                    error=error,
                    metadata=dict(query.metadata),
                )
            )
        return responses

    async def health_check_all(self) -> dict[str, WorkerHealth]:
        """
        Health-check every registered worker concurrently.

        Returns:
            Mapping of worker name to health snapshot.
        """
        names = list(self._workers)
        checks = await asyncio.gather(
            *(self._workers[name].health_check() for name in names),
            return_exceptions=True,
        )

        health: dict[str, WorkerHealth] = {}
        for name, result in zip(names, checks):
            if isinstance(result, WorkerHealth):
                health[name] = result
            else:
                health[name] = WorkerHealth(
                    worker_name=name,
                    state=self._workers[name].state,
                    healthy=False,
                    detail=str(result),
                )
        return health


__all__ = [
    "WorkerManager",
]
