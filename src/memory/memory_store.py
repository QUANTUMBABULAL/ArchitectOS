"""
Persistent memory for research sessions.

MemoryStore persists research sessions, worker responses, synthesized
reports, and user feedback in a local SQLite database. All operations are
async: blocking SQLite work runs in worker threads, and writes are
serialized with an asyncio lock. Other layers exchange plain dataclass
records and never see SQL.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

from src.config import Settings, get_settings
from src.exceptions import MemoryError as MemoryStorageError
from src.logger import get_logger

T = TypeVar("T")


def _utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ResearchRecord:
    """
    One persisted research session.

    Attributes:
        research_id: Stable research session identifier.
        goal: Original user goal.
        status: Session status (running, completed, failed).
        plan: Serialized plan structure.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of the last status change.
    """

    research_id: str
    goal: str
    status: str
    plan: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    """
    One persisted worker (or local) response.

    Attributes:
        response_id: Stable response identifier.
        research_id: Owning research session.
        step_id: Plan step that produced the response.
        source: Worker name or ``local``.
        prompt: Prompt that was asked.
        answer: Answer text.
        success: Whether the response is valid.
        error: Error description for failed responses.
        attempts: Attempts used to produce the response.
        elapsed_seconds: Time spent producing the response.
        created_at: ISO timestamp of creation.
    """

    response_id: str
    research_id: str
    step_id: str
    source: str
    prompt: str
    answer: str
    success: bool
    error: Optional[str] = None
    attempts: int = 1
    elapsed_seconds: float = 0.0
    created_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class ReportRecord:
    """
    One persisted synthesized report.

    Attributes:
        report_id: Stable report identifier.
        research_id: Owning research session.
        content: Report text.
        consensus: Serialized consensus analysis.
        created_at: ISO timestamp of creation.
    """

    report_id: str
    research_id: str
    content: str
    consensus: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """
    One persisted user feedback entry.

    Attributes:
        feedback_id: Stable feedback identifier.
        research_id: Research session the feedback concerns.
        rating: Integer rating (1-5).
        comment: Optional free-form comment.
        created_at: ISO timestamp of creation.
    """

    feedback_id: str
    research_id: str
    rating: int
    comment: Optional[str] = None
    created_at: str = field(default_factory=_utc_now)


_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS research_sessions (
        research_id TEXT PRIMARY KEY,
        goal TEXT NOT NULL,
        status TEXT NOT NULL,
        plan_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS responses (
        response_id TEXT PRIMARY KEY,
        research_id TEXT NOT NULL,
        step_id TEXT NOT NULL,
        source TEXT NOT NULL,
        prompt TEXT NOT NULL,
        answer TEXT NOT NULL,
        success INTEGER NOT NULL,
        error TEXT,
        attempts INTEGER NOT NULL DEFAULT 1,
        elapsed_seconds REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (research_id)
            REFERENCES research_sessions (research_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reports (
        report_id TEXT PRIMARY KEY,
        research_id TEXT NOT NULL,
        content TEXT NOT NULL,
        consensus_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (research_id)
            REFERENCES research_sessions (research_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id TEXT PRIMARY KEY,
        research_id TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comment TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (research_id)
            REFERENCES research_sessions (research_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_responses_research
        ON responses (research_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_reports_research
        ON reports (research_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_feedback_research
        ON feedback (research_id)
    """,
)


class MemoryStore:
    """
    SQLite-backed persistence for the research operating system.

    Every public method is async. Blocking SQLite calls run in worker
    threads; writes are serialized with an asyncio lock so concurrent
    orchestration tasks cannot interleave partial writes.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the memory store.

        Args:
            db_path: Optional explicit database file path.
            settings: Optional application settings used to derive the
                default path.
        """
        self._settings = settings or get_settings()
        self._db_path = db_path or (
            Path(self._settings.data_dir) / "memory.db"
        )
        self._write_lock = asyncio.Lock()
        self._initialized = False
        self._logger = get_logger(__name__)

    @property
    def db_path(self) -> Path:
        """
        Return the database file path.

        Returns:
            SQLite database path.
        """
        return self._db_path

    async def initialize(self) -> None:
        """
        Create the database file and schema if needed.

        Raises:
            MemoryStorageError: If the schema cannot be created.
        """
        if self._initialized:
            return

        def create_schema() -> None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.commit()

        await self._run(create_schema, write=True)
        self._initialized = True
        self._logger.info("Memory store ready at %s", self._db_path)

    async def save_research(self, record: ResearchRecord) -> None:
        """
        Persist a new research session.

        Args:
            record: Research session record.

        Raises:
            MemoryStorageError: If the write fails.
        """
        await self._ensure_initialized()

        def write() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO research_sessions
                        (research_id, goal, status, plan_json,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.research_id,
                        record.goal,
                        record.status,
                        json.dumps(record.plan),
                        record.created_at,
                        record.updated_at,
                    ),
                )
                connection.commit()

        await self._run(write, write=True)

    async def update_research_status(
        self,
        research_id: str,
        status: str,
    ) -> None:
        """
        Update the status of a research session.

        Args:
            research_id: Research session identifier.
            status: New status value.

        Raises:
            MemoryStorageError: If the session does not exist or the
                write fails.
        """
        await self._ensure_initialized()

        def write() -> None:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE research_sessions
                    SET status = ?, updated_at = ?
                    WHERE research_id = ?
                    """,
                    (status, _utc_now(), research_id),
                )
                connection.commit()
                if cursor.rowcount == 0:
                    raise ValueError(
                        f"Research session not found: {research_id}"
                    )

        await self._run(write, write=True)

    async def get_research(
        self,
        research_id: str,
    ) -> Optional[ResearchRecord]:
        """
        Fetch one research session.

        Args:
            research_id: Research session identifier.

        Returns:
            Research record or None when absent.

        Raises:
            MemoryStorageError: If the read fails.
        """
        await self._ensure_initialized()

        def read() -> Optional[ResearchRecord]:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT research_id, goal, status, plan_json,
                           created_at, updated_at
                    FROM research_sessions
                    WHERE research_id = ?
                    """,
                    (research_id,),
                ).fetchone()

            if row is None:
                return None
            return ResearchRecord(
                research_id=row[0],
                goal=row[1],
                status=row[2],
                plan=json.loads(row[3]),
                created_at=row[4],
                updated_at=row[5],
            )

        return await self._run(read)

    async def list_research(self, limit: int = 50) -> list[ResearchRecord]:
        """
        List recent research sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            Research records, newest first.

        Raises:
            MemoryStorageError: If the read fails.
        """
        await self._ensure_initialized()
        bounded_limit = max(1, min(limit, 500))

        def read() -> list[ResearchRecord]:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT research_id, goal, status, plan_json,
                           created_at, updated_at
                    FROM research_sessions
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (bounded_limit,),
                ).fetchall()

            return [
                ResearchRecord(
                    research_id=row[0],
                    goal=row[1],
                    status=row[2],
                    plan=json.loads(row[3]),
                    created_at=row[4],
                    updated_at=row[5],
                )
                for row in rows
            ]

        return await self._run(read)

    async def save_response(self, record: ResponseRecord) -> None:
        """
        Persist one response.

        Args:
            record: Response record.

        Raises:
            MemoryStorageError: If the write fails.
        """
        await self._ensure_initialized()

        def write() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO responses
                        (response_id, research_id, step_id, source,
                         prompt, answer, success, error, attempts,
                         elapsed_seconds, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.response_id,
                        record.research_id,
                        record.step_id,
                        record.source,
                        record.prompt,
                        record.answer,
                        1 if record.success else 0,
                        record.error,
                        record.attempts,
                        record.elapsed_seconds,
                        record.created_at,
                    ),
                )
                connection.commit()

        await self._run(write, write=True)

    async def get_responses(
        self,
        research_id: str,
    ) -> list[ResponseRecord]:
        """
        Fetch all responses for a research session.

        Args:
            research_id: Research session identifier.

        Returns:
            Response records in creation order.

        Raises:
            MemoryStorageError: If the read fails.
        """
        await self._ensure_initialized()

        def read() -> list[ResponseRecord]:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT response_id, research_id, step_id, source,
                           prompt, answer, success, error, attempts,
                           elapsed_seconds, created_at
                    FROM responses
                    WHERE research_id = ?
                    ORDER BY created_at ASC
                    """,
                    (research_id,),
                ).fetchall()

            return [
                ResponseRecord(
                    response_id=row[0],
                    research_id=row[1],
                    step_id=row[2],
                    source=row[3],
                    prompt=row[4],
                    answer=row[5],
                    success=bool(row[6]),
                    error=row[7],
                    attempts=row[8],
                    elapsed_seconds=row[9],
                    created_at=row[10],
                )
                for row in rows
            ]

        return await self._run(read)

    async def save_report(self, record: ReportRecord) -> None:
        """
        Persist one synthesized report.

        Args:
            record: Report record.

        Raises:
            MemoryStorageError: If the write fails.
        """
        await self._ensure_initialized()

        def write() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO reports
                        (report_id, research_id, content,
                         consensus_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.report_id,
                        record.research_id,
                        record.content,
                        json.dumps(record.consensus),
                        record.created_at,
                    ),
                )
                connection.commit()

        await self._run(write, write=True)

    async def get_reports(self, research_id: str) -> list[ReportRecord]:
        """
        Fetch all reports for a research session.

        Args:
            research_id: Research session identifier.

        Returns:
            Report records, newest first.

        Raises:
            MemoryStorageError: If the read fails.
        """
        await self._ensure_initialized()

        def read() -> list[ReportRecord]:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT report_id, research_id, content,
                           consensus_json, created_at
                    FROM reports
                    WHERE research_id = ?
                    ORDER BY created_at DESC
                    """,
                    (research_id,),
                ).fetchall()

            return [
                ReportRecord(
                    report_id=row[0],
                    research_id=row[1],
                    content=row[2],
                    consensus=json.loads(row[3]),
                    created_at=row[4],
                )
                for row in rows
            ]

        return await self._run(read)

    async def save_feedback(self, record: FeedbackRecord) -> None:
        """
        Persist one feedback entry.

        Args:
            record: Feedback record.

        Raises:
            MemoryStorageError: If the rating is invalid or the write
                fails.
        """
        if not 1 <= record.rating <= 5:
            raise MemoryStorageError(
                f"Feedback rating must be between 1 and 5, "
                f"got {record.rating}",
                code="MEMORY_FEEDBACK_RATING_INVALID",
            )

        await self._ensure_initialized()

        def write() -> None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO feedback
                        (feedback_id, research_id, rating, comment,
                         created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.feedback_id,
                        record.research_id,
                        record.rating,
                        record.comment,
                        record.created_at,
                    ),
                )
                connection.commit()

        await self._run(write, write=True)

    async def get_feedback(
        self,
        research_id: str,
    ) -> list[FeedbackRecord]:
        """
        Fetch all feedback for a research session.

        Args:
            research_id: Research session identifier.

        Returns:
            Feedback records, newest first.

        Raises:
            MemoryStorageError: If the read fails.
        """
        await self._ensure_initialized()

        def read() -> list[FeedbackRecord]:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT feedback_id, research_id, rating, comment,
                           created_at
                    FROM feedback
                    WHERE research_id = ?
                    ORDER BY created_at DESC
                    """,
                    (research_id,),
                ).fetchall()

            return [
                FeedbackRecord(
                    feedback_id=row[0],
                    research_id=row[1],
                    rating=row[2],
                    comment=row[3],
                    created_at=row[4],
                )
                for row in rows
            ]

        return await self._run(read)

    def _connect(self) -> sqlite3.Connection:
        """
        Open a SQLite connection with sane defaults.

        Returns:
            SQLite connection.
        """
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    async def _ensure_initialized(self) -> None:
        """Initialize the schema on first use."""
        if not self._initialized:
            await self.initialize()

    async def _run(
        self,
        operation: Callable[[], T],
        write: bool = False,
    ) -> T:
        """
        Run a blocking database operation in a worker thread.

        Args:
            operation: Zero-argument callable performing database work.
            write: Whether the operation mutates the database. Writes are
                serialized with the store's lock.

        Returns:
            Operation result.

        Raises:
            MemoryStorageError: If the operation fails.
        """
        try:
            if write:
                async with self._write_lock:
                    return await asyncio.to_thread(operation)
            return await asyncio.to_thread(operation)
        except MemoryStorageError:
            raise
        except Exception as exc:
            raise MemoryStorageError(
                f"Memory store operation failed: {exc}",
                code="MEMORY_OPERATION_FAILED",
            ) from exc


def new_id() -> str:
    """
    Generate a new stable identifier for memory records.

    Returns:
        Hex UUID string.
    """
    return uuid4().hex


__all__ = [
    "FeedbackRecord",
    "MemoryStore",
    "ReportRecord",
    "ResearchRecord",
    "ResponseRecord",
    "new_id",
]
