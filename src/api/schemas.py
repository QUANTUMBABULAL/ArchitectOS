"""
Pydantic schemas for the REST API.

Request and response models are kept separate from domain dataclasses so
the wire format can evolve without touching orchestration internals.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """
    Request to run a research session.

    Attributes:
        goal: Research goal to investigate.
    """

    goal: str = Field(min_length=3, max_length=4000)


class ResearchResponse(BaseModel):
    """
    Result of a completed research session.

    Attributes:
        research_id: Persisted research session identifier.
        goal: Original research goal.
        report: Final synthesized report.
        consensus: Aggregated consensus metadata.
        steps: Executed plan step summaries.
    """

    research_id: str
    goal: str
    report: str
    consensus: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]] = Field(default_factory=list)


class ResearchSummary(BaseModel):
    """
    Summary of one persisted research session.

    Attributes:
        research_id: Research session identifier.
        goal: Research goal.
        status: Session status.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    research_id: str
    goal: str
    status: str
    created_at: str
    updated_at: str


class StoredResponse(BaseModel):
    """
    One persisted response.

    Attributes:
        response_id: Response identifier.
        step_id: Producing plan step.
        source: Worker name or ``local``.
        prompt: Prompt asked.
        answer: Answer text.
        success: Whether the response is valid.
        error: Error description for failures.
        attempts: Attempts used.
        elapsed_seconds: Production time.
        created_at: Creation timestamp.
    """

    response_id: str
    step_id: str
    source: str
    prompt: str
    answer: str
    success: bool
    error: Optional[str] = None
    attempts: int = 1
    elapsed_seconds: float = 0.0
    created_at: str


class StoredReport(BaseModel):
    """
    One persisted report.

    Attributes:
        report_id: Report identifier.
        content: Report text.
        consensus: Consensus metadata.
        created_at: Creation timestamp.
    """

    report_id: str
    content: str
    consensus: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class FeedbackRequest(BaseModel):
    """
    Feedback submission for a research session.

    Attributes:
        rating: Rating between 1 and 5.
        comment: Optional free-form comment.
    """

    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=4000)


class FeedbackResponse(BaseModel):
    """
    Acknowledgement of stored feedback.

    Attributes:
        feedback_id: Feedback identifier.
        research_id: Research session identifier.
    """

    feedback_id: str
    research_id: str


class HealthResponse(BaseModel):
    """
    System health snapshot.

    Attributes:
        status: Overall status: ``ok`` or ``degraded``.
        ollama: Whether local inference is reachable.
        browser: Browser session status description.
        workers: Per-worker health details.
    """

    status: str
    ollama: bool
    browser: str
    workers: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    "ResearchRequest",
    "ResearchResponse",
    "ResearchSummary",
    "StoredReport",
    "StoredResponse",
]
