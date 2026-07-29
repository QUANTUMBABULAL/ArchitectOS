"""
Debate package: iterative multi-provider research.

DebateEngine runs successive rounds of consultation. ContradictionEngine
turns recommendation-level disagreement into follow-ups aimed at the
specific providers involved, so later rounds interrogate the actual point
of conflict rather than re-broadcasting the question.
"""

from .contradiction_engine import (
    ContradictionEngine,
    QuestionKind,
    TargetedQuestion,
)
from .debate_engine import DebateEngine, DebateOutcome, DebateRound, StopReason
from .report import render_report
from .synthesis import FinalAnswer, build_final_answer

__all__ = [
    "ContradictionEngine",
    "DebateEngine",
    "DebateOutcome",
    "DebateRound",
    "FinalAnswer",
    "QuestionKind",
    "StopReason",
    "TargetedQuestion",
    "build_final_answer",
    "render_report",
]
