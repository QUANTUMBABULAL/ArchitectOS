"""
Tests for classification without model calls, and for request metrics.

The decision engine must resolve unambiguous input locally. Every model
call costs seconds on a local 4B model, so a fast path that silently
regressed into calling the model would reintroduce the latency problem
these tests exist to prevent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest

from src.brain.decision_engine import DecisionEngine, TaskComplexity
from src.brain.ollama_client import RequestMetrics
from src.config import get_settings


class RecordingClient:
    """
    Test double recording every generate call.

    Substitutes for OllamaClient so tests assert on whether the model
    would have been consulted, without needing a server.
    """

    def __init__(self, response: str = '{"complexity":"complex"}') -> None:
        """
        Initialize the double.

        Args:
            response: Canned JSON response returned by generate.
        """
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self.last_metrics: Optional[RequestMetrics] = None

    async def generate(self, **kwargs: Any) -> str:
        """Record the call and return the canned response."""
        self.calls.append(kwargs)
        return self._response

    async def chat(self, *args: Any, **kwargs: Any) -> str:
        """Return a canned chat reply."""
        return "local answer"

    def set_model(self, model: str) -> None:
        """Accept a model switch."""

    async def health_check(self) -> bool:
        """Report healthy."""
        return True


def make_engine(response: str = '{"complexity":"complex"}'):
    """
    Build an engine wired to a recording client.

    Args:
        response: Canned model response.

    Returns:
        Tuple of engine and its recording client.
    """
    client = RecordingClient(response)
    engine = DecisionEngine(client, settings=get_settings())  # type: ignore[arg-type]
    return engine, client


class TestFastPath:
    """Inputs that must never reach the model."""

    @pytest.mark.parametrize(
        "text",
        ["hi", "Hello", "thanks", "  hey  ", "good morning", "ok", "PING"],
    )
    def test_greetings_skip_the_model(self, text: str) -> None:
        """Conversational input is classified locally."""
        engine, client = make_engine()
        result = asyncio.run(engine.classify_complexity(text))

        assert result.complexity is TaskComplexity.SIMPLE
        assert result.source == "fast_path"
        assert client.calls == []

    def test_trailing_punctuation_still_matches(self) -> None:
        """Punctuation does not defeat the fast path."""
        engine, client = make_engine()
        result = asyncio.run(engine.classify_complexity("hello!"))
        assert result.source == "fast_path"
        assert client.calls == []

    def test_short_phrase_without_signal_skips_model(self) -> None:
        """A very short phrase cannot require multi-source research."""
        engine, client = make_engine()
        result = asyncio.run(engine.classify_complexity("define entropy"))

        assert result.complexity is TaskComplexity.SIMPLE
        assert result.source == "fast_path"
        assert client.calls == []

    def test_short_phrase_with_signal_reaches_model(self) -> None:
        """A complexity signal defers the decision to the model."""
        engine, client = make_engine()
        result = asyncio.run(engine.classify_complexity("compare X and Y"))

        assert result.source == "model"
        assert len(client.calls) == 1

    def test_long_input_reaches_model(self) -> None:
        """Longer input is ambiguous and must be classified by the model."""
        engine, client = make_engine()
        asyncio.run(
            engine.classify_complexity(
                "what are the licensing implications of shipping this"
            )
        )
        assert len(client.calls) == 1


class TestBoundedGeneration:
    """The model call must always be bounded."""

    def test_call_is_capped_and_schema_constrained(self) -> None:
        """
        Classification must pass a token cap, a schema, and a short
        timeout. Without these the call can run until the read timeout.
        """
        engine, client = make_engine()
        asyncio.run(engine.classify_complexity("compare X and Y"))

        call = client.calls[0]
        assert call["num_predict"] > 0
        assert call["response_format"]["type"] == "object"
        assert call["timeout_seconds"] <= get_settings().ollama_timeout
        assert call["operation"] == "classify"


class TestCaching:
    """Repeated classification must not re-consult the model."""

    def test_second_identical_call_is_cached(self) -> None:
        """An identical task is served from cache."""
        engine, client = make_engine()
        task = "compare postgres and mysql"

        first = asyncio.run(engine.classify_complexity(task))
        second = asyncio.run(engine.classify_complexity(task))

        assert len(client.calls) == 1
        assert second.complexity is first.complexity

    def test_cache_is_case_and_space_insensitive(self) -> None:
        """Trivial formatting differences hit the same cache entry."""
        engine, client = make_engine()
        asyncio.run(engine.classify_complexity("compare A and B"))
        asyncio.run(engine.classify_complexity("  COMPARE   A and B "))

        assert len(client.calls) == 1

    def test_clear_cache_forces_recompute(self) -> None:
        """Clearing the cache allows a fresh model call."""
        engine, client = make_engine()
        asyncio.run(engine.classify_complexity("compare A and B"))
        engine.clear_cache()
        asyncio.run(engine.classify_complexity("compare A and B"))

        assert len(client.calls) == 2


class TestHeuristicFallback:
    """Fallback only when the model genuinely fails."""

    def test_malformed_response_falls_back(self) -> None:
        """Unparseable output degrades to the heuristic, not an error."""
        engine, _ = make_engine(response="not json at all")
        result = asyncio.run(
            engine.classify_complexity("compare A and B please")
        )

        assert result.used_fallback is True
        assert result.source == "heuristic"

    def test_fallback_still_detects_complexity(self) -> None:
        """The heuristic recognizes research signals."""
        engine, _ = make_engine(response="garbage")
        result = asyncio.run(
            engine.classify_complexity("research the latest benchmarks")
        )
        assert result.complexity is TaskComplexity.COMPLEX


class TestRequestMetrics:
    """Metrics reporting used by the timing logs."""

    def test_rate_computed_from_counters(self) -> None:
        """Token rate is derived from eval count and duration."""
        metrics = RequestMetrics(
            operation="classify",
            model="qwen3:4b",
            wall_seconds=2.0,
            prompt_chars=40,
            prompt_tokens=12,
            response_tokens=20,
            eval_seconds=2.0,
        )
        assert metrics.tokens_per_second == pytest.approx(10.0)

    def test_rate_is_none_without_counters(self) -> None:
        """Missing counters yield no rate rather than a wrong one."""
        metrics = RequestMetrics(
            operation="classify",
            model="qwen3:4b",
            wall_seconds=1.0,
            prompt_chars=10,
        )
        assert metrics.tokens_per_second is None

    def test_describe_flags_truncation(self) -> None:
        """A capped generation is visible in the log line."""
        metrics = RequestMetrics(
            operation="decompose",
            model="qwen3:4b",
            wall_seconds=1.0,
            prompt_chars=10,
            truncated=True,
        )
        assert "TRUNCATED" in metrics.describe()
        assert "op=decompose" in metrics.describe()
