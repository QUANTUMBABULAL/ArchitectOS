"""
Decision engine backed by the local Ollama model.

The DecisionEngine is the coordinator of the research operating system.
It is deliberately NOT the primary knowledge source: its job is deciding
WHEN external AI systems should be consulted and WHICH ones, answering
only simple tasks locally. Every model-backed decision has a deterministic
heuristic fallback because small local models cannot be trusted to always
produce valid structured output.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.config import Settings, get_settings
from src.exceptions import BrainError
from src.logger import get_logger
from src.prompts import (
    CLASSIFIER_SCHEMA,
    CLASSIFIER_SYSTEM_PROMPT,
    LOCAL_ANSWER_SYSTEM_PROMPT,
    classifier_prompt,
)

from .ollama_client import ChatMessage, OllamaClient


class TaskComplexity(str, Enum):
    """
    Complexity verdict for a research task.

    SIMPLE tasks are handled by the local model alone. COMPLEX tasks
    trigger consultation of external AI systems through workers.
    """

    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass(frozen=True, slots=True)
class ComplexityAssessment:
    """
    Result of classifying a task's complexity.

    Attributes:
        complexity: Complexity verdict.
        confidence: Verdict confidence between 0.0 and 1.0.
        reasoning: Short explanation of the verdict.
        used_fallback: True when the heuristic fallback produced the
            verdict instead of the local model.
        source: Which mechanism produced the verdict: ``model``,
            ``fast_path``, ``cache``, or ``heuristic``. Recorded so the
            proportion of requests that avoid the model is observable.
    """

    complexity: TaskComplexity
    confidence: float
    reasoning: str
    used_fallback: bool = False
    source: str = "model"


@dataclass(frozen=True, slots=True)
class ConsultationDecision:
    """
    Decision about consulting external AI systems.

    Attributes:
        should_consult: Whether external consultation is needed.
        workers: Names of workers to consult, in priority order.
        rationale: Short explanation of the routing decision.
        assessment: Complexity assessment behind the decision.
    """

    should_consult: bool
    workers: list[str] = field(default_factory=list)
    rationale: str = ""
    assessment: Optional[ComplexityAssessment] = None


_ROUTER_SYSTEM_PROMPT = (
    "You are the routing brain of a research operating system. Given a "
    "complex task and a list of available workers with their capability "
    "tags, choose which workers to consult. Respond ONLY with a JSON "
    'object of the form {"workers": ["name", ...], '
    '"rationale": "<one sentence>"}. Choose at least one worker.'
)

_COMPLEX_HINT_PATTERNS: tuple[str, ...] = (
    r"\bresearch\b",
    r"\bcompare\b",
    r"\banalyz\w*\b",
    r"\bevaluat\w*\b",
    r"\bdesign\b",
    r"\barchitect\w*\b",
    r"\blatest\b",
    r"\bcurrent\b",
    r"\brecent\b",
    r"\b20\d\d\b",
    r"\bpros and cons\b",
    r"\btrade-?offs?\b",
    r"\bstate of the art\b",
    r"\bbenchmark\w*\b",
    r"\bimplement\w*\b",
    r"\bwhy\b.+\bhow\b",
)

# Inputs that are unambiguously conversational. Sending these to a model
# costs seconds and cannot change the outcome, so they are decided
# locally. Matched only against the whole normalized input.
_TRIVIAL_INPUTS: frozenset[str] = frozenset(
    {
        "hi",
        "hey",
        "hello",
        "yo",
        "sup",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "cool",
        "nice",
        "bye",
        "goodbye",
        "good morning",
        "good evening",
        "good night",
        "how are you",
        "who are you",
        "what can you do",
        "help",
        "test",
        "ping",
    }
)

# Below this word count, with no complexity signal present, a task cannot
# plausibly require multi-source research.
_TRIVIAL_WORD_LIMIT = 4


class DecisionEngine:
    """
    Coordinator that classifies tasks and routes consultations.

    The engine keeps a bounded conversation history so multi-turn
    coordination has context, supports custom system prompts, and exposes
    the underlying client's health check and model switching.
    """

    def __init__(
        self,
        client: OllamaClient,
        system_prompt: Optional[str] = None,
        max_history_messages: int = 20,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the decision engine.

        Args:
            client: Ollama client used for all local inference.
            system_prompt: Optional system prompt for local answering.
            max_history_messages: Maximum retained conversation messages.
            settings: Optional application settings supplying generation
                caps and cache size.
        """
        if max_history_messages < 2:
            raise BrainError(
                "max_history_messages must be at least 2",
                code="BRAIN_HISTORY_TOO_SMALL",
            )

        self._client = client
        self._settings = settings or get_settings()
        self._system_prompt = system_prompt or LOCAL_ANSWER_SYSTEM_PROMPT
        self._max_history = max_history_messages
        self._history: list[ChatMessage] = []
        self._logger = get_logger(__name__)
        self._cache: OrderedDict[str, ComplexityAssessment] = OrderedDict()
        self._cache_limit = max(0, self._settings.classification_cache_size)

    @property
    def client(self) -> OllamaClient:
        """
        Return the underlying Ollama client.

        Returns:
            Ollama client.
        """
        return self._client

    @property
    def history(self) -> list[ChatMessage]:
        """
        Return a copy of the conversation history.

        Returns:
            Conversation messages.
        """
        return list(self._history)

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Replace the system prompt used for local answering.

        Args:
            system_prompt: New system prompt.

        Raises:
            BrainError: If the prompt is empty.
        """
        if not system_prompt or not system_prompt.strip():
            raise BrainError(
                "System prompt cannot be empty",
                code="BRAIN_SYSTEM_PROMPT_EMPTY",
            )
        self._system_prompt = system_prompt.strip()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    def switch_model(self, model: str) -> None:
        """
        Switch the local model.

        Args:
            model: Model name to activate.
        """
        self._client.set_model(model)

    async def health_check(self) -> bool:
        """
        Check whether local inference is available.

        Returns:
            True when the Ollama server is reachable.
        """
        return await self._client.health_check()

    async def classify_complexity(self, task: str) -> ComplexityAssessment:
        """
        Classify a task as simple or complex.

        Resolution order, cheapest first:

        1. A deterministic fast path for unambiguous inputs. Greetings and
           very short phrases cannot require multi-source research, so no
           model call is made.
        2. An in-memory cache of previous verdicts.
        3. The local model, constrained by a JSON schema and a hard token
           cap so the call completes in seconds.
        4. A keyword and length heuristic, if the model call fails.

        Args:
            task: Task description.

        Returns:
            Complexity assessment.

        Raises:
            BrainError: If the task is empty.
        """
        cleaned = self._require_task(task)

        fast = self._fast_path(cleaned)
        if fast is not None:
            self._logger.info(
                "Complexity resolved without a model call: %s (%s)",
                fast.complexity.value,
                fast.reasoning,
            )
            return fast

        cache_key = self._cache_key(cleaned)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            self._logger.debug(
                "Complexity served from cache: %s", cached.complexity.value
            )
            return cached

        try:
            raw = await self._client.generate(
                prompt=classifier_prompt(cleaned),
                system=CLASSIFIER_SYSTEM_PROMPT,
                options={"temperature": 0.0},
                num_predict=self._settings.ollama_classify_tokens,
                response_format=CLASSIFIER_SCHEMA,
                timeout_seconds=self._settings.ollama_fast_timeout,
                operation="classify",
            )
            data = self._parse_json_object(raw)
            verdict = str(data.get("complexity", "")).strip().lower()
            if verdict not in {
                TaskComplexity.SIMPLE.value,
                TaskComplexity.COMPLEX.value,
            }:
                raise ValueError(f"Invalid complexity verdict: {verdict!r}")

            assessment = ComplexityAssessment(
                complexity=TaskComplexity(verdict),
                confidence=self._clamp_confidence(data.get("confidence")),
                reasoning="Classified by the local model",
                source="model",
            )
            self._remember(cache_key, assessment)
            return assessment
        except Exception as exc:
            self._logger.warning(
                "Model-based complexity classification failed (%s); "
                "using heuristic fallback",
                exc,
            )
            return self._heuristic_assessment(cleaned)

    def _fast_path(self, task: str) -> Optional[ComplexityAssessment]:
        """
        Resolve unambiguous inputs without consulting the model.

        Only decides cases where a model call cannot plausibly change the
        outcome. Anything with a complexity signal, or longer than a short
        phrase, is deferred to the model.

        Args:
            task: Cleaned task description.

        Returns:
            Assessment when the input is unambiguous, otherwise None.
        """
        normalized = task.lower().strip().rstrip("?!.,")
        words = normalized.split()

        if normalized in _TRIVIAL_INPUTS:
            return ComplexityAssessment(
                complexity=TaskComplexity.SIMPLE,
                confidence=1.0,
                reasoning="Recognized conversational input",
                source="fast_path",
            )

        has_signal = any(
            re.search(pattern, normalized)
            for pattern in _COMPLEX_HINT_PATTERNS
        )

        if len(words) <= _TRIVIAL_WORD_LIMIT and not has_signal:
            return ComplexityAssessment(
                complexity=TaskComplexity.SIMPLE,
                confidence=0.8,
                reasoning=(
                    f"Only {len(words)} word(s) and no research signal"
                ),
                source="fast_path",
            )

        return None

    def _remember(
        self,
        cache_key: str,
        assessment: ComplexityAssessment,
    ) -> None:
        """
        Store a verdict in the bounded cache.

        Args:
            cache_key: Normalized cache key.
            assessment: Verdict to remember.
        """
        if self._cache_limit <= 0:
            return
        self._cache[cache_key] = assessment
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    @staticmethod
    def _cache_key(task: str) -> str:
        """
        Build a normalized cache key for a task.

        Args:
            task: Cleaned task description.

        Returns:
            Cache key insensitive to case and surrounding whitespace.
        """
        return " ".join(task.lower().split())

    def clear_cache(self) -> None:
        """Drop all cached complexity verdicts."""
        self._cache.clear()

    async def decide_consultation(
        self,
        task: str,
        available_workers: dict[str, frozenset[str]],
    ) -> ConsultationDecision:
        """
        Decide whether and which external AI systems to consult.

        Simple tasks are never routed externally. Complex tasks are routed
        to workers chosen by the local model from the available set, with
        an all-workers fallback when routing output is unusable.

        Args:
            task: Task description.
            available_workers: Mapping of worker name to capability tags.

        Returns:
            Consultation decision.

        Raises:
            BrainError: If the task is empty.
        """
        cleaned = self._require_task(task)
        assessment = await self.classify_complexity(cleaned)

        if assessment.complexity == TaskComplexity.SIMPLE:
            return ConsultationDecision(
                should_consult=False,
                workers=[],
                rationale=(
                    "Task classified as simple; handling locally. "
                    f"{assessment.reasoning}"
                ),
                assessment=assessment,
            )

        if not available_workers:
            return ConsultationDecision(
                should_consult=False,
                workers=[],
                rationale=(
                    "Task is complex but no workers are available; "
                    "answering locally as a fallback"
                ),
                assessment=assessment,
            )

        catalog = "\n".join(
            f"- {name}: {', '.join(sorted(capabilities)) or 'general'}"
            for name, capabilities in sorted(available_workers.items())
        )

        try:
            raw = await self._client.generate(
                prompt=(
                    f"Task: {cleaned}\n\nAvailable workers:\n{catalog}"
                ),
                system=_ROUTER_SYSTEM_PROMPT,
                options={"temperature": 0.0},
                num_predict=self._settings.ollama_classify_tokens,
                response_format={
                    "type": "object",
                    "properties": {
                        "workers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["workers"],
                },
                timeout_seconds=self._settings.ollama_fast_timeout,
                operation="route",
            )
            data = self._parse_json_object(raw)
            requested = data.get("workers", [])
            if not isinstance(requested, list):
                raise ValueError("workers field is not a list")

            workers = [
                str(name).strip().lower()
                for name in requested
                if str(name).strip().lower() in available_workers
            ]
            rationale = str(data.get("rationale", "")).strip()
        except Exception as exc:
            self._logger.warning(
                "Model-based worker routing failed (%s); consulting all "
                "available workers",
                exc,
            )
            workers = []
            rationale = ""

        if not workers:
            workers = sorted(available_workers)
            rationale = rationale or (
                "Routing fallback: consulting every available worker"
            )

        return ConsultationDecision(
            should_consult=True,
            workers=workers,
            rationale=rationale,
            assessment=assessment,
        )

    async def answer_locally(self, task: str) -> str:
        """
        Answer a simple task with the local model.

        The exchange is recorded in the bounded conversation history so
        follow-up coordination has context.

        Args:
            task: Task description.

        Returns:
            Local model answer.

        Raises:
            BrainError: If the task is empty or inference fails.
        """
        cleaned = self._require_task(task)

        messages = [ChatMessage(role="system", content=self._system_prompt)]
        messages.extend(self._history)
        messages.append(ChatMessage(role="user", content=cleaned))

        answer = await self._client.chat(messages, operation="answer_local")

        self._history.append(ChatMessage(role="user", content=cleaned))
        self._history.append(
            ChatMessage(role="assistant", content=answer)
        )
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return answer

    def _heuristic_assessment(self, task: str) -> ComplexityAssessment:
        """
        Classify complexity with a deterministic heuristic.

        Args:
            task: Cleaned task description.

        Returns:
            Complexity assessment marked as fallback.
        """
        lowered = task.lower()
        hits = sum(
            1
            for pattern in _COMPLEX_HINT_PATTERNS
            if re.search(pattern, lowered)
        )
        word_count = len(task.split())

        is_complex = hits >= 1 or word_count > 40
        complexity = (
            TaskComplexity.COMPLEX if is_complex else TaskComplexity.SIMPLE
        )
        reasoning = (
            f"Heuristic verdict: {hits} complexity keyword hit(s), "
            f"{word_count} words"
        )
        return ComplexityAssessment(
            complexity=complexity,
            confidence=0.5,
            reasoning=reasoning,
            used_fallback=True,
            source="heuristic",
        )

    @staticmethod
    def _require_task(task: str) -> str:
        """
        Validate and clean a task string.

        Args:
            task: Raw task description.

        Returns:
            Stripped task description.

        Raises:
            BrainError: If the task is empty.
        """
        if not task or not task.strip():
            raise BrainError(
                "Task cannot be empty",
                code="BRAIN_TASK_EMPTY",
            )
        return task.strip()

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        """
        Extract and parse the first JSON object in model output.

        Args:
            raw: Raw model output.

        Returns:
            Parsed JSON object.

        Raises:
            ValueError: If no valid JSON object can be extracted.
        """
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            raise ValueError("No JSON object found in model output")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not an object")
        return data

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        """
        Normalize a confidence value into [0.0, 1.0].

        Args:
            value: Raw confidence value from the model.

        Returns:
            Clamped confidence, defaulting to 0.5 when unusable.
        """
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, confidence))


__all__ = [
    "ComplexityAssessment",
    "ConsultationDecision",
    "DecisionEngine",
    "TaskComplexity",
]
