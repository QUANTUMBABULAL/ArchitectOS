"""
Deterministic pre-LLM routing for incoming requests.

FastRouter inspects a request with cheap ordered rules and decides where
it should go. Its purpose is to keep the local model off the critical path
for requests whose destination is obvious. A greeting, a ``/status``
command, or ``12 * 7`` does not need a language model to be understood,
and consulting one costs seconds on a local model.

Design
------
Routing is a list of :class:`Rule` values evaluated in order; the first
match wins. Each rule carries a confidence. When the winning confidence
is below the router's threshold the request is escalated to the Decision
Engine, which is the existing semantic classifier. The router therefore
never *replaces* the brain — it only spares it work it does not need.

Rules are supplied at construction time rather than hardcoded in the
matching loop, so new request classes can be added without modifying
FastRouter itself, and callers can inject their own rule sets in tests.

The router is pure: it performs no I/O, holds no async code, and depends
only on the standard library plus project exceptions and logging. That
makes it cheap to call on every request and trivial to unit test.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Pattern, Sequence

from src.logger import get_logger

DEFAULT_CONFIDENCE_THRESHOLD = 0.7


class RouteTarget(str, Enum):
    """
    Destinations a request can be routed to.

    Attributes:
        COMMAND: A built-in operator command such as help or status,
            handled by the shell without any model.
        GREETING: Conversational filler answered from a canned reply.
        CALCULATE: Pure arithmetic evaluated locally.
        LOCAL_ANSWER: Answered by the local model without research.
        RESEARCH: Sent to the research orchestrator for planning and
            external consultation.
        BROWSER: Requires the browser stack specifically.
        DECISION_ENGINE: Ambiguous; the semantic classifier decides.
    """

    COMMAND = "command"
    GREETING = "greeting"
    CALCULATE = "calculate"
    LOCAL_ANSWER = "local_answer"
    RESEARCH = "research"
    BROWSER = "browser"
    DECISION_ENGINE = "decision_engine"

    @property
    def uses_model(self) -> bool:
        """
        Return whether reaching this target involves a model call.

        Returns:
            True when the target requires local or remote inference.
        """
        return self in {
            RouteTarget.LOCAL_ANSWER,
            RouteTarget.RESEARCH,
            RouteTarget.DECISION_ENGINE,
        }


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """
    Outcome of routing one request.

    Attributes:
        target: Selected destination.
        confidence: Confidence in the selection, between 0.0 and 1.0.
        rule: Name of the rule that matched, or ``default`` when none did.
        reason: Short human-readable explanation.
        payload: Rule-extracted data, such as the arithmetic expression
            for a CALCULATE decision.
        escalated: True when the match was below threshold and the
            request was redirected to the Decision Engine.
    """

    target: RouteTarget
    confidence: float
    rule: str
    reason: str
    payload: Optional[str] = None
    escalated: bool = False

    @property
    def avoids_model(self) -> bool:
        """
        Return whether this decision resolves without inference.

        Returns:
            True when no model call is required.
        """
        return not self.target.uses_model


@dataclass(frozen=True, slots=True)
class Rule:
    """
    One routing rule.

    Attributes:
        name: Stable identifier used in logs and metrics.
        target: Destination applied when the rule matches.
        confidence: Confidence contributed by this rule.
        matcher: Predicate evaluated against the normalized request.
        reason: Explanation recorded on the decision.
        extract: Optional extractor producing a payload from the original
            (non-normalized) request text.
    """

    name: str
    target: RouteTarget
    confidence: float
    matcher: Callable[[str], bool]
    reason: str
    extract: Optional[Callable[[str], Optional[str]]] = None


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

_GREETINGS: frozenset[str] = frozenset(
    {
        "hi", "hey", "hello", "yo", "sup", "howdy", "hiya",
        "thanks", "thank you", "thx", "ty", "cheers",
        "ok", "okay", "cool", "nice", "great", "got it",
        "bye", "goodbye", "see you", "later",
        "good morning", "good afternoon", "good evening", "good night",
        "how are you", "how's it going", "what's up",
    }
)

_COMMAND_PATTERN: Pattern[str] = re.compile(
    r"^/(help|status|exit|quit|model|clear|reset|version)\b"
)

_BARE_COMMANDS: frozenset[str] = frozenset(
    {"help", "status", "exit", "quit", "version"}
)

# Arithmetic only: digits, operators, parentheses, decimal points, spaces.
# Requires at least one operator so a bare number is not treated as a sum.
_ARITHMETIC_PATTERN: Pattern[str] = re.compile(
    r"^[\d\s().+\-*/%^]+$"
)
_ARITHMETIC_OPERATOR: Pattern[str] = re.compile(r"[+\-*/%^]")

_CALC_PREFIX: Pattern[str] = re.compile(
    r"^(?:calc(?:ulate)?|compute|what\s+is|what's|how\s+much\s+is|eval)\b"
    r"[\s:]*(?P<expr>[\d\s().+\-*/%^]+)$"
)

_BROWSER_PATTERN: Pattern[str] = re.compile(
    r"\b(?:open|navigate\s+to|go\s+to|visit|load|screenshot(?:\s+of)?|"
    r"click|scrape|fill\s+in|log\s+in\s+to)\b"
)
_BROWSER_CONTEXT: Pattern[str] = re.compile(
    r"\b(?:browser|chrome|chatgpt|tab|page|url|website|site)\b"
    r"|https?://|www\.|\.com\b|\.org\b|\.io\b|\.net\b"
)

_SHOPPING_PATTERN: Pattern[str] = re.compile(
    r"\b(?:buy|purchase|order|shop|shopping|price\s+of|prices|pricing|"
    r"cheapest|cheaper|deal|deals|discount|coupon|in\s+stock|"
    r"best\s+(?:budget\s+)?(?:laptop|phone|monitor|headphones|gpu|"
    r"keyboard|mouse|camera|tv)|under\s*[$£€]\s*\d|"
    r"[$£€]\s*\d+(?:\s|$))\b"
)

_WEB_SEARCH_PATTERN: Pattern[str] = re.compile(
    r"\b(?:search\s+(?:for|the\s+web)|look\s+up|google|"
    r"latest|newest|recent|current(?:ly)?|today|this\s+week|"
    r"news|headlines|weather|release[sd]?\s+(?:date|version)|"
    r"who\s+is\s+(?:the\s+)?current|what\s+happened)\b"
    r"|\b20[2-9]\d\b"
)

_RESEARCH_PATTERN: Pattern[str] = re.compile(
    r"\b(?:research|compare|comparison|versus|vs\.?|"
    r"pros\s+and\s+cons|trade-?offs?|benchmark|state\s+of\s+the\s+art|"
    r"evaluate|analy[sz]e|investigate|survey|alternatives?\s+to|"
    r"recommend|which\s+is\s+better)\b"
)

_SHORT_WORD_LIMIT = 4


def _normalize(text: str) -> str:
    """
    Normalize request text for matching.

    Args:
        text: Raw request text.

    Returns:
        Lower-cased text with collapsed whitespace and trailing
        punctuation removed.
    """
    collapsed = " ".join((text or "").lower().split())
    return collapsed.rstrip("?!.,;")


def _extract_expression(text: str) -> Optional[str]:
    """
    Extract an arithmetic expression from request text.

    Handles both a bare expression and a natural-language prefix such as
    ``"what is 2+2"``.

    Args:
        text: Raw request text.

    Returns:
        Expression text, or None when none is present.
    """
    stripped = (text or "").strip().rstrip("?!.,;")

    if _ARITHMETIC_PATTERN.match(stripped) and _ARITHMETIC_OPERATOR.search(
        stripped
    ):
        return stripped.replace("^", "**")

    match = _CALC_PREFIX.match(stripped.lower())
    if match:
        expression = match.group("expr").strip()
        if expression and _ARITHMETIC_OPERATOR.search(expression):
            return expression.replace("^", "**")
    return None


def default_rules() -> tuple[Rule, ...]:
    """
    Build the default rule set in evaluation order.

    Order matters: the most specific and cheapest checks come first, and
    the first match wins. Commands precede greetings so ``/help`` is not
    read as conversational; arithmetic precedes research so ``2 vs 3``
    style text does not become a research task.

    Returns:
        Ordered rules.
    """
    return (
        Rule(
            name="operator_command",
            target=RouteTarget.COMMAND,
            confidence=1.0,
            matcher=lambda text: bool(_COMMAND_PATTERN.match(text))
            or text in _BARE_COMMANDS,
            reason="Recognized built-in operator command",
        ),
        Rule(
            name="greeting",
            target=RouteTarget.GREETING,
            confidence=1.0,
            matcher=lambda text: text in _GREETINGS,
            reason="Recognized conversational greeting",
        ),
        Rule(
            name="arithmetic",
            target=RouteTarget.CALCULATE,
            confidence=0.99,
            matcher=lambda text: _extract_expression(text) is not None,
            reason="Pure arithmetic evaluated locally",
            extract=_extract_expression,
        ),
        Rule(
            name="browser_automation",
            target=RouteTarget.BROWSER,
            confidence=0.85,
            matcher=lambda text: bool(_BROWSER_PATTERN.search(text))
            and bool(_BROWSER_CONTEXT.search(text)),
            reason="Browser action against a specific page or site",
        ),
        Rule(
            name="shopping",
            target=RouteTarget.RESEARCH,
            confidence=0.9,
            matcher=lambda text: bool(_SHOPPING_PATTERN.search(text)),
            reason="Product or price lookup requires current web data",
        ),
        Rule(
            name="web_search",
            target=RouteTarget.RESEARCH,
            confidence=0.9,
            matcher=lambda text: bool(_WEB_SEARCH_PATTERN.search(text)),
            reason="Request depends on current information",
        ),
        Rule(
            name="research_intent",
            target=RouteTarget.RESEARCH,
            confidence=0.85,
            matcher=lambda text: bool(_RESEARCH_PATTERN.search(text)),
            reason="Explicit comparison or investigation intent",
        ),
        Rule(
            name="short_phrase",
            target=RouteTarget.LOCAL_ANSWER,
            confidence=0.75,
            matcher=lambda text: 0 < len(text.split()) <= _SHORT_WORD_LIMIT,
            reason="Too short to require multi-source research",
        ),
    )


class FastRouter:
    """
    Rule-based router evaluated before any model call.

    The router is deliberately conservative: it claims a request only when
    a rule matches with confidence at or above the threshold. Everything
    else is escalated to the Decision Engine, so behaviour degrades to the
    previous architecture rather than to a wrong answer.

    Routing counts are tracked so the proportion of requests avoiding the
    model is observable at runtime.
    """

    def __init__(
        self,
        rules: Optional[Sequence[Rule]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initialize the router.

        Args:
            rules: Ordered rules to evaluate. Defaults to
                :func:`default_rules`.
            confidence_threshold: Minimum confidence required to accept a
                rule's target. Matches below this escalate to the
                Decision Engine.
        """
        self._rules: tuple[Rule, ...] = tuple(
            rules if rules is not None else default_rules()
        )
        self._threshold = max(0.0, min(1.0, confidence_threshold))
        self._counts: Counter[str] = Counter()
        self._logger = get_logger(__name__)

    @property
    def rules(self) -> tuple[Rule, ...]:
        """
        Return the active rules in evaluation order.

        Returns:
            Ordered rules.
        """
        return self._rules

    @property
    def confidence_threshold(self) -> float:
        """
        Return the acceptance threshold.

        Returns:
            Minimum confidence required to accept a rule match.
        """
        return self._threshold

    @property
    def counts(self) -> dict[str, int]:
        """
        Return how many requests were routed to each target.

        Returns:
            Mapping of target value to count.
        """
        return dict(self._counts)

    @property
    def model_avoidance_rate(self) -> Optional[float]:
        """
        Return the share of routed requests that avoided inference.

        Returns:
            Ratio between 0.0 and 1.0, or None when nothing was routed.
        """
        total = sum(self._counts.values())
        if total == 0:
            return None
        avoided = sum(
            count
            for target, count in self._counts.items()
            if not RouteTarget(target).uses_model
        )
        return avoided / total

    def route(self, request: str) -> RoutingDecision:
        """
        Route one request.

        Args:
            request: Raw request text.

        Returns:
            Routing decision. Empty input and unmatched input both resolve
            to the Decision Engine rather than raising, so the caller has
            a single code path.
        """
        normalized = _normalize(request)

        if not normalized:
            return self._record(
                RoutingDecision(
                    target=RouteTarget.DECISION_ENGINE,
                    confidence=0.0,
                    rule="default",
                    reason="Empty request",
                )
            )

        for rule in self._rules:
            if not rule.matcher(normalized):
                continue

            if rule.confidence < self._threshold:
                return self._record(
                    RoutingDecision(
                        target=RouteTarget.DECISION_ENGINE,
                        confidence=rule.confidence,
                        rule=rule.name,
                        reason=(
                            f"Rule '{rule.name}' matched but confidence "
                            f"{rule.confidence:.2f} is below threshold "
                            f"{self._threshold:.2f}"
                        ),
                        escalated=True,
                    )
                )

            payload = rule.extract(request) if rule.extract else None
            return self._record(
                RoutingDecision(
                    target=rule.target,
                    confidence=rule.confidence,
                    rule=rule.name,
                    reason=rule.reason,
                    payload=payload,
                )
            )

        return self._record(
            RoutingDecision(
                target=RouteTarget.DECISION_ENGINE,
                confidence=0.0,
                rule="default",
                reason="No rule matched; deferring to semantic classifier",
            )
        )

    def _record(self, decision: RoutingDecision) -> RoutingDecision:
        """
        Log and count a routing decision.

        Args:
            decision: Decision to record.

        Returns:
            The same decision, for call-site convenience.
        """
        self._counts[decision.target.value] += 1
        self._logger.info(
            "Routed to %s via rule '%s' (confidence %.2f, model=%s): %s",
            decision.target.value,
            decision.rule,
            decision.confidence,
            "yes" if decision.target.uses_model else "no",
            decision.reason,
        )
        return decision


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "FastRouter",
    "Rule",
    "RouteTarget",
    "RoutingDecision",
    "default_rules",
]
