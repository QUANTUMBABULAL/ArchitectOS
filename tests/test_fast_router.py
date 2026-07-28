"""
Tests for deterministic pre-inference routing.

The router's value is entirely in what it prevents: a request routed to
COMMAND, GREETING, or CALCULATE must never reach a model. These tests
assert that property directly, plus the escalation contract that keeps
behaviour safe when no rule is confident.
"""

from __future__ import annotations

import pytest

from src.routing import (
    FastRouter,
    RouteTarget,
    Rule,
    default_rules,
)


@pytest.fixture()
def router() -> FastRouter:
    """Return a router with the default rule set."""
    return FastRouter()


class TestInstantPaths:
    """Requests that must resolve with no inference."""

    @pytest.mark.parametrize(
        "text",
        ["hi", "Hello", "  hey  ", "thanks", "good morning", "bye", "OK"],
    )
    def test_greetings(self, router: FastRouter, text: str) -> None:
        """Greetings route to the canned reply."""
        decision = router.route(text)
        assert decision.target is RouteTarget.GREETING
        assert decision.avoids_model is True

    @pytest.mark.parametrize(
        "text",
        ["/help", "/status", "/exit", "/model qwen3:4b", "help", "status"],
    )
    def test_commands(self, router: FastRouter, text: str) -> None:
        """Operator commands route to the command handler."""
        decision = router.route(text)
        assert decision.target is RouteTarget.COMMAND
        assert decision.avoids_model is True

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2+2", "2+2"),
            ("12 * 7", "12 * 7"),
            ("(3 + 4) / 2", "(3 + 4) / 2"),
            ("what is 15 * 3", "15 * 3"),
            ("calculate 100 / 4", "100 / 4"),
            ("2^8", "2**8"),
        ],
    )
    def test_arithmetic(
        self,
        router: FastRouter,
        text: str,
        expected: str,
    ) -> None:
        """Arithmetic routes locally and yields the expression payload."""
        decision = router.route(text)
        assert decision.target is RouteTarget.CALCULATE
        assert decision.payload == expected
        assert decision.avoids_model is True

    def test_bare_number_is_not_arithmetic(self, router: FastRouter) -> None:
        """A number with no operator is not a calculation."""
        assert router.route("42").target is not RouteTarget.CALCULATE

    def test_greeting_precedes_short_phrase(
        self,
        router: FastRouter,
    ) -> None:
        """
        Rule order matters: 'hi' matches both the greeting rule and the
        short-phrase rule, and the greeting rule must win because it needs
        no model at all.
        """
        assert router.route("hi").target is RouteTarget.GREETING


class TestResearchRouting:
    """Requests that require external data."""

    @pytest.mark.parametrize(
        "text",
        [
            "buy a mechanical keyboard",
            "price of an rtx 4090",
            "cheapest flight to tokyo",
            "best budget laptop under $800",
            "any deals on monitors",
        ],
    )
    def test_shopping(self, router: FastRouter, text: str) -> None:
        """Product and price intent routes to research."""
        assert router.route(text).target is RouteTarget.RESEARCH

    @pytest.mark.parametrize(
        "text",
        [
            "search for python 3.13 release notes",
            "what is the latest version of postgres",
            "look up today's weather in berlin",
            "who is the current ceo of intel",
            "what happened at the 2026 olympics",
        ],
    )
    def test_web_search(self, router: FastRouter, text: str) -> None:
        """Current-information intent routes to research."""
        assert router.route(text).target is RouteTarget.RESEARCH

    @pytest.mark.parametrize(
        "text",
        [
            "compare rust and go for web servers",
            "pros and cons of microservices",
            "benchmark sqlite against duckdb",
            "which is better, tabs or spaces",
        ],
    )
    def test_research_intent(self, router: FastRouter, text: str) -> None:
        """Explicit comparison intent routes to research."""
        assert router.route(text).target is RouteTarget.RESEARCH


class TestBrowserRouting:
    """Requests that name a browser action against a page."""

    @pytest.mark.parametrize(
        "text",
        [
            "open chatgpt.com",
            "navigate to https://example.com",
            "take a screenshot of the page",
            "go to the browser and click submit",
        ],
    )
    def test_browser_actions(self, router: FastRouter, text: str) -> None:
        """Browser verbs plus page context route to the browser."""
        assert router.route(text).target is RouteTarget.BROWSER

    def test_verb_without_context_is_not_browser(
        self,
        router: FastRouter,
    ) -> None:
        """
        A browser verb alone is ambiguous. 'open a bank account' must not
        be mistaken for automation.
        """
        assert (
            router.route("open a bank account for my business").target
            is not RouteTarget.BROWSER
        )


class TestEscalation:
    """Behaviour when no rule is confident enough."""

    def test_unmatched_goes_to_decision_engine(
        self,
        router: FastRouter,
    ) -> None:
        """Unrecognized input defers to the semantic classifier."""
        decision = router.route(
            "refactor the memory layer to use a write-ahead log"
        )
        assert decision.target is RouteTarget.DECISION_ENGINE
        assert decision.rule == "default"

    def test_empty_input_goes_to_decision_engine(
        self,
        router: FastRouter,
    ) -> None:
        """Empty input never raises."""
        decision = router.route("")
        assert decision.target is RouteTarget.DECISION_ENGINE
        assert decision.confidence == 0.0

    def test_low_confidence_rule_escalates(self) -> None:
        """
        A rule below the threshold escalates rather than deciding. This is
        the safety property: raising the threshold degrades to the old
        architecture instead of producing wrong routes.
        """
        rules = (
            Rule(
                name="unsure",
                target=RouteTarget.RESEARCH,
                confidence=0.4,
                matcher=lambda text: True,
                reason="deliberately unsure",
            ),
        )
        router = FastRouter(rules=rules, confidence_threshold=0.7)
        decision = router.route("anything at all")

        assert decision.target is RouteTarget.DECISION_ENGINE
        assert decision.escalated is True
        assert decision.rule == "unsure"

    def test_threshold_is_clamped(self) -> None:
        """Out-of-range thresholds are clamped, not rejected."""
        assert FastRouter(confidence_threshold=5.0).confidence_threshold == 1.0
        assert FastRouter(confidence_threshold=-1.0).confidence_threshold == 0.0


class TestObservability:
    """Counts and rates used for the routing report."""

    def test_counts_accumulate(self, router: FastRouter) -> None:
        """Each routed request increments its target count."""
        router.route("hi")
        router.route("hello")
        router.route("compare a and b")

        counts = router.counts
        assert counts[RouteTarget.GREETING.value] == 2
        assert counts[RouteTarget.RESEARCH.value] == 1

    def test_avoidance_rate(self, router: FastRouter) -> None:
        """The rate reflects the share of requests skipping inference."""
        router.route("hi")
        router.route("2+2")
        router.route("compare a and b")

        assert router.model_avoidance_rate == pytest.approx(2 / 3)

    def test_avoidance_rate_is_none_when_idle(
        self,
        router: FastRouter,
    ) -> None:
        """No requests means no rate rather than a misleading zero."""
        assert router.model_avoidance_rate is None


class TestRuleSet:
    """Properties of the default rule configuration."""

    def test_rules_are_ordered_and_named_uniquely(self) -> None:
        """Rule names are unique, so logs identify a rule unambiguously."""
        names = [rule.name for rule in default_rules()]
        assert len(names) == len(set(names))

    def test_custom_rules_replace_defaults(self) -> None:
        """Injected rules fully replace the defaults."""
        rules = (
            Rule(
                name="everything_is_research",
                target=RouteTarget.RESEARCH,
                confidence=1.0,
                matcher=lambda text: True,
                reason="test rule",
            ),
        )
        router = FastRouter(rules=rules)
        assert router.route("hi").target is RouteTarget.RESEARCH

    def test_target_model_flags(self) -> None:
        """Instant targets are marked as not using a model."""
        assert RouteTarget.GREETING.uses_model is False
        assert RouteTarget.COMMAND.uses_model is False
        assert RouteTarget.CALCULATE.uses_model is False
        assert RouteTarget.RESEARCH.uses_model is True
        assert RouteTarget.DECISION_ENGINE.uses_model is True
