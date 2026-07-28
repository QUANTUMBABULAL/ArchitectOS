"""
Tests for navigation settling and reload-loop detection.

These cover the root cause of the original failure: a worker searching for
a prompt input while the page was still redirecting. The tracker decides
when a page has stopped moving, and it must not report "settled" during a
redirect chain — that mistake is what produced endless reloads and a
misleading composer-not-found timeout.
"""

from __future__ import annotations

import pytest

from src.browser.page_state import (
    DEFAULT_INPUT_SELECTORS,
    NavigationTracker,
    PageState,
    StabilizationFailure,
    StabilizationResult,
    compile_patterns,
    matches_any,
)


class TestPageStateSemantics:
    """What each state permits."""

    def test_only_input_ready_can_prompt(self) -> None:
        """
        The core guarantee: nothing but INPUT_READY may receive a prompt.
        """
        assert PageState.INPUT_READY.can_prompt is True
        for state in (
            PageState.UNKNOWN,
            PageState.LOADING,
            PageState.REDIRECTING,
            PageState.RELOAD_LOOP,
            PageState.LOGIN_PAGE,
            PageState.CHALLENGE_PAGE,
            PageState.CHAT_READY,
            PageState.ERROR,
        ):
            assert state.can_prompt is False

    def test_transient_states_are_not_settled(self) -> None:
        """Loading and redirecting are explicitly unsettled."""
        assert PageState.LOADING.is_settled is False
        assert PageState.REDIRECTING.is_settled is False
        assert PageState.INPUT_READY.is_settled is True

    def test_human_states(self) -> None:
        """Sign-in and challenge pages need a person."""
        assert PageState.LOGIN_PAGE.needs_human is True
        assert PageState.CHALLENGE_PAGE.needs_human is True
        assert PageState.CHAT_READY.needs_human is False

    def test_dashboard_labels(self) -> None:
        """Every state renders an operator-facing label."""
        assert PageState.LOADING.label == "LOADING..."
        assert PageState.CHAT_READY.label == "WAITING FOR INPUT"
        assert PageState.LOGIN_PAGE.label == "LOGIN_REQUIRED"
        assert PageState.INPUT_READY.label == "READY"
        for state in PageState:
            assert state.label


class TestSettling:
    """Deciding when the URL has stopped changing."""

    def test_not_settled_during_redirects(self) -> None:
        """
        A page moving through a redirect chain is never settled. This is
        the assertion that prevents searching for an input mid-redirect.
        """
        tracker = NavigationTracker(settle_seconds=2.0)
        tracker.record(0.0, "https://gemini.google.com/app")
        tracker.record(0.5, "https://accounts.google.com/signin")
        tracker.record(1.0, "https://accounts.google.com/oauth")
        tracker.record(1.5, "https://gemini.google.com/app")

        assert tracker.is_settled(1.5) is False

    def test_settles_after_the_window(self) -> None:
        """A stable URL across the full window is settled."""
        tracker = NavigationTracker(settle_seconds=2.0)
        for index in range(7):
            tracker.record(index * 0.5, "https://chatgpt.com/")

        assert tracker.is_settled(3.0) is True

    def test_not_settled_before_the_window_is_covered(self) -> None:
        """
        A single sample must not look settled. Otherwise a page observed
        once immediately after navigation would be treated as stable.
        """
        tracker = NavigationTracker(settle_seconds=2.0)
        tracker.record(5.0, "https://chatgpt.com/")

        assert tracker.is_settled(5.0) is False

    def test_late_change_unsettles(self) -> None:
        """A change inside the window resets settling."""
        tracker = NavigationTracker(settle_seconds=2.0)
        for index in range(6):
            tracker.record(index * 0.5, "https://chatgpt.com/")
        tracker.record(3.0, "https://chatgpt.com/c/abc")

        assert tracker.is_settled(3.0) is False

    def test_empty_tracker_is_not_settled(self) -> None:
        """No observations means no verdict."""
        assert NavigationTracker().is_settled(1.0) is False


class TestLoopDetection:
    """Recognising a page that will never settle."""

    def test_detects_excessive_transitions(self) -> None:
        """Too many navigations is a loop."""
        tracker = NavigationTracker(max_url_changes=4)
        for index in range(10):
            tracker.record(
                index * 0.2, f"https://example.com/page{index}"
            )

        assert tracker.is_looping() is True

    def test_detects_bouncing_between_two_urls(self) -> None:
        """
        The signature of an expired session: the page keeps returning to a
        URL it already left. Transition count alone would miss this if the
        limit were high.
        """
        tracker = NavigationTracker(max_url_changes=99, max_revisits=3)
        for index in range(10):
            tracker.record(
                index * 0.2,
                "https://gemini.google.com/app"
                if index % 2 == 0
                else "https://accounts.google.com/signin",
            )

        assert tracker.revisit_count() > 3
        assert tracker.is_looping() is True

    def test_normal_navigation_is_not_a_loop(self) -> None:
        """Ordinary forward navigation must not be flagged."""
        tracker = NavigationTracker(max_url_changes=8, max_revisits=3)
        tracker.record(0.0, "https://chatgpt.com/")
        tracker.record(0.5, "https://chatgpt.com/c/new")
        for index in range(6):
            tracker.record(1.0 + index * 0.5, "https://chatgpt.com/c/new")

        assert tracker.is_looping() is False

    def test_loop_summary_names_the_urls(self) -> None:
        """The summary is actionable, not just a count."""
        tracker = NavigationTracker()
        tracker.record(0.0, "https://gemini.google.com/app")
        tracker.record(0.5, "https://accounts.google.com/signin")

        summary = tracker.loop_summary()
        assert "accounts.google.com" in summary
        assert "gemini.google.com" in summary

    def test_transition_and_distinct_counts(self) -> None:
        """
        Counting distinguishes repeats from genuine changes. The sequence
        a,a,b,b,a contains two transitions (a to b, b to a) across two
        distinct URLs; consecutive repeats are not transitions.
        """
        tracker = NavigationTracker()
        for url in ("a", "a", "b", "b", "a"):
            tracker.record(0.0, url)

        assert tracker.transitions == 2
        assert tracker.distinct_urls == 2
        # Two separate visits to "a" — the revisit signal.
        assert tracker.revisit_count() == 2


class TestUrlPatterns:
    """Classifying settled pages by URL."""

    def test_matches_login_hosts(self) -> None:
        """Google's sign-in host is recognised."""
        patterns = compile_patterns([r"accounts\.google\.com"])
        assert matches_any(
            "https://accounts.google.com/signin/v2", patterns
        )

    def test_is_case_insensitive(self) -> None:
        """Host casing does not defeat matching."""
        patterns = compile_patterns([r"accounts\.google\.com"])
        assert matches_any("https://ACCOUNTS.GOOGLE.COM/x", patterns)

    def test_non_matching_url(self) -> None:
        """An unrelated URL does not match."""
        patterns = compile_patterns([r"accounts\.google\.com"])
        assert matches_any("https://chatgpt.com/", patterns) is False

    def test_invalid_pattern_is_skipped(self) -> None:
        """A malformed pattern is ignored rather than raising."""
        patterns = compile_patterns([r"[unclosed", r"valid"])
        assert len(patterns) == 1

    def test_empty_url_is_safe(self) -> None:
        """A missing URL never raises."""
        assert matches_any("", compile_patterns([r"x"])) is False


class TestInputSelectors:
    """Fallback selectors for prompt inputs."""

    def test_covers_the_common_shapes(self) -> None:
        """Textarea, contenteditable, and role=textbox are all covered."""
        joined = " ".join(DEFAULT_INPUT_SELECTORS)
        assert "textarea" in joined
        assert 'contenteditable="true"' in joined
        assert 'role="textbox"' in joined

    def test_selectors_are_unique(self) -> None:
        """No wasted lookups from duplicates."""
        assert len(DEFAULT_INPUT_SELECTORS) == len(
            set(DEFAULT_INPUT_SELECTORS)
        )


class TestStabilizationResult:
    """The verdict object."""

    def test_ok_only_for_input_ready(self) -> None:
        """Only an INPUT_READY result is usable."""
        assert StabilizationResult(state=PageState.INPUT_READY).ok is True
        assert StabilizationResult(state=PageState.CHAT_READY).ok is False

    def test_describe_includes_failure(self) -> None:
        """The log line names the structured cause."""
        described = StabilizationResult(
            state=PageState.RELOAD_LOOP,
            url="https://accounts.google.com/",
            failure=StabilizationFailure.AUTH_EXPIRED,
            url_changes=12,
        ).describe()

        assert "RELOAD_LOOP" in described
        assert "authentication_expired" in described
        assert "changes=12" in described
