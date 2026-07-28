"""
Tests for Chrome launch failure classification.

These tests encode behaviour verified against Chromium source: the
process-singleton hand-off prints a "used existing browser" message and
exits with a NORMAL status code, so it must be classified as a distinct,
non-retryable failure rather than as a generic launch error.
"""

from __future__ import annotations

from src.browser.launch_diagnostics import (
    LaunchFailureKind,
    classify_launch_failure,
    detect_handoff,
)


# Reproduces the shape of a real Playwright failure for this case.
HANDOFF_OUTPUT = (
    "browserType.launchPersistentContext: Failed to launch the browser "
    "process.\n"
    "<launching> C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe "
    "--user-data-dir=C:\\data\\chrome-profiles\\automation about:blank\n"
    "<launched> pid=25256\n"
    "[pid=25256][out] Opening in existing browser session.\n"
    "[pid=25256] <process did exit: exitCode=0, signal=null>"
)


class TestHandoffDetection:
    """Detection of the process-singleton hand-off."""

    def test_classifies_handoff(self) -> None:
        """Hand-off output is classified as SINGLETON_HANDOFF."""
        diagnosis = classify_launch_failure(HANDOFF_OUTPUT)
        assert diagnosis.kind is LaunchFailureKind.SINGLETON_HANDOFF
        assert diagnosis.is_handoff is True

    def test_handoff_is_not_retryable(self) -> None:
        """A hand-off is deterministic, so retrying must not be advised."""
        assert classify_launch_failure(HANDOFF_OUTPUT).retryable is False

    def test_predicate_agrees_with_classifier(self) -> None:
        """The narrow predicate matches the full classification."""
        assert detect_handoff(HANDOFF_OUTPUT) is True
        assert detect_handoff("some unrelated failure") is False

    def test_detection_is_case_insensitive(self) -> None:
        """Signatures match regardless of casing."""
        assert detect_handoff("OPENING IN EXISTING BROWSER SESSION.") is True

    def test_remedy_mentions_user_data_dir(self) -> None:
        """The directory is echoed back so the error is actionable."""
        diagnosis = classify_launch_failure(
            HANDOFF_OUTPUT, user_data_dir="C:\\data\\automation"
        )
        assert "C:\\data\\automation" in diagnosis.summary


class TestOtherFailureKinds:
    """Classification of the remaining known failure categories."""

    def test_profile_in_use(self) -> None:
        """An explicit profile-lock message is distinguished from hand-off."""
        diagnosis = classify_launch_failure(
            "The profile appears to be in use by another Chromium process"
        )
        assert diagnosis.kind is LaunchFailureKind.PROFILE_IN_USE
        assert diagnosis.retryable is True

    def test_missing_executable(self) -> None:
        """A missing binary is classified distinctly."""
        diagnosis = classify_launch_failure(
            "Chromium distribution 'chrome' is not found. Executable "
            "doesn't exist at /usr/bin/chrome"
        )
        assert diagnosis.kind is LaunchFailureKind.EXECUTABLE_NOT_FOUND
        assert diagnosis.retryable is False

    def test_missing_dependencies(self) -> None:
        """Missing shared libraries are classified distinctly."""
        diagnosis = classify_launch_failure(
            "error while loading shared libraries: libnss3.so"
        )
        assert diagnosis.kind is LaunchFailureKind.MISSING_DEPENDENCIES

    def test_timeout_is_retryable(self) -> None:
        """A launch timeout may succeed on retry."""
        diagnosis = classify_launch_failure("Timeout 30000ms exceeded")
        assert diagnosis.kind is LaunchFailureKind.TIMEOUT
        assert diagnosis.retryable is True

    def test_unknown_makes_no_claim(self) -> None:
        """Unrecognized text is not guessed at."""
        diagnosis = classify_launch_failure("something entirely novel")
        assert diagnosis.kind is LaunchFailureKind.UNKNOWN
        assert diagnosis.retryable is False

    def test_empty_input_is_unknown(self) -> None:
        """Empty and None-like input never raises."""
        assert classify_launch_failure("").kind is LaunchFailureKind.UNKNOWN
        assert detect_handoff("") is False


class TestPrecedence:
    """Signature precedence when multiple categories could match."""

    def test_handoff_wins_over_timeout(self) -> None:
        """
        Playwright wraps hand-off output in a message that also contains
        the word 'timeout'. The more specific cause must win, otherwise a
        deterministic failure is misreported as retryable.
        """
        text = HANDOFF_OUTPUT + "\nTimeout 30000ms exceeded"
        assert (
            classify_launch_failure(text).kind
            is LaunchFailureKind.SINGLETON_HANDOFF
        )
