"""
Classification of Chrome launch failures into actionable diagnoses.

Playwright surfaces browser launch problems as a single opaque exception
whose message embeds the browser's own stdout. Several distinct root
causes are therefore indistinguishable to callers unless the text is
parsed. This module performs that parsing in one place so the factory,
the manager, and the workers all interpret failures identically.

The most important case is the Chromium *process singleton hand-off*.
Chromium enforces one browser instance per user data directory. When a
launch discovers an existing instance for the same directory it notifies
that instance, prints a localized "used existing browser" message, and
returns a NORMAL exit code rather than an error code. A supervising
process that inspects only the exit status therefore observes an apparent
success with no usable browser. Detecting this explicitly is the
difference between an actionable error and a silent failure.

Reference: Chromium ``chrome/app/chrome_main_delegate.cc``, function
``AcquireProcessSingleton``, ``ProcessSingleton::PROCESS_NOTIFIED``
branch, which prints ``IDS_USED_EXISTING_BROWSER`` and returns
``CHROME_RESULT_CODE_NORMAL_EXIT_PROCESS_NOTIFIED``. The singleton is
keyed on the user data directory per ``chrome/browser/process_singleton.h``.

This module intentionally depends only on the standard library so it can
be unit tested without a browser, a Playwright install, or a display.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LaunchFailureKind(str, Enum):
    """
    Root-cause categories for a failed Chrome launch.

    Attributes are ordered from most specific to least specific; the
    classifier returns the first category whose signature matches.
    """

    SINGLETON_HANDOFF = "singleton_handoff"
    PROFILE_IN_USE = "profile_in_use"
    EXECUTABLE_NOT_FOUND = "executable_not_found"
    TIMEOUT = "timeout"
    MISSING_DEPENDENCIES = "missing_dependencies"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LaunchDiagnosis:
    """
    Structured interpretation of a Chrome launch failure.

    Attributes:
        kind: Root-cause category.
        summary: One-line description of what went wrong.
        remedy: Operator-facing guidance for resolving the cause.
        retryable: True when retrying the same launch unchanged could
            plausibly succeed. False for deterministic failures, where a
            retry only adds latency.
    """

    kind: LaunchFailureKind
    summary: str
    remedy: str
    retryable: bool

    @property
    def is_handoff(self) -> bool:
        """
        Return whether this diagnosis is a process-singleton hand-off.

        Returns:
            True when the launch deferred to an existing browser instance.
        """
        return self.kind is LaunchFailureKind.SINGLETON_HANDOFF


# Signatures are matched case-insensitively against the combined
# exception text and captured browser output.
_HANDOFF_SIGNATURES: tuple[str, ...] = (
    "opening in existing browser session",
    "used existing browser",
)

_PROFILE_IN_USE_SIGNATURES: tuple[str, ...] = (
    "profile appears to be in use",
    "the profile is already in use",
    "already in use by another instance",
    "failed to create a processsingleton",
    "singletonlock",
)

_EXECUTABLE_SIGNATURES: tuple[str, ...] = (
    "executable doesn't exist",
    "executable does not exist",
    "no such file or directory",
    "cannot find the path",
    "the system cannot find the file",
)

_TIMEOUT_SIGNATURES: tuple[str, ...] = (
    "timeout",
    "timed out",
)

_DEPENDENCY_SIGNATURES: tuple[str, ...] = (
    "error while loading shared libraries",
    "missing dependencies",
    "host system is missing dependencies",
)


def _matches(haystack: str, needles: tuple[str, ...]) -> bool:
    """
    Check whether any signature appears in the text.

    Args:
        haystack: Lower-cased text to search.
        needles: Lower-cased signatures to look for.

    Returns:
        True when at least one signature is present.
    """
    return any(needle in haystack for needle in needles)


def classify_launch_failure(
    error_text: str,
    user_data_dir: str | None = None,
) -> LaunchDiagnosis:
    """
    Classify a Chrome launch failure from its error text.

    Args:
        error_text: Combined exception message and any captured browser
            output. Empty or whitespace-only input yields UNKNOWN.
        user_data_dir: Optional user data directory involved in the
            launch, quoted back in the remedy to make it actionable.

    Returns:
        Structured diagnosis. Never raises; unrecognized input is
        classified as UNKNOWN rather than guessed at.
    """
    text = (error_text or "").lower()
    location = (
        f" (user data directory: {user_data_dir})" if user_data_dir else ""
    )

    if _matches(text, _HANDOFF_SIGNATURES):
        return LaunchDiagnosis(
            kind=LaunchFailureKind.SINGLETON_HANDOFF,
            summary=(
                "Chrome deferred to an already-running browser instance "
                "and exited instead of starting a controllable browser"
                f"{location}"
            ),
            remedy=(
                "Chromium allows one instance per user data directory. "
                "The launch reached an existing instance for this "
                "directory, so no automation browser was created. Verify "
                "the automation user data directory is genuinely separate "
                "from the directory the running Chrome is using, and that "
                "no policy or environment override is redirecting it. "
                "Using a non-branded Chromium build avoids contending "
                "with the everyday Chrome installation."
            ),
            retryable=False,
        )

    if _matches(text, _PROFILE_IN_USE_SIGNATURES):
        return LaunchDiagnosis(
            kind=LaunchFailureKind.PROFILE_IN_USE,
            summary=f"The Chrome profile is locked by another process{location}",
            remedy=(
                "Close the browser holding the profile, or point the "
                "automation profile at a directory no other process uses. "
                "If the owning process is gone, a stale lock may remain "
                "in the user data directory."
            ),
            retryable=True,
        )

    if _matches(text, _EXECUTABLE_SIGNATURES):
        return LaunchDiagnosis(
            kind=LaunchFailureKind.EXECUTABLE_NOT_FOUND,
            summary="The configured browser executable was not found",
            remedy=(
                "Set an explicit executable path, or install the browser "
                "binaries required by the configured channel."
            ),
            retryable=False,
        )

    if _matches(text, _DEPENDENCY_SIGNATURES):
        return LaunchDiagnosis(
            kind=LaunchFailureKind.MISSING_DEPENDENCIES,
            summary="The host is missing shared libraries the browser needs",
            remedy="Install the browser's system dependencies, then retry.",
            retryable=False,
        )

    if _matches(text, _TIMEOUT_SIGNATURES):
        return LaunchDiagnosis(
            kind=LaunchFailureKind.TIMEOUT,
            summary="The browser did not start within the launch timeout",
            remedy=(
                "Increase the launch timeout, or check whether the host is "
                "under heavy load or blocking the browser process."
            ),
            retryable=True,
        )

    return LaunchDiagnosis(
        kind=LaunchFailureKind.UNKNOWN,
        summary="The browser failed to launch for an unrecognized reason",
        remedy=(
            "Inspect the full browser output. The failure did not match "
            "any known signature, so no specific remedy is asserted."
        ),
        retryable=False,
    )


def detect_handoff(error_text: str) -> bool:
    """
    Report whether text indicates a process-singleton hand-off.

    Provided as a narrow predicate for callers that only need this one
    case and should not depend on the whole classification result.

    Args:
        error_text: Combined exception message and browser output.

    Returns:
        True when hand-off signatures are present.
    """
    return _matches((error_text or "").lower(), _HANDOFF_SIGNATURES)


__all__ = [
    "LaunchDiagnosis",
    "LaunchFailureKind",
    "classify_launch_failure",
    "detect_handoff",
]
