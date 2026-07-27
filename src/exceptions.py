"""
Custom exception classes for the AI Research Operator.

This module defines the exception hierarchy for the application, enabling
precise error handling and meaningful error messages throughout the system.

Exception Hierarchy:
    AIResearchOperatorError (Base)
        - ConfigurationError: Configuration-related failures
        - ValidationError: Input validation failures
        - BrowserError: Browser automation failures
            - BrowserLaunchError: Browser launch failures
            - BrowserAttachError: Browser attach failures
            - ProfileLockedError: Browser profile lock failures
            - RemoteDebugUnavailableError: Remote debugging unavailable
            - ChromeNotFoundError: Google Chrome discovery failures
        - OrchestratorError: Orchestration failures
        - MemoryError: Memory/state management failures
        - PlannerError: Planning failures
        - BrainError: Reasoning engine failures
        - ConsensusError: Consensus mechanism failures
        - WorkerError: Worker process failures

Type hints and docstrings are provided for all exceptions.
"""

from typing import Optional


class AIResearchOperatorError(Exception):
    """
    Base exception for the AI Research Operator.

    All custom exceptions in this module inherit from this base exception,
    enabling comprehensive error handling for the entire application.

    Attributes:
        message: Detailed error message explaining the failure.
        code: Optional error code for categorization and logging.
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
    ) -> None:
        """
        Initialize the base exception.

        Args:
            message: Detailed description of the error.
            code: Optional error code for categorization.
        """
        self.message = message
        self.code = code or self.__class__.__name__
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error message with code."""
        return f"[{self.code}] {self.message}"


class ConfigurationError(AIResearchOperatorError):
    """
    Raised when configuration is invalid or incomplete.

    Triggered during settings initialization, environment variable
    validation, or configuration file parsing failures.
    """

    pass


class ValidationError(AIResearchOperatorError):
    """
    Raised when input validation fails.

    Triggered when function arguments, API payloads, or data structures
    do not conform to expected schemas or constraints.
    """

    pass


class BrowserError(AIResearchOperatorError):
    """
    Raised when browser automation operations fail.

    Triggered by Playwright failures, navigation errors, element
    selection failures, or browser state management issues.
    """

    pass


class BrowserLaunchError(BrowserError):
    """
    Raised when launching a browser fails.

    Triggered by Chrome startup failures, invalid launch configuration,
    launch timeouts, or Playwright launch errors.
    """

    pass


class BrowserAttachError(BrowserError):
    """
    Raised when attaching to an existing browser fails.

    Triggered by CDP connection failures, missing contexts, incompatible
    remote debugging endpoints, or attach timeouts.
    """

    pass


class ProfileLockedError(BrowserLaunchError):
    """
    Raised when a Chrome profile is already locked by another process.

    Triggered before launch when lock indicators are found in a selected
    profile directory.
    """

    pass


class RemoteDebugUnavailableError(BrowserAttachError):
    """
    Raised when Chrome remote debugging is unavailable.

    Triggered when the configured remote debugging endpoint is not reachable
    or does not expose the expected Chrome DevTools Protocol metadata.
    """

    pass


class ChromeNotFoundError(BrowserLaunchError):
    """
    Raised when Google Chrome cannot be located.

    Triggered when automatic discovery and explicit configuration do not
    resolve to an installed Chrome executable.
    """

    pass


class OrchestratorError(AIResearchOperatorError):
    """
    Raised when orchestration or workflow execution fails.

    Triggered by task coordination failures, workflow state issues,
    or orchestration logic errors.
    """

    pass


class MemoryError(AIResearchOperatorError):
    """
    Raised when memory/state management operations fail.

    Triggered by storage failures, state inconsistencies, or
    memory access violations.
    """

    pass


class PlannerError(AIResearchOperatorError):
    """
    Raised when task planning operations fail.

    Triggered by plan generation failures, strategy selection errors,
    or planning constraint violations.
    """

    pass


class BrainError(AIResearchOperatorError):
    """
    Raised when reasoning engine operations fail.

    Triggered by inference failures, reasoning timeout, or
    decision-making logic errors.
    """

    pass


class ConsensusError(AIResearchOperatorError):
    """
    Raised when consensus mechanism operations fail.

    Triggered by agreement failures, voting mechanism errors,
    or consensus timeout.
    """

    pass


class ProviderAuthError(AIResearchOperatorError):
    """
    Raised when a provider requires an interactive sign-in.

    Deliberately distinct from BrowserError and WorkerError: an
    unauthenticated provider is not a broken one. Reloading the tab,
    restarting the worker, or relaunching the browser cannot fix it, and
    doing so destroys the very page the user needs in order to sign in.
    The correct response is to pause that provider and wait for a human.
    """

    pass


class ProviderChallengeError(AIResearchOperatorError):
    """
    Raised when a provider presents a human verification challenge.

    Triggered by CAPTCHA, bot-detection interstitials, or a re-login
    prompt. Distinct from a generic worker failure because the remedy is
    manual and provider-specific: that provider is paused while research
    continues with the others. Never retried automatically, since
    retrying a CAPTCHA cannot succeed and may harden the challenge.
    """

    pass


class WorkerError(AIResearchOperatorError):
    """
    Raised when background worker operations fail.

    Triggered by task processing failures, worker timeout,
    or worker state management errors.
    """

    pass


__all__ = [
    "AIResearchOperatorError",
    "ConfigurationError",
    "ValidationError",
    "BrowserError",
    "BrowserLaunchError",
    "BrowserAttachError",
    "ProfileLockedError",
    "RemoteDebugUnavailableError",
    "ChromeNotFoundError",
    "OrchestratorError",
    "MemoryError",
    "PlannerError",
    "BrainError",
    "ConsensusError",
    "WorkerError",
]
