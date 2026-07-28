"""
Application-wide constants for the AI Research Operator.

This module defines all constant values used throughout the application,
including timeouts, retry policies, path configurations, and feature flags.

Constants are organized by category for easy discovery and modification.
All values are type-hinted and documented.

Categories:
    - Timeouts: Request and operation timeouts
    - Retry Policies: Retry limits and backoff strategies
    - Paths: Directory and file locations
    - Browser: Playwright configuration
    - API: REST API defaults
    - Logging: Logging configuration
    - Memory: State and cache management
"""

from typing import Final

# ============================================================================
# TIMEOUTS (in seconds)
# ============================================================================

DEFAULT_TIMEOUT: Final[float] = 30.0
"""Default timeout for most operations."""

BROWSER_LAUNCH_TIMEOUT: Final[float] = 60.0
"""Timeout for launching browser instances."""

BROWSER_NAVIGATION_TIMEOUT: Final[float] = 30.0
"""Timeout for browser page navigation."""

INFERENCE_TIMEOUT: Final[float] = 120.0
"""Timeout for AI inference operations."""

CONSENSUS_TIMEOUT: Final[float] = 60.0
"""Timeout for consensus mechanism operations."""

ORCHESTRATOR_SHUTDOWN_TIMEOUT: Final[float] = 30.0
"""Timeout for graceful orchestrator shutdown."""

# ============================================================================
# RETRY POLICIES
# ============================================================================

MAX_RETRIES: Final[int] = 3
"""Maximum number of retry attempts for failed operations."""

INITIAL_RETRY_DELAY: Final[float] = 1.0
"""Initial delay in seconds for exponential backoff."""

MAX_RETRY_DELAY: Final[float] = 30.0
"""Maximum delay in seconds between retry attempts."""

BACKOFF_MULTIPLIER: Final[float] = 2.0
"""Multiplier for exponential backoff calculation."""

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT: Final[str] = "."
"""Root directory of the project."""

SRC_DIR: Final[str] = "src"
"""Source code directory."""

LOGS_DIR: Final[str] = "logs"
"""Logs directory."""

DATA_DIR: Final[str] = "data"
"""Data directory for persistent storage."""

CACHE_DIR: Final[str] = ".cache"
"""Cache directory for temporary files."""

# ============================================================================
# BROWSER CONFIGURATION
# ============================================================================

BROWSER_HEADLESS: Final[bool] = True
"""Run browser in headless mode."""

BROWSER_TIMEOUT: Final[float] = 30.0
"""Default timeout for browser operations."""

BROWSER_WAIT_FOR_SELECTOR_TIMEOUT: Final[float] = 10.0
"""Timeout for waiting for element selectors."""

BROWSER_VIEWPORT_WIDTH: Final[int] = 1920
"""Default browser viewport width."""

BROWSER_VIEWPORT_HEIGHT: Final[int] = 1080
"""Default browser viewport height."""

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_HOST: Final[str] = "0.0.0.0"
"""API server host."""

API_PORT: Final[int] = 8000
"""API server port."""

API_WORKERS: Final[int] = 4
"""Number of API worker processes."""

API_TIMEOUT: Final[float] = 60.0
"""API request timeout."""

API_MAX_CONTENT_LENGTH: Final[int] = 10_485_760  # 10 MB
"""Maximum request content length."""

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL: Final[str] = "INFO"
"""Default logging level."""

LOG_FORMAT: Final[str] = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
"""Default log message format."""

LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Date format for log timestamps."""

MAX_LOG_FILE_SIZE: Final[int] = 10_485_760  # 10 MB
"""Maximum size of a single log file before rotation."""

BACKUP_LOG_COUNT: Final[int] = 5
"""Number of backup log files to retain."""

# ============================================================================
# MEMORY AND STATE MANAGEMENT
# ============================================================================

MAX_MEMORY_ENTRIES: Final[int] = 10000
"""Maximum number of entries in memory store."""

MEMORY_CLEANUP_INTERVAL: Final[float] = 300.0
"""Interval in seconds for memory cleanup operations."""

CACHE_TTL: Final[float] = 3600.0
"""Default cache time-to-live in seconds."""

STATE_PERSISTENCE_ENABLED: Final[bool] = True
"""Enable persistent state storage."""

# ============================================================================
# FEATURE FLAGS
# ============================================================================

DEBUG_MODE: Final[bool] = False
"""Enable debug mode for development."""

VERBOSE_LOGGING: Final[bool] = False
"""Enable verbose logging output."""

PERFORMANCE_TRACKING: Final[bool] = True
"""Enable performance metrics tracking."""

# ============================================================================
# ERROR HANDLING
# ============================================================================

SUPPRESS_BROWSER_ERRORS: Final[bool] = False
"""Suppress browser-specific errors in production."""

DETAILED_ERROR_MESSAGES: Final[bool] = True
"""Include detailed error messages in responses."""

# ============================================================================
# CONSENSUS PARAMETERS
# ============================================================================

CONSENSUS_MIN_AGENTS: Final[int] = 2
"""Minimum number of agents required for consensus."""

CONSENSUS_AGREEMENT_THRESHOLD: Final[float] = 0.51
"""Minimum agreement ratio for consensus (0.0 to 1.0)."""

CONSENSUS_MAX_ITERATIONS: Final[int] = 10
"""Maximum iterations for consensus mechanism."""

# ============================================================================
# PLANNER PARAMETERS
# ============================================================================

PLANNER_MAX_DEPTH: Final[int] = 10
"""Maximum planning depth for task decomposition."""

PLANNER_MAX_BRANCHES: Final[int] = 5
"""Maximum branches to explore per planning step."""

PLANNER_OPTIMIZATION_ENABLED: Final[bool] = True
"""Enable plan optimization."""

__all__ = [
    # Timeouts
    "DEFAULT_TIMEOUT",
    "BROWSER_LAUNCH_TIMEOUT",
    "BROWSER_NAVIGATION_TIMEOUT",
    "INFERENCE_TIMEOUT",
    "CONSENSUS_TIMEOUT",
    "ORCHESTRATOR_SHUTDOWN_TIMEOUT",
    # Retry Policies
    "MAX_RETRIES",
    "INITIAL_RETRY_DELAY",
    "MAX_RETRY_DELAY",
    "BACKOFF_MULTIPLIER",
    # Paths
    "PROJECT_ROOT",
    "SRC_DIR",
    "LOGS_DIR",
    "DATA_DIR",
    "CACHE_DIR",
    # Browser
    "BROWSER_HEADLESS",
    "BROWSER_TIMEOUT",
    "BROWSER_WAIT_FOR_SELECTOR_TIMEOUT",
    "BROWSER_VIEWPORT_WIDTH",
    "BROWSER_VIEWPORT_HEIGHT",
    # API
    "API_HOST",
    "API_PORT",
    "API_WORKERS",
    "API_TIMEOUT",
    "API_MAX_CONTENT_LENGTH",
    # Logging
    "LOG_LEVEL",
    "LOG_FORMAT",
    "LOG_DATE_FORMAT",
    "MAX_LOG_FILE_SIZE",
    "BACKUP_LOG_COUNT",
    # Memory
    "MAX_MEMORY_ENTRIES",
    "MEMORY_CLEANUP_INTERVAL",
    "CACHE_TTL",
    "STATE_PERSISTENCE_ENABLED",
    # Feature Flags
    "DEBUG_MODE",
    "VERBOSE_LOGGING",
    "PERFORMANCE_TRACKING",
    # Error Handling
    "SUPPRESS_BROWSER_ERRORS",
    "DETAILED_ERROR_MESSAGES",
    # Consensus
    "CONSENSUS_MIN_AGENTS",
    "CONSENSUS_AGREEMENT_THRESHOLD",
    "CONSENSUS_MAX_ITERATIONS",
    # Planner
    "PLANNER_MAX_DEPTH",
    "PLANNER_MAX_BRANCHES",
    "PLANNER_OPTIMIZATION_ENABLED",
]
