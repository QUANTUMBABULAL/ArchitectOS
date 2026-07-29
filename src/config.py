"""
Configuration management using Pydantic.

This module defines the application settings and configuration schema.
It handles environment variable loading, validation, and provides
type-safe access to all configuration values.

Features:
    - Environment variable loading via .env files
    - Type validation using Pydantic
    - Default value management
    - Runtime configuration overrides
    - Automatic documentation

Usage:
    from src.config import Settings

    settings = Settings()
    print(settings.api_host)
    print(settings.api_port)
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    """
    Application settings with environment variable support.

    Configuration is loaded from:
    1. Environment variables
    2. .env file in project root
    3. Default values defined in field definitions

    All settings are type-validated and documented.

    Attributes:
        Environment configuration for all system components.
    """

    # ========================================================================
    # ENVIRONMENT
    # ========================================================================

    environment: str = Field(
        default="development",
        description="Execution environment (development, staging, production)",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # ========================================================================
    # API SETTINGS
    # ========================================================================

    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )

    api_port: int = Field(
        default=8000,
        description="API server port",
    )

    api_workers: int = Field(
        default=4,
        description="Number of API worker processes",
    )

    api_timeout: float = Field(
        default=60.0,
        description="API request timeout in seconds",
    )

    # ========================================================================
    # BROWSER SETTINGS
    # ========================================================================

    browser_headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )

    browser_mode: str = Field(
        default="auto",
        description="Browser connection mode (auto, launch, attach)",
    )

    automation_profile: str = Field(
        default="ArchitectOS",
        description=(
            "Chrome profile folder used by ArchitectOS, stored under "
            "DATA_DIR/chrome-profile/. Never the user's normal profile"
        ),
    )

    browser_maximized: bool = Field(
        default=True,
        description=(
            "Open Chrome maximized with no fixed viewport, so the window "
            "behaves like an ordinary Chrome session"
        ),
    )

    remote_debug_port: int = Field(
        default=9222,
        description="Chrome remote debugging port for attach mode",
    )

    chrome_path: Optional[str] = Field(
        default=None,
        description="Optional path to the Google Chrome executable",
    )

    headless: Optional[bool] = Field(
        default=None,
        description="Optional browser headless override",
    )

    typing_delay: int = Field(
        default=40,
        description="Default keyboard typing delay in milliseconds",
    )

    attach_timeout: float = Field(
        default=5.0,
        description="Timeout for attaching to remote-debug Chrome",
    )

    browser_timeout: float = Field(
        default=30.0,
        description="Default browser operation timeout in seconds",
    )

    browser_viewport_width: int = Field(
        default=1920,
        description="Browser viewport width in pixels",
    )

    browser_viewport_height: int = Field(
        default=1080,
        description="Browser viewport height in pixels",
    )

    # ========================================================================
    # OLLAMA/BRAIN SETTINGS
    # ========================================================================

    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API server host",
    )

    ollama_model: str = Field(
        default="qwen3:4b",
        description="Default Ollama model to use",
    )

    ollama_timeout: float = Field(
        default=120.0,
        description="Ollama read timeout for long inference calls",
    )

    ollama_fast_timeout: float = Field(
        default=25.0,
        description=(
            "Ollama read timeout for bounded structured calls such as "
            "complexity classification, which must fail fast"
        ),
    )

    ollama_connect_timeout: float = Field(
        default=5.0,
        description="Ollama TCP connect timeout in seconds",
    )

    ollama_num_ctx: int = Field(
        default=4096,
        description="Context window size passed to the local model",
    )

    ollama_keep_alive: str = Field(
        default="10m",
        description=(
            "How long Ollama keeps the model resident after a request, "
            "avoiding repeated model load cost"
        ),
    )

    ollama_think: bool = Field(
        default=False,
        description=(
            "Allow thinking models to emit a reasoning preamble. Disabled "
            "by default because it multiplies latency for short answers"
        ),
    )

    ollama_classify_tokens: int = Field(
        default=64,
        description="Generation cap for complexity classification",
    )

    ollama_decompose_tokens: int = Field(
        default=400,
        description="Generation cap for goal decomposition",
    )

    classification_cache_size: int = Field(
        default=256,
        description="Number of complexity verdicts cached in memory",
    )

    # ========================================================================
    # LOGGING SETTINGS
    # ========================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (None for console only)",
    )

    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format string",
    )

    # ========================================================================
    # STORAGE SETTINGS
    # ========================================================================

    data_dir: str = Field(
        default="data",
        description="Directory for persistent data storage",
    )

    cache_dir: str = Field(
        default=".cache",
        description="Directory for temporary cache files",
    )

    state_persistence: bool = Field(
        default=True,
        description="Enable persistent state storage",
    )

    # ========================================================================
    # TIMEOUT SETTINGS
    # ========================================================================

    default_timeout: float = Field(
        default=30.0,
        description="Default timeout for operations in seconds",
    )

    browser_launch_timeout: float = Field(
        default=60.0,
        description="Browser launch timeout in seconds",
    )

    inference_timeout: float = Field(
        default=120.0,
        description="AI inference timeout in seconds",
    )

    # ========================================================================
    # RETRY SETTINGS
    # ========================================================================

    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
    )

    retry_delay: float = Field(
        default=1.0,
        description="Initial retry delay in seconds",
    )

    retry_backoff_multiplier: float = Field(
        default=2.0,
        description="Multiplier for exponential backoff",
    )

    # ========================================================================
    # CONSENSUS SETTINGS
    # ========================================================================

    consensus_enabled: bool = Field(
        default=True,
        description="Enable consensus mechanism",
    )

    consensus_min_agents: int = Field(
        default=2,
        description="Minimum agents for consensus",
    )

    consensus_threshold: float = Field(
        default=0.51,
        description="Agreement threshold for consensus (0.0 to 1.0)",
    )

    # ========================================================================
    # PROVIDER SETTINGS
    # ========================================================================

    enabled_providers: str = Field(
        default="chatgpt,gemini,grok,claude,deepseek",
        description=(
            "Comma-separated providers that participate in research, in "
            "tab-open order. Leave empty to enable every provider that "
            "ships enabled by default. Unknown names are ignored"
        ),
    )

    disabled_providers: str = Field(
        default="",
        description=(
            "Comma-separated providers that must never launch, be "
            "monitored, be recovered, or appear in consensus. Takes "
            "precedence over the enabled list"
        ),
    )

    research_synthesize: bool = Field(
        default=False,
        description=(
            "Run local-model synthesis over the gathered answers on the "
            "/research fast path. Off by default to keep the path free of "
            "inference latency"
        ),
    )

    provider_timeout: float = Field(
        default=180.0,
        description="Per-provider timeout for one prompt in seconds",
    )

    login_wait_seconds: float = Field(
        default=180.0,
        description=(
            "How long to wait for a manual sign-in before pausing the "
            "provider. Set to 0 to pause immediately without waiting. "
            "ArchitectOS never enters credentials; it only waits for the "
            "user to sign in inside the browser window"
        ),
    )

    login_poll_seconds: float = Field(
        default=3.0,
        description=(
            "Seconds between checks while waiting for a manual sign-in"
        ),
    )

    session_monitor_interval: float = Field(
        default=60.0,
        description=(
            "Seconds between provider health checks. Set to 0 to disable "
            "background monitoring and tab recovery"
        ),
    )

    conversation_context_chars: int = Field(
        default=120000,
        description=(
            "Approximate character budget per provider conversation before "
            "it is reset. Set to 0 to never reset automatically"
        ),
    )

    open_session_on_start: bool = Field(
        default=False,
        description=(
            "Launch the browser and open every provider tab during startup "
            "rather than on first research request"
        ),
    )

    auto_hide_browser: bool = Field(
        default=True,
        description=(
            "Hide the automation Chrome windows once every provider is "
            "signed in, keeping Mission Control the only visible surface. "
            "The windows are restored automatically whenever a provider "
            "needs a manual sign-in or verification, and on demand via "
            "Show Browser"
        ),
    )

    research_fresh_conversation: bool = Field(
        default=False,
        description=(
            "Start a brand-new provider conversation for every research "
            "run. Off by default: existing threads are continued, and new "
            "chats happen only on explicit request, context overflow, or "
            "when a provider forces one"
        ),
    )

    research_mode: str = Field(
        default="operator",
        description=(
            "How research requests are executed. 'operator' plans the "
            "request into subtasks and distributes them across providers "
            "(the research engine); 'debate' asks every provider the same "
            "question over multiple rounds (the previous behaviour)"
        ),
    )

    research_planning: bool = Field(
        default=True,
        description=(
            "Use the local model to decompose requests into subtasks. "
            "When off, a deterministic template plan is used instead"
        ),
    )

    final_answer_synthesis: bool = Field(
        default=True,
        description=(
            "Use the local Ollama model to write the single concise final "
            "answer. When off (or when the model fails), a deterministic "
            "summary is built from the consensus data instead"
        ),
    )

    # ========================================================================
    # DEBATE SETTINGS
    # ========================================================================

    debate_enabled: bool = Field(
        default=True,
        description="Use multi-round debate for research requests",
    )

    debate_max_rounds: int = Field(
        default=3,
        description="Maximum debate rounds before stopping",
    )

    debate_confidence_threshold: float = Field(
        default=0.8,
        description=(
            "Consensus confidence at which debate stops early (0.0 to 1.0)"
        ),
    )

    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================

    verbose_logging: bool = Field(
        default=False,
        description="Enable verbose logging",
    )

    performance_tracking: bool = Field(
        default=True,
        description="Enable performance metrics tracking",
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """
        Validate environment value.

        Args:
            v: Environment string to validate.

        Returns:
            Validated environment string.

        Raises:
            ValueError: If environment is not valid.
        """
        valid_environments = {"development", "staging", "production"}
        if v not in valid_environments:
            raise ValueError(
                f"Invalid environment '{v}'. "
                f"Must be one of: {valid_environments}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """
        Validate logging level.

        Args:
            v: Log level string to validate.

        Returns:
            Validated log level string.

        Raises:
            ValueError: If log level is not valid.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log level '{v}'. "
                f"Must be one of: {valid_levels}"
            )
        return v.upper()

    @field_validator("research_mode")
    @classmethod
    def validate_research_mode(cls, v: str) -> str:
        """
        Validate the research execution mode.

        Args:
            v: Raw mode string.

        Returns:
            Normalized mode.

        Raises:
            ValueError: If the mode is not recognized.
        """
        mode = v.lower().strip()
        valid_modes = {"operator", "debate"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid research mode '{v}'. Must be one of: {valid_modes}"
            )
        return mode

    @field_validator("browser_mode")
    @classmethod
    def validate_browser_mode(cls, v: str) -> str:
        """
        Validate browser connection mode.

        Args:
            v: Browser mode string to validate.

        Returns:
            Normalized browser mode.

        Raises:
            ValueError: If browser mode is not valid.
        """
        mode = v.lower().strip()
        valid_modes = {"auto", "launch", "attach"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid browser mode '{v}'. Must be one of: {valid_modes}"
            )
        return mode

    @field_validator("consensus_threshold", "debate_confidence_threshold")
    @classmethod
    def validate_consensus_threshold(cls, v: float) -> float:
        """
        Validate consensus threshold.

        Args:
            v: Threshold value to validate.

        Returns:
            Validated threshold value.

        Raises:
            ValueError: If threshold is not in valid range.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"Consensus threshold must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator(
        "api_workers",
        "consensus_min_agents",
        "max_retries",
        "remote_debug_port",
        "typing_delay",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """
        Validate positive integer values.

        Args:
            v: Integer value to validate.

        Returns:
            Validated integer value.

        Raises:
            ValueError: If value is not positive.
        """
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator(
        "api_timeout",
        "browser_timeout",
        "browser_launch_timeout",
        "default_timeout",
        "inference_timeout",
        "retry_delay",
        "attach_timeout",
        "ollama_timeout",
        "ollama_fast_timeout",
        "ollama_connect_timeout",
    )
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """
        Validate positive float values.

        Args:
            v: Float value to validate.

        Returns:
            Validated float value.

        Raises:
            ValueError: If value is not positive.
        """
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    def create_directories(self) -> None:
        """
        Create required directories if they don't exist.

        Creates data and cache directories specified in configuration.
        """
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __repr__(self) -> str:
        """Return string representation of settings."""
        return (
            f"Settings("
            f"environment={self.environment!r}, "
            f"api_host={self.api_host!r}, "
            f"api_port={self.api_port}, "
            f"debug={self.debug}"
            f")"
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_SETTINGS: Optional[Settings] = None


def _values_from_environment() -> dict[str, str]:
    """
    Collect field values from the process environment.

    Loads the project ``.env`` first (without overriding variables the
    user exported explicitly), then maps every UPPER_CASE variable whose
    name matches a Settings field. Pydantic performs the type coercion,
    so ``"true"`` becomes a bool and ``"9222"`` an int.

    Returns:
        Field values found in the environment.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # A missing dotenv package degrades to exported variables only.
        pass

    values: dict[str, str] = {}
    for name in Settings.model_fields:
        raw = os.environ.get(name.upper())
        if raw is not None and raw.strip() != "":
            values[name] = raw
    return values


def get_settings() -> Settings:
    """
    Get or create the global settings instance.

    The first call loads ``.env`` and environment variables; later calls
    return the same instance. Previously this returned a fresh
    default-valued Settings on every call and never read the
    environment at all, so nothing in ``.env`` had any effect.

    Returns:
        Settings instance with configuration loaded from environment.

    Example:
        >>> from src.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.api_port)
    """
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings(**_values_from_environment())
    return _SETTINGS


__all__ = [
    "Settings",
    "get_settings",
]
