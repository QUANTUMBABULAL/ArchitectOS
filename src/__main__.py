"""
AI Research Operator - Executable Entry Point.

Boots the complete system and runs the interactive terminal loop:

1. Loads and validates configuration from the environment
2. Initializes the logging system
3. Initializes persistent memory and connects to the local Ollama server
4. Verifies the BrowserManager (the browser itself launches lazily on
   the first request that requires external research)
5. Starts the ResearchOrchestrator runtime
6. Displays "ResearchOS Ready" and waits for user commands

Usage:
    python -m src
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from src.application import ResearchOperatorApp
from src.config import Settings, get_settings
from src.constants import LOG_FORMAT, LOG_LEVEL
from src.exceptions import AIResearchOperatorError, ConfigurationError
from src.logger import configure_logging, get_logger


def load_settings() -> Settings:
    """
    Load and validate application settings.

    Returns:
        Validated application settings.

    Raises:
        ConfigurationError: If configuration cannot be loaded or is
            invalid.
    """
    try:
        return get_settings()
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to load configuration: {exc}",
            code="CONFIG_LOAD_FAILED",
        ) from exc


def print_startup_info(settings: Settings) -> None:
    """
    Print system startup information to the console.

    Args:
        settings: Loaded application settings.
    """
    print("\n" + "=" * 70)
    print("AI RESEARCH OPERATOR - SYSTEM STARTUP")
    print("=" * 70)
    print(f"Environment:        {settings.environment}")
    print(f"Debug Mode:         {settings.debug}")
    print(f"Ollama Host:        {settings.ollama_host}")
    print(f"Ollama Model:       {settings.ollama_model}")
    print(f"Browser Mode:       {settings.browser_mode}")
    print(f"Log Level:          {settings.log_level}")
    print(f"Data Directory:     {Path(settings.data_dir).absolute()}")
    print(f"Cache Directory:    {Path(settings.cache_dir).absolute()}")
    print("=" * 70)


async def run_application(settings: Settings) -> int:
    """
    Initialize the application, run the terminal loop, and shut down.

    Args:
        settings: Loaded application settings.

    Returns:
        Process exit code: 0 on clean exit, 1 on initialization failure.
    """
    logger = get_logger(__name__)
    app = ResearchOperatorApp(settings=settings)

    try:
        await app.initialize()
    except AIResearchOperatorError as exc:
        logger.error("Initialization failed: %s", exc)
        print(f"\n[!] Startup failed: {exc}")
        await app.shutdown()
        return 1

    print_startup_info(settings)
    logger.info("AI Research Operator initialized successfully")

    try:
        await app.run_repl()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        await app.shutdown()

    print("Goodbye.")
    return 0


def main() -> int:
    """
    Main entry point for the AI Research Operator.

    Configures logging, loads settings, and runs the async application
    runtime.

    Returns:
        0 on clean exit, 1 on any failure.
    """
    # Bootstrap logging with constants so early failures are visible,
    # then re-derive the level from validated settings.
    configure_logging(
        level=getattr(logging, LOG_LEVEL),
        format_string=LOG_FORMAT,
    )
    logger = get_logger(__name__)

    try:
        settings = load_settings()
        level = getattr(logging, settings.log_level)
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    try:
        return asyncio.run(run_application(settings))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    except Exception as exc:
        logger.exception("Unexpected fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
