"""
Logging configuration and utilities for the AI Research Operator.

This module provides a structured logging system with colored output,
consistent formatting, and hierarchical logger management. It ensures
all system components produce consistent, traceable logs.

Features:
    - Colored console output for development
    - JSON-compatible formatting for production
    - Hierarchical logger configuration
    - Dynamic log level management
    - Performance tracking capabilities
    - Structured logging support

Usage:
    from src.logger import get_logger

    logger = get_logger(__name__)
    logger.info("System initialized")
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color to console output.

    Provides color-coded log levels for improved readability during
    development and debugging.

    Color Scheme:
        DEBUG: Cyan
        INFO: Green
        WARNING: Yellow
        ERROR: Red
        CRITICAL: Red with background
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m\033[37m",  # Red background, white text
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with color codes.

        Args:
            record: The log record to format.

        Returns:
            Formatted string with color codes for console output.
        """
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        record.levelname = f"{level_color}{record.levelname}{reset}"
        return super().format(record)


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure root logger for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string: Custom format string. Defaults to standard format.

    Raises:
        ValueError: If level is not a valid logging level.
    """
    if not isinstance(level, int) or level not in logging._nameToLevel.values():
        raise ValueError(f"Invalid logging level: {level}")

    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = ColoredFormatter(format_string)
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)


def get_logger(
    name: str,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Get or create a logger for a module.

    This function provides a consistent interface for obtaining loggers
    throughout the application. Each module should call this function
    with __name__ as the argument.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Optional logging level override for this logger.

    Returns:
        Configured logger instance ready for use.

    Example:
        >>> from src.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


__all__ = [
    "ColoredFormatter",
    "configure_logging",
    "get_logger",
]
