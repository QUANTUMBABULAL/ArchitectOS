"""
Reusable decorators for common patterns in the AI Research Operator.

This module provides decorators for cross-cutting concerns such as
logging, error handling, retries, caching, and performance tracking.

Features:
    - Automatic logging
    - Error handling and recovery
    - Exponential backoff retry logic
    - Simple caching
    - Performance tracking
    - Type validation

Usage:
    from src.utils.decorators import retry, timed, logged

    @logged
    @retry(max_attempts=3)
    def my_function():
        pass
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar

from src.constants import BACKOFF_MULTIPLIER, INITIAL_RETRY_DELAY, MAX_RETRIES
from src.exceptions import AIResearchOperatorError

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def logged(func: F) -> F:
    """
    Decorator that logs function entry and exit with arguments and results.

    Useful for debugging and understanding execution flow. Logs are produced
    at DEBUG level with function name, arguments, and return values.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function that logs execution.

    Example:
        >>> @logged
        ... def add(a, b):
        ...     return a + b
        >>> add(2, 3)
        5
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        logger.debug(
            f"Calling {func_name} with args={args}, kwargs={kwargs}"
        )

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func_name} returned {result!r}")
            return result
        except Exception as e:
            logger.error(f"{func_name} raised {type(e).__name__}: {e}")
            raise

    return wrapper  # type: ignore


def timed(func: F) -> F:
    """
    Decorator that measures and logs function execution time.

    Useful for performance monitoring. Execution time is logged at INFO level.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function that measures execution time.

    Example:
        >>> @timed
        ... def slow_function():
        ...     time.sleep(0.1)
        >>> slow_function()
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        func_name = func.__name__

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func_name} failed after {elapsed:.2f}s: {e}"
            )
            raise

    return wrapper  # type: ignore


def retry(
    max_attempts: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
    backoff_multiplier: float = BACKOFF_MULTIPLIER,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries function with exponential backoff.

    Retries the decorated function if it raises an exception from the
    specified exception tuple. Uses exponential backoff to space out
    retry attempts.

    Args:
        max_attempts: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        backoff_multiplier: Multiplier for exponential backoff.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Decorator function.

    Raises:
        The original exception if all retry attempts are exhausted.

    Example:
        >>> @retry(max_attempts=3)
        ... def flaky_operation():
        ...     # Might fail initially
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(
                        f"Attempt {attempt}/{max_attempts} for {func_name}"
                    )
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func_name} failed after {max_attempts} "
                            f"attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"{func_name} failed (attempt {attempt}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    delay *= backoff_multiplier

        return wrapper  # type: ignore

    return decorator


def cache(ttl: Optional[float] = None) -> Callable[[F], F]:
    """
    Decorator that caches function results with optional TTL.

    Caches the result of a function for reuse on subsequent calls with
    the same arguments. Optionally expires cached values after TTL seconds.

    Args:
        ttl: Time-to-live in seconds. None means cache forever.

    Returns:
        Decorator function.

    Note:
        Function arguments must be hashable.

    Example:
        >>> @cache(ttl=60)
        ... def expensive_computation(x):
        ...     return x ** 2
    """

    def decorator(func: F) -> F:
        cache_store: dict = {}
        cache_timestamps: dict = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))

            if key in cache_store:
                if ttl is None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache_store[key]

                elapsed = time.time() - cache_timestamps[key]
                if elapsed < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache_store[key]

                logger.debug(f"Cache expired for {func.__name__}")
                del cache_store[key]
                del cache_timestamps[key]

            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            cache_store[key] = result
            cache_timestamps[key] = time.time()
            return result

        return wrapper  # type: ignore

    return decorator


def validate_args(**expected_types: type) -> Callable[[F], F]:
    """
    Decorator that validates argument types at runtime.

    Checks that decorated function arguments match expected types.
    Raises ValidationError if type mismatch is detected.

    Args:
        **expected_types: Keyword arguments mapping parameter names to types.

    Returns:
        Decorator function.

    Raises:
        ValidationError: If argument type doesn't match expected type.

    Example:
        >>> @validate_args(name=str, age=int)
        ... def create_person(name, age):
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from src.exceptions import ValidationError

            for param_name, expected_type in expected_types.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not isinstance(value, expected_type):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be "
                            f"{expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def handle_errors(
    default_return: Any = None,
    log_traceback: bool = True,
) -> Callable[[F], F]:
    """
    Decorator that handles exceptions and returns default value on error.

    Catches all exceptions, logs them, and returns a default value instead
    of propagating the exception.

    Args:
        default_return: Value to return on exception.
        log_traceback: Whether to log full traceback.

    Returns:
        Decorator function.

    Example:
        >>> @handle_errors(default_return=None)
        ... def risky_operation():
        ...     raise ValueError("Something went wrong")
        >>> risky_operation() is None
        True
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_traceback:
                    logger.exception(
                        f"Error in {func.__name__}: {e}", exc_info=True
                    )
                else:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return

        return wrapper  # type: ignore

    return decorator


__all__ = [
    "logged",
    "timed",
    "retry",
    "cache",
    "validate_args",
    "handle_errors",
]
