"""
Custom type definitions and type aliases for the AI Research Operator.

This module provides type definitions that enhance code clarity and enable
better IDE support and type checking across the application.

Type Categories:
    - JSON Types: Type aliases for JSON-compatible data
    - Async Types: Type aliases for async operations
    - Callback Types: Type aliases for callback functions
    - Entity Types: Domain-specific type definitions

Usage:
    from src.utils.types import JSONValue, AsyncCallback

    def process_data(data: JSONValue) -> JSONValue:
        pass
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

# ============================================================================
# JSON TYPES
# ============================================================================

JSONValue = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]
"""
Type alias for JSON-compatible values.

Represents any value that can be serialized to JSON format.
Used for API requests, responses, and data storage.
"""

JSONDict = Dict[str, JSONValue]
"""
Type alias for JSON-compatible dictionaries.

A dictionary where all keys are strings and values are JSON-compatible.
"""

JSONList = List[JSONValue]
"""
Type alias for JSON-compatible lists.

A list containing only JSON-compatible values.
"""

# ============================================================================
# ASYNC TYPES
# ============================================================================

AsyncCallback = Callable[..., Awaitable[Any]]
"""
Type alias for async callback functions.

Used for async hooks and event handlers throughout the system.
"""

AsyncFunction = Callable[..., Awaitable[Any]]
"""
Type alias for async functions.

Generic async function type with any arguments and return type.
"""

# ============================================================================
# CALLBACK TYPES
# ============================================================================

Callback = Callable[..., Any]
"""
Type alias for synchronous callback functions.

Used for synchronous hooks and event handlers.
"""

ErrorCallback = Callable[[Exception], None]
"""
Type alias for error callback functions.

Called when an error occurs during operation.
"""

ProgressCallback = Callable[[float], None]
"""
Type alias for progress callback functions.

Called with progress value (0.0 to 1.0) during long-running operations.
"""

CompletionCallback = Callable[[Any], None]
"""
Type alias for completion callback functions.

Called with result when operation completes.
"""

# ============================================================================
# ENTITY TYPES
# ============================================================================

AgentId = str
"""
Type alias for agent identifiers.

Unique string identifier for AI agents in the system.
"""

TaskId = str
"""
Type alias for task identifiers.

Unique string identifier for tasks managed by the orchestrator.
"""

PlanId = str
"""
Type alias for plan identifiers.

Unique string identifier for execution plans created by the planner.
"""

MemoryKey = str
"""
Type alias for memory storage keys.

String key for accessing values in the memory system.
"""

# ============================================================================
# RESULT TYPES
# ============================================================================

Result = Union[Any, Exception]
"""
Type alias for operation results.

Can be either a successful result or an exception.
"""

OptionalResult = Optional[Any]
"""
Type alias for optional operation results.

Result that might not be available.
"""

# ============================================================================
# CONFIGURATION TYPES
# ============================================================================

ConfigDict = Dict[str, Any]
"""
Type alias for configuration dictionaries.

Used for passing configuration parameters throughout the system.
"""

# ============================================================================
# VALIDATION TYPES
# ============================================================================

ValidationRules = Dict[str, Any]
"""
Type alias for validation rule dictionaries.

Specifies constraints and rules for validating data.
"""

ValidationResult = Dict[str, Union[bool, List[str]]]
"""
Type alias for validation results.

Contains success status and list of validation errors if any.
"""

__all__ = [
    # JSON Types
    "JSONValue",
    "JSONDict",
    "JSONList",
    # Async Types
    "AsyncCallback",
    "AsyncFunction",
    # Callback Types
    "Callback",
    "ErrorCallback",
    "ProgressCallback",
    "CompletionCallback",
    # Entity Types
    "AgentId",
    "TaskId",
    "PlanId",
    "MemoryKey",
    # Result Types
    "Result",
    "OptionalResult",
    # Configuration Types
    "ConfigDict",
    # Validation Types
    "ValidationRules",
    "ValidationResult",
]
