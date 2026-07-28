"""
Routing package: deterministic request dispatch ahead of inference.

FastRouter classifies incoming requests with cheap ordered rules so the
local model is consulted only for genuinely ambiguous input. The package
holds no I/O and no async code, keeping routing cheap enough to run on
every request.
"""

from .calculator import evaluate_expression, format_result
from .fast_router import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    FastRouter,
    RouteTarget,
    RoutingDecision,
    Rule,
    default_rules,
)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "FastRouter",
    "RouteTarget",
    "RoutingDecision",
    "Rule",
    "default_rules",
    "evaluate_expression",
    "format_result",
]
