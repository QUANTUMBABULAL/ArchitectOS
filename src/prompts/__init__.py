"""
Prompts package for prompt templates and output schemas.

Prompt text is centralized here so latency-sensitive wording and the
JSON schemas that constrain model output can be reviewed in one place,
independently of the components that consume them.
"""

from .brain_prompts import (
    CLASSIFIER_SCHEMA,
    CLASSIFIER_SYSTEM_PROMPT,
    DECOMPOSER_SCHEMA,
    DECOMPOSER_SYSTEM_PROMPT,
    LOCAL_ANSWER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    classifier_prompt,
    decomposer_prompt,
)

__all__ = [
    "CLASSIFIER_SCHEMA",
    "CLASSIFIER_SYSTEM_PROMPT",
    "DECOMPOSER_SCHEMA",
    "DECOMPOSER_SYSTEM_PROMPT",
    "LOCAL_ANSWER_SYSTEM_PROMPT",
    "SYNTHESIS_SYSTEM_PROMPT",
    "classifier_prompt",
    "decomposer_prompt",
]
