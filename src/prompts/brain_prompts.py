"""
Prompts and output schemas for local-model reasoning tasks.

Prompts here are deliberately terse. The local model is used for routing
decisions whose answers are a handful of tokens, so every additional
instruction costs latency twice: once evaluating the prompt and again
because a chattier instruction invites a longer answer.

Output shapes are expressed as JSON schemas rather than as prose
instructions. Ollama's ``format`` parameter constrains generation to a
supplied schema, which removes the need to ask the model politely for
JSON and removes the need to salvage JSON out of prose afterwards.

Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

from typing import Any, Final

# ---------------------------------------------------------------------------
# Complexity classification
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT: Final[str] = (
    "Classify whether a task needs external web research. "
    "simple = greetings, arithmetic, definitions, formatting, or "
    "summarizing text already provided. "
    "complex = current events, comparisons, multi-step analysis, "
    "or anything needing up-to-date or specialised knowledge. "
    "Answer with JSON only."
)

CLASSIFIER_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "complexity": {"type": "string", "enum": ["simple", "complex"]},
        "confidence": {"type": "number"},
    },
    "required": ["complexity", "confidence"],
}


def classifier_prompt(task: str) -> str:
    """
    Build the classification prompt for a task.

    Args:
        task: Cleaned task description.

    Returns:
        Prompt text.
    """
    return f"Task: {task}"


# ---------------------------------------------------------------------------
# Goal decomposition
# ---------------------------------------------------------------------------

DECOMPOSER_SYSTEM_PROMPT: Final[str] = (
    "Split a research goal into independent, self-contained "
    "sub-questions, each answerable on its own without seeing the "
    "others. Be concise. Answer with JSON only."
)

DECOMPOSER_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["sub_questions"],
}


def decomposer_prompt(goal: str, max_sub_questions: int) -> str:
    """
    Build the decomposition prompt for a goal.

    Args:
        goal: Cleaned research goal.
        max_sub_questions: Upper bound on sub-questions requested.

    Returns:
        Prompt text.
    """
    return (
        f"Goal: {goal}\n"
        f"Produce at most {max_sub_questions} sub-questions."
    )


# ---------------------------------------------------------------------------
# Local answering
# ---------------------------------------------------------------------------

LOCAL_ANSWER_SYSTEM_PROMPT: Final[str] = (
    "You are a concise local assistant inside a research operating "
    "system. Answer the task directly and briefly. If the task actually "
    "needs deep, current, or specialised knowledge you do not have, say "
    "so explicitly instead of guessing."
)

SYNTHESIS_SYSTEM_PROMPT: Final[str] = (
    "You are the synthesis component of a research operating system. "
    "You are given a research goal, sub-questions, and answers gathered "
    "from external AI systems, possibly with consensus notes about "
    "agreement and contradictions. Write a clear, well-structured final "
    "report in Markdown that answers the goal. Attribute claims to their "
    "sources when they disagree, and state open questions explicitly. "
    "Do not invent information that is not in the gathered material."
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
