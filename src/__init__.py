"""
AI Research Operator - Main Package.

A production-quality orchestration system for managing multiple AI systems
through browser automation. This package provides the core framework for
coordinating reasoning engines, browser interactions, and worker processes.

Modules:
    api: REST API interfaces for system interaction
    brain: Reasoning and decision-making logic
    browser: Playwright-based browser automation wrapper
    config: Configuration management and settings
    consensus: Multi-agent consensus mechanisms
    constants: Application-wide constants
    exceptions: Custom exception classes
    logger: Structured logging system
    memory: State and knowledge storage
    orchestrator: Main coordination and workflow management
    planner: Task planning and execution strategies
    prompts: Prompt templates and management
    utils: Utility functions and helpers
    workers: Background task processing

Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "AI Research Team"

from src.config import Settings
from src.logger import get_logger

__all__ = [
    "Settings",
    "get_logger",
]
