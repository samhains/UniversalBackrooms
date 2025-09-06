"""
Centralized model configuration and helpers.

This file defines the available model aliases and their provider-specific
API identifiers. Other modules should import from here instead of duplicating
choices or configuration.
"""

from __future__ import annotations

from typing import Dict, Any, List


# Single source of truth for model definitions
MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "sonnet": {
        "api_name": "claude-3-5-sonnet-20240620",
        "display_name": "Claude",
        "company": "anthropic",
    },
    "opus": {
        "api_name": "claude-3-opus-20240229",
        "display_name": "Claude",
        "company": "anthropic",
    },
    "opus4": {
        "api_name": "anthropic/claude-opus-4.1",
        "display_name": "Claude",
        "company": "openrouter",
    },
    "gpt4o": {
        "api_name": "gpt-4o-2024-08-06",
        "display_name": "GPT4o",
        "company": "openai",
    },
    # OpenRouter models
    "grok4": {
        "api_name": "x-ai/grok-4",
        "display_name": "grok-4",
        "company": "openrouter",
    },
    "gpt5": {
        "api_name": "openai/gpt-5-chat",
        "display_name": "GPT-5",
        "company": "openrouter",
    },
      "25pro": {
        "api_name": "google/gemini-2.5-pro",
        "display_name": "Gemini 2.5-pro",
        "company": "openrouter",
    },
      "v3": {
        "api_name": "deepseek/deepseek-chat-v3-0324",
        "display_name": "DeepSeek V3",
        "company": "openrouter",
    },
      "v31": {
        "api_name": "deepseek/deepseek-chat-v3.1",
        "display_name": "DeepSeek V3.1",
        "company": "openrouter",
    },
      "k2": {
        "api_name": "moonshotai/kimi-k2-0905",
        "display_name": "Kimi K2",
        "company": "openrouter",
    },
    "o1-preview": {"api_name": "o1-preview", "display_name": "O1", "company": "openai"},
    "o1-mini": {"api_name": "o1-mini", "display_name": "Mini", "company": "openai"},
    "hermes": {
        "api_name": "nousresearch/hermes-4-405b",
        "display_name": "Hermes 405B",
        "company": "openrouter",
        # Toggle Hermes 4 hybrid reasoning mode (OpenRouter)
        "reasoning_enabled": False,
    },
    "hermes_reasoning": {
        "api_name": "nousresearch/hermes-4-405b",
        "display_name": "Hermes 405B (Reasoning)",
        "company": "openrouter",
        "reasoning_enabled": True,
    },
}


def get_model_choices(include_cli: bool = True) -> List[str]:
    """Return valid CLI choices derived from MODEL_INFO.

    When include_cli is True, the special 'cli' pseudo-model is appended.
    """
    choices = sorted(MODEL_INFO.keys())
    if include_cli:
        choices.append("cli")
    return choices


def get_model_info() -> Dict[str, Dict[str, Any]]:
    """Expose the model information mapping."""
    return MODEL_INFO

