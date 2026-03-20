"""Model Adapters for different LLM providers.

This package provides concrete implementations of ModelAdapter for:
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 3.5 Sonnet, etc.)
"""

from evoskill.adapters.openai import OpenAIAdapter
from evoskill.adapters.anthropic import AnthropicAdapter

__all__ = [
    "OpenAIAdapter",
    "AnthropicAdapter",
]
