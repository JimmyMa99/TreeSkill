"""Model Adapters for different LLM providers.

This package provides concrete implementations of ModelAdapter for:
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 4.5 Sonnet, etc.)

Local models (llama.cpp, vLLM, Ollama) expose OpenAI-compatible APIs,
so use OpenAIAdapter with a custom ``base_url``.
"""

from evoskill.adapters.openai import OpenAIAdapter
from evoskill.adapters.anthropic import AnthropicAdapter

__all__ = [
    "OpenAIAdapter",
    "AnthropicAdapter",
]
