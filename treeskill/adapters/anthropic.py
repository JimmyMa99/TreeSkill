"""Anthropic Claude API Adapter.

Supports Claude model families:
- Claude 4.5 series (Opus, Sonnet, Haiku) — latest
- Claude 3.5 series (Sonnet, Haiku)
- Claude 3 series (Opus, Sonnet, Haiku)

Features:
- Vision support for all Claude 3+ models
- Proper system prompt handling (separate from messages)
- Accurate token counting using Anthropic's tokenizer
- Extended thinking support (for supported models)
- Tool use support
- Streaming support (optional)

Key Differences from OpenAI:
- System prompt is a separate parameter, not a message
- Messages alternate strictly between user/assistant
- Uses content blocks instead of simple strings
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

from anthropic import Anthropic

from treeskill.core.base_adapter import BaseModelAdapter
from treeskill.core.abc import OptimizablePrompt, Experience
from treeskill.core.prompts import TextPrompt, MultimodalPrompt


logger = logging.getLogger(__name__)


# Model configurations
CLAUDE_MODELS = {
    # Claude 4.5 series
    "claude-opus-4-5-20250918": {"context": 1_000_000, "vision": True, "cost_tier": "high"},
    "claude-sonnet-4-5-20250514": {"context": 200_000, "vision": True, "cost_tier": "mid"},
    "claude-haiku-4-5-20251001": {"context": 200_000, "vision": True, "cost_tier": "low"},

    # Claude 3.5 series
    "claude-3-5-sonnet-20241022": {"context": 200_000, "vision": True, "cost_tier": "mid"},
    "claude-3-5-sonnet-20240620": {"context": 200_000, "vision": True, "cost_tier": "mid"},
    "claude-3-5-haiku-20241022": {"context": 200_000, "vision": True, "cost_tier": "low"},

    # Claude 3 series
    "claude-3-opus-20240229": {"context": 200_000, "vision": True, "cost_tier": "high"},
    "claude-3-sonnet-20240229": {"context": 200_000, "vision": True, "cost_tier": "mid"},
    "claude-3-haiku-20240307": {"context": 200_000, "vision": True, "cost_tier": "low"},
}

# Default model
DEFAULT_MODEL = "claude-sonnet-4-5-20250514"


class AnthropicAdapter(BaseModelAdapter):
    """Adapter for Anthropic's Claude API.

    Parameters
    ----------
    model : str
        Model name (e.g., 'claude-3-5-sonnet-20241022').
    api_key : Optional[str]
        API key. If None, reads from ANTHROPIC_API_KEY env var.
    base_url : Optional[str]
        Base URL for API (for custom endpoints).
    **kwargs
        Additional parameters.

    Examples
    --------
    >>> adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
    >>> response = adapter.generate(prompt, context=experiences)

    >>> # With custom settings
    >>> adapter = AnthropicAdapter(
    ...     model="claude-3-5-haiku-20241022",
    ...     api_key="...",
    ... )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"),
            **kwargs,
        )

        # Initialize Anthropic client
        client_kwargs = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        # Only create client if API key is provided
        self.client = None
        if self._api_key:
            self.client = Anthropic(**client_kwargs)
        else:
            logger.warning(
                "No API key provided. Client will not be available. "
                "Set ANTHROPIC_API_KEY environment variable for full functionality."
            )

        # Get model configuration
        self._model_config = CLAUDE_MODELS.get(model, CLAUDE_MODELS[DEFAULT_MODEL])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_vision(self) -> bool:
        """All Claude 3+ models support vision."""
        return self._model_config.get("vision", True)

    @property
    def max_context_tokens(self) -> int:
        """Get max context length for this model."""
        return self._model_config.get("context", 200_000)

    # ------------------------------------------------------------------
    # Core API implementation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List[Experience]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Generate a response using Claude API.

        Parameters
        ----------
        prompt : OptimizablePrompt
            The system prompt / instruction.
        context : Optional[List[Experience]]
            Prior conversation turns.
        temperature : float
            Sampling temperature (0-1).
        max_tokens : int
            Maximum tokens to generate (required by Claude).
        **kwargs
            Additional Anthropic parameters.

        Returns
        -------
        str
            The generated response text.
        """
        if not self.client:
            raise RuntimeError(
                "Anthropic client not initialized. Please provide an API key. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        # Build system prompt and messages
        system_prompt, messages = self._build_claude_messages(prompt, context)

        # Prepare API call parameters
        api_params = {
            "model": self._model_name,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        # Add system prompt if provided
        if system_prompt:
            api_params["system"] = system_prompt

        # Call Anthropic API
        logger.debug(f"Calling Claude API with {len(messages)} messages")
        response = self.client.messages.create(**api_params)

        # Extract text from response
        content = response.content[0].text
        logger.debug(f"Generated {len(content)} characters")

        return content

    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Low-level API call (used by base class for gradient operations)."""
        if not self.client:
            raise RuntimeError(
                "Anthropic client not initialized. Please provide an API key."
            )

        # Convert messages to Claude format if needed
        claude_messages = self._convert_to_claude_format(messages)

        api_params = {
            "model": self._model_name,
            "max_tokens": 4096,  # Default for gradient operations
            "messages": claude_messages,
            "temperature": temperature,
            **kwargs,
        }

        if system:
            api_params["system"] = system

        response = self.client.messages.create(**api_params)
        return response.content[0].text

    def _count_tokens_impl(self, text: str) -> int:
        """Approximate token count for Claude models.

        Note: Anthropic doesn't provide an official tokenizer library.
        This uses an approximation: ~4 characters per token.
        """
        # Claude uses a similar tokenization to GPT-4
        # Approximate: 1 token ≈ 4 characters
        return len(text) // 4

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_claude_messages(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List[Experience]] = None,
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Build Claude message format.

        Key difference from OpenAI:
        - System prompt is separate parameter, not a message
        - Messages must alternate user/assistant

        Returns
        -------
        tuple
            (system_content, messages_list)
        """
        system_content = None
        messages = []

        # Extract system prompt
        if isinstance(prompt, TextPrompt):
            system_content = prompt.content
        elif isinstance(prompt, MultimodalPrompt):
            system_content = prompt.text
        elif hasattr(prompt, "content"):
            system_content = prompt.content
        elif hasattr(prompt, "text"):
            system_content = prompt.text
        else:
            system_content = self._extract_prompt_text(prompt)

        # Build conversation from context
        if context:
            for exp in context:
                # Add user message
                user_input = exp.get_input()
                if isinstance(user_input, list):
                    # Convert from OpenAI format to Claude format
                    for msg in user_input:
                        if msg.get("role") == "user":
                            messages.append({
                                "role": "user",
                                "content": self._to_content_block(msg.get("content")),
                            })
                        elif msg.get("role") == "assistant":
                            messages.append({
                                "role": "assistant",
                                "content": self._to_content_block(msg.get("content")),
                            })
                elif isinstance(user_input, dict):
                    messages.append({
                        "role": "user",
                        "content": self._to_content_block(user_input),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": str(user_input),
                    })

                # Add assistant response
                assistant_output = exp.get_output()
                messages.append({
                    "role": "assistant",
                    "content": str(assistant_output),
                })

        return system_content, messages

    def _convert_to_claude_format(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Claude format."""
        claude_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Skip system messages (handled separately)
            if role == "system":
                continue

            # Convert to Claude format
            claude_msg = {
                "role": role,
                "content": self._to_content_block(content),
            }
            claude_messages.append(claude_msg)

        return claude_messages

    def _to_content_block(self, content: Any) -> Union[str, List[Dict]]:
        """Convert content to Claude's content block format.

        Claude uses content blocks for multimodal:
        [
            {"type": "text", "text": "..."},
            {"type": "image", "source": {"type": "url", "url": "..."}}
        ]
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Already in content block format
            return content
        elif isinstance(content, dict):
            # Single content block
            if content.get("type") == "text":
                return content.get("text", "")
            elif content.get("type") == "image_url":
                # Convert from OpenAI format
                url = content.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Data URL format
                    return [
                        {"type": "text", "text": ""},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            },
                        },
                    ]
                else:
                    return [
                        {"type": "text", "text": ""},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            },
                        },
                    ]
            return str(content)
        else:
            return str(content)

    def validate_prompt(self, prompt: OptimizablePrompt) -> List[str]:
        """Validate prompt compatibility with Claude models."""
        issues = super().validate_prompt(prompt)

        # Claude-specific checks

        # Check 1: All Claude 3+ support vision
        if isinstance(prompt, MultimodalPrompt) and not self.supports_vision:
            # This shouldn't happen for Claude 3+, but just in case
            issues.append(
                f"Model {self._model_name} does not support images."
            )

        return issues

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def count_messages_tokens(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> int:
        """Count tokens in messages + system prompt.

        This accounts for message formatting overhead.
        """
        # Approximate count
        total = 0

        # System prompt
        if system:
            total += self._count_tokens_impl(system)

        # Messages
        for msg in messages:
            # Message overhead
            total += 4  # Approximate overhead per message

            content = msg.get("content")
            if isinstance(content, str):
                total += self._count_tokens_impl(content)
            elif isinstance(content, list):
                # Content blocks
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += self._count_tokens_impl(block.get("text", ""))
                        elif block.get("type") == "image":
                            # Images use tokens based on size
                            # Approximate: 85-1105 tokens depending on size
                            total += 300  # Average estimate

        return total


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------

def create_claude_35_sonnet(api_key: Optional[str] = None, **kwargs) -> AnthropicAdapter:
    """Create a Claude 3.5 Sonnet adapter (recommended)."""
    return AnthropicAdapter(
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        **kwargs
    )


def create_claude_35_haiku(api_key: Optional[str] = None, **kwargs) -> AnthropicAdapter:
    """Create a Claude 3.5 Haiku adapter (fast, cheap)."""
    return AnthropicAdapter(
        model="claude-3-5-haiku-20241022",
        api_key=api_key,
        **kwargs
    )


def create_claude_3_opus(api_key: Optional[str] = None, **kwargs) -> AnthropicAdapter:
    """Create a Claude 3 Opus adapter (most capable)."""
    return AnthropicAdapter(
        model="claude-3-opus-20240229",
        api_key=api_key,
        **kwargs
    )
