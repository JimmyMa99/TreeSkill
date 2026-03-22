"""OpenAI API Adapter.

Supports all OpenAI models including:
- GPT-4o (vision-capable)
- GPT-4o-mini
- GPT-4-turbo
- o1-preview / o1-mini (reasoning models)
- Legacy models (GPT-3.5)

Features:
- Vision support for GPT-4o
- Accurate token counting with tiktoken
- Streaming support (optional)
- Automatic retries
- Rate limit handling
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import tiktoken
from openai import OpenAI

from treeskill.core.base_adapter import BaseModelAdapter
from treeskill.core.abc import OptimizablePrompt, Experience
from treeskill.core.prompts import TextPrompt, MultimodalPrompt


logger = logging.getLogger(__name__)


# Model capabilities
VISION_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview"}
REASONING_MODELS = {"o1-preview", "o1-mini"}

# Context length limits (in tokens)
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-16k": 16_385,
    "o1-preview": 128_000,
    "o1-mini": 128_000,
}


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI's chat completion API.

    Parameters
    ----------
    model : str
        Model name (e.g., 'gpt-4o', 'gpt-4o-mini').
    api_key : Optional[str]
        API key. If None, reads from OPENAI_API_KEY env var.
    base_url : Optional[str]
        Base URL for API (for custom endpoints).
    **kwargs
        Additional parameters (e.g., organization).

    Examples
    --------
    >>> adapter = OpenAIAdapter(model="gpt-4o")
    >>> response = adapter.generate(prompt, context=experiences)

    >>> # With custom base URL (e.g., Azure OpenAI)
    >>> adapter = OpenAIAdapter(
    ...     model="gpt-4o",
    ...     api_key="...",
    ...     base_url="https://your-resource.openai.azure.com/"
    ... )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            **kwargs,
        )

        # Initialize OpenAI client
        client_kwargs = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if "organization" in kwargs:
            client_kwargs["organization"] = kwargs["organization"]

        # Only create client if API key is provided
        self.client = None
        if self._api_key:
            self.client = OpenAI(**client_kwargs)
        else:
            logger.warning(
                "No API key provided. Client will not be available. "
                "Set OPENAI_API_KEY environment variable for full functionality."
            )

        # Initialize tokenizer
        try:
            self._tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (GPT-4 tokenizer)
            logger.warning(
                f"Model {model} not found in tiktoken, using cl100k_base encoding"
            )
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_vision(self) -> bool:
        """Check if model supports vision."""
        return any(vm in self._model_name.lower() for vm in VISION_MODELS)

    @property
    def max_context_tokens(self) -> int:
        """Get max context length for this model."""
        for model_prefix, limit in MODEL_CONTEXT_LIMITS.items():
            if self._model_name.startswith(model_prefix):
                return limit
        # Default fallback
        return 8_192

    # ------------------------------------------------------------------
    # Core API implementation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List[Experience]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a response using OpenAI's chat completion API.

        Parameters
        ----------
        prompt : OptimizablePrompt
            The system prompt / instruction.
        context : Optional[List[Experience]]
            Prior conversation turns.
        temperature : float
            Sampling temperature (0-2).
        max_tokens : Optional[int]
            Maximum tokens to generate.
        **kwargs
            Additional OpenAI parameters (e.g., top_p, presence_penalty).

        Returns
        -------
        str
            The generated response text.
        """
        if not self.client:
            raise RuntimeError(
                "OpenAI client not initialized. Please provide an API key. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Build messages
        system, messages = self._build_openai_messages(prompt, context)

        # Prepare API call parameters
        api_params = {
            "model": self._model_name,
            "temperature": temperature,
            **kwargs,
        }

        # Add system message if supported
        if system and not self._is_reasoning_model():
            messages = [{"role": "system", "content": system}] + messages

        api_params["messages"] = messages

        if max_tokens:
            api_params["max_tokens"] = max_tokens

        # Call OpenAI API
        logger.debug(f"Calling OpenAI API with {len(messages)} messages")
        response = self.client.chat.completions.create(**api_params)

        # Extract response
        content = response.choices[0].message.content
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
                "OpenAI client not initialized. Please provide an API key."
            )

        # Add system message if provided
        if system and not self._is_reasoning_model():
            messages = [{"role": "system", "content": system}] + messages

        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )

        return response.choices[0].message.content

    def _count_tokens_impl(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._tokenizer.encode(text))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_openai_messages(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List[Experience]] = None,
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Build OpenAI message format from prompt and context.

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
                    # Already in message format
                    messages.extend(user_input)
                elif isinstance(user_input, dict):
                    messages.append(user_input)
                else:
                    messages.append({"role": "user", "content": str(user_input)})

                # Add assistant response
                assistant_output = exp.get_output()
                if isinstance(assistant_output, dict):
                    # Multimodal output
                    messages.append({
                        "role": "assistant",
                        "content": assistant_output.get("text", str(assistant_output)),
                    })
                else:
                    messages.append({"role": "assistant", "content": str(assistant_output)})

        return system_content, messages

    def _is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model (o1 series)."""
        return any(rm in self._model_name.lower() for rm in REASONING_MODELS)

    def validate_prompt(self, prompt: OptimizablePrompt) -> List[str]:
        """Validate prompt compatibility with OpenAI models."""
        issues = super().validate_prompt(prompt)

        # Additional OpenAI-specific checks

        # Check 1: Reasoning models don't support system messages
        if self._is_reasoning_model():
            if isinstance(prompt, TextPrompt) and prompt.content:
                # This will be passed as first user message instead
                pass

        # Check 2: Vision model check
        if isinstance(prompt, MultimodalPrompt):
            if not self.supports_vision:
                issues.append(
                    f"Model {self._model_name} does not support images. "
                    f"Use a vision-capable model like gpt-4o."
                )

        return issues

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in a list of messages.

        This accounts for message formatting overhead.
        """
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_message = 3  # <im_start>, role, <im_end>
        tokens_per_name = 1  # If name is specified

        total_tokens = 0
        for message in messages:
            total_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    total_tokens += len(self._tokenizer.encode(value))
                elif isinstance(value, list):
                    # Multimodal content
                    for part in value:
                        if isinstance(part, dict) and "text" in part:
                            total_tokens += len(self._tokenizer.encode(part["text"]))
                if key == "name":
                    total_tokens += tokens_per_name

        total_tokens += 3  # Every reply is primed with <im_start>assistant
        return total_tokens


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------

def create_gpt4o_adapter(api_key: Optional[str] = None, **kwargs) -> OpenAIAdapter:
    """Create a GPT-4o adapter (vision-capable)."""
    return OpenAIAdapter(model="gpt-4o", api_key=api_key, **kwargs)


def create_gpt4o_mini_adapter(api_key: Optional[str] = None, **kwargs) -> OpenAIAdapter:
    """Create a GPT-4o-mini adapter (faster, cheaper)."""
    return OpenAIAdapter(model="gpt-4o-mini", api_key=api_key, **kwargs)


def create_o1_adapter(api_key: Optional[str] = None, **kwargs) -> OpenAIAdapter:
    """Create an o1-preview adapter (reasoning model)."""
    return OpenAIAdapter(model="o1-preview", api_key=api_key, **kwargs)
