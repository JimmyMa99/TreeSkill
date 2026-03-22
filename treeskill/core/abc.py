"""Abstract base classes for the optimization framework.

These ABCs define the interfaces that make the framework:
- Model-agnostic (supports OpenAI, Anthropic, local models, etc.)
- Multimodal-ready (text, images, audio, video)
- Train-free (no gradient computation, only API calls)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union
from pathlib import Path


class OptimizablePrompt(ABC):
    """Base class for all optimizable prompts.

    A prompt can be text-based, multimodal, or structured (e.g., with JSON schema).
    The key is that it can be iteratively improved via textual gradients.
    """

    @abstractmethod
    def to_model_input(self) -> Any:
        """Convert to the format expected by the model API.

        Returns
        -------
        Any
            The format depends on the model:
            - str for text-only models
            - List[Dict] for OpenAI-style multimodal
            - Dict for structured prompts
        """
        pass

    @abstractmethod
    def apply_gradient(self, gradient: "TextualGradient") -> "OptimizablePrompt":
        """Apply a textual gradient to create a new prompt version.

        Note: This is typically implemented by delegating to a ModelAdapter,
        which uses an LLM to perform the actual rewrite.

        Parameters
        ----------
        gradient : TextualGradient
            The gradient describing how to improve this prompt.

        Returns
        -------
        OptimizablePrompt
            A new prompt instance with the gradient applied.
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize to a dictionary for storage.

        Returns
        -------
        Dict[str, Any]
            Must contain enough info to reconstruct via deserialize().
        """
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> "OptimizablePrompt":
        """Reconstruct from serialized form.

        Parameters
        ----------
        data : Dict[str, Any]
            Output from a previous serialize() call.

        Returns
        -------
        OptimizablePrompt
            A new instance with the same state.
        """
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version identifier (e.g., 'v1.0', 'v1.1')."""
        pass

    @abstractmethod
    def bump_version(self) -> "OptimizablePrompt":
        """Return a copy with an incremented version."""
        pass


class TextualGradient(ABC):
    """Represents a 'gradient' in the textual prompt space.

    Unlike neural network gradients (which are vectors), textual gradients
    are natural language descriptions of how to improve a prompt.

    Example
    -------
    "The current prompt is too formal. Make it more conversational
    by using contractions and adding phrases like 'By the way...'"
    """

    @abstractmethod
    def __str__(self) -> str:
        """Return the gradient as a human-readable string."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the gradient."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextualGradient":
        """Deserialize a gradient."""
        pass


class Feedback(ABC):
    """Feedback on a single model output.

    Feedback can come from:
    - Human users (via CLI commands like /bad, /rewrite)
    - Auto-judge models (LLM-based evaluation)
    - Task-specific metrics (e.g., code tests passing)
    """

    @abstractmethod
    def to_score(self) -> float:
        """Convert feedback to a scalar score in [0, 1].

        Returns
        -------
        float
            0.0 = completely wrong, 1.0 = perfect
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the feedback."""
        pass

    @property
    @abstractmethod
    def is_negative(self) -> bool:
        """Return True if this indicates a failure (score < 0.5)."""
        pass


class Experience(ABC):
    """A single interaction experience with the model.

    This is analogous to a training sample in ML, but for prompt optimization.
    An experience consists of:
    - The input context (what was given to the model)
    - The model's output
    - Optional feedback on the output
    """

    @abstractmethod
    def get_input(self) -> Any:
        """Return the input that was fed to the model."""
        pass

    @abstractmethod
    def get_output(self) -> Any:
        """Return the model's output."""
        pass

    @abstractmethod
    def get_feedback(self) -> Optional[Feedback]:
        """Return the feedback, if any."""
        pass

    @abstractmethod
    def attach_feedback(self, feedback: Feedback) -> "Experience":
        """Return a copy with feedback attached."""
        pass

    @abstractmethod
    def to_training_sample(self) -> Dict[str, Any]:
        """Convert to a format suitable for gradient computation.

        Returns
        -------
        Dict[str, Any]
            Typically contains 'input', 'output', 'feedback' keys.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this experience."""
        pass

    @property
    @abstractmethod
    def is_failure(self) -> bool:
        """Return True if feedback exists and is negative."""
        pass


class ModelAdapter(ABC):
    """Adapter for different LLM APIs (OpenAI, Anthropic, etc.).

    This is the core abstraction that makes the framework model-agnostic.
    Each adapter implements:
    - Generation (calling the model API)
    - Gradient computation (using a judge model to analyze failures)
    - Gradient application (using an LLM to rewrite prompts)
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')."""
        pass

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Return True if this model supports image inputs."""
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Return the maximum context length."""
        pass

    @abstractmethod
    def count_tokens(self, prompt: OptimizablePrompt) -> int:
        """Count the number of tokens in a prompt.

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt to count.

        Returns
        -------
        int
            Token count (approximate for some tokenizers).
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List[Experience]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Any:
        """Generate a response from the model.

        Parameters
        ----------
        prompt : OptimizablePrompt
            The system prompt / instruction.
        context : Optional[List[Experience]]
            Prior conversation turns (for chat models).
        temperature : float
            Sampling temperature (0 = deterministic, 1 = creative).
        **kwargs
            Model-specific parameters.

        Returns
        -------
        Any
            The model's output (typically str or multimodal content).
        """
        pass

    @abstractmethod
    def compute_gradient(
        self,
        prompt: OptimizablePrompt,
        failures: List[Experience],
        target: Optional[str] = None,
        **kwargs,
    ) -> TextualGradient:
        """Compute a textual gradient by analyzing failures.

        This is the "forward pass" of TGD (Textual Gradient Descent).

        Parameters
        ----------
        prompt : OptimizablePrompt
            The current prompt that produced the failures.
        failures : List[Experience]
            Experiences with negative feedback.
        target : Optional[str]
            User-specified optimization direction (e.g., "more human-like").
        **kwargs
            Additional instructions for the judge model.

        Returns
        -------
        TextualGradient
            A gradient describing how to improve the prompt.
        """
        pass

    @abstractmethod
    def apply_gradient(
        self,
        prompt: OptimizablePrompt,
        gradient: TextualGradient,
        conservative: bool = False,
        **kwargs,
    ) -> OptimizablePrompt:
        """Apply a gradient to update the prompt.

        This is the "backward pass" of TGD, typically done by asking
        an LLM to rewrite the prompt based on the gradient.

        Parameters
        ----------
        prompt : OptimizablePrompt
            The current prompt.
        gradient : TextualGradient
            The gradient to apply.
        conservative : bool
            If True, make minimal changes (analogous to low learning rate).
        **kwargs
            Additional rewrite instructions.

        Returns
        -------
        OptimizablePrompt
            A new prompt with the gradient applied.
        """
        pass

    @abstractmethod
    def validate_prompt(self, prompt: OptimizablePrompt) -> List[str]:
        """Check if a prompt is compatible with this model.

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt to validate.

        Returns
        -------
        List[str]
            List of issues (empty if compatible).
        """
        pass


class PromptSerializer(Protocol):
    """Protocol for saving/loading prompts to disk.

    Implementations can support YAML, JSON, or custom formats.
    """

    def save(self, prompt: OptimizablePrompt, path: Path) -> None:
        """Save a prompt to disk."""
        ...

    def load(self, path: Path) -> OptimizablePrompt:
        """Load a prompt from disk."""
        ...
