"""Base ModelAdapter with common functionality.

This provides a foundation for building adapters for different LLM APIs.
Concrete adapters (OpenAI, Anthropic, etc.) inherit from this class
and implement the abstract methods.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from evoskill.core.abc import ModelAdapter, OptimizablePrompt, TextualGradient, Experience
from evoskill.core.gradient import SimpleGradient, MultimodalGradient


logger = logging.getLogger(__name__)


class BaseModelAdapter(ModelAdapter):
    """Base implementation of ModelAdapter with shared logic.

    This class implements common functionality:
    - Token counting (approximate)
    - Prompt validation
    - Gradient computation scaffolding
    - Gradient application scaffolding

    Subclasses need to implement:
    - generate()
    - _call_api() (low-level API call)
    - _count_tokens_impl() (accurate tokenization)
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model_name : str
            Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').
        api_key : Optional[str]
            API key (can also be set via environment variable).
        base_url : Optional[str]
            Base URL for API (for custom endpoints).
        **kwargs
            Additional model-specific parameters.
        """
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_vision(self) -> bool:
        """Default: False. Override in vision-capable adapters."""
        return False

    @property
    def max_context_tokens(self) -> int:
        """Default: 4096. Override for models with larger context."""
        return 4096

    # ------------------------------------------------------------------
    # Abstract methods that subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List[Experience]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Any:
        """Generate a response from the model.

        This is the primary method that subclasses implement.
        """
        pass

    @abstractmethod
    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Low-level API call.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            OpenAI-style message list.
        system : Optional[str]
            System prompt (for models that separate it from messages).
        temperature : float
            Sampling temperature.

        Returns
        -------
        str
            The model's response text.
        """
        pass

    @abstractmethod
    def _count_tokens_impl(self, text: str) -> int:
        """Count tokens using the model's tokenizer.

        Subclasses should use tiktoken, Anthropic's tokenizer, etc.

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        -------
        int
            Token count.
        """
        pass

    # ------------------------------------------------------------------
    # Implemented methods with default behavior
    # ------------------------------------------------------------------

    def count_tokens(self, prompt: OptimizablePrompt) -> int:
        """Count tokens in a prompt.

        Default implementation uses approximate counting.
        Override for more accurate counting.
        """
        # Extract text from prompt
        if hasattr(prompt, "content"):
            text = prompt.content
        elif hasattr(prompt, "text"):
            text = prompt.text
        elif hasattr(prompt, "instruction"):
            text = prompt.instruction
        else:
            text = str(prompt.to_model_input())

        return self._count_tokens_impl(text)

    def validate_prompt(self, prompt: OptimizablePrompt) -> List[str]:
        """Check if a prompt is compatible with this model.

        Returns a list of issues (empty if compatible).
        """
        issues = []

        # Check 1: Token length
        tokens = self.count_tokens(prompt)
        if tokens > self.max_context_tokens:
            issues.append(
                f"Prompt too long: {tokens} tokens > {self.max_context_tokens} max"
            )

        # Check 2: Multimodal support
        from evoskill.core.prompts import MultimodalPrompt

        if isinstance(prompt, MultimodalPrompt) and not self.supports_vision:
            issues.append("Model does not support vision, but prompt contains images")

        # Check 3: Required fields
        if not hasattr(prompt, "to_model_input"):
            issues.append("Prompt must implement to_model_input()")

        return issues

    def compute_gradient(
        self,
        prompt: OptimizablePrompt,
        failures: List[Experience],
        target: Optional[str] = None,
        **kwargs,
    ) -> TextualGradient:
        """Compute a textual gradient by analyzing failures.

        This method:
        1. Formats the failure cases
        2. Calls the judge model (typically the same model, but can be overridden)
        3. Returns a gradient describing how to improve the prompt

        Parameters
        ----------
        prompt : OptimizablePrompt
            The current prompt.
        failures : List[Experience]
            Experiences with negative feedback.
        target : Optional[str]
            User-specified optimization direction.
        **kwargs
            Additional parameters (e.g., different judge model).

        Returns
        -------
        TextualGradient
            A gradient describing how to improve the prompt.
        """
        # Format failures
        failure_descriptions = []
        for exp in failures:
            if hasattr(exp, "get_input"):
                user_input = exp.get_input()
                if isinstance(user_input, list):
                    # Conversation format
                    user_text = user_input[-1].get("content", "")
                elif isinstance(user_input, dict):
                    user_text = user_input.get("text", str(user_input))
                else:
                    user_text = str(user_input)

                agent_response = exp.get_output()
                if isinstance(agent_response, dict):
                    agent_text = agent_response.get("text", str(agent_response))
                else:
                    agent_text = str(agent_response)

                feedback = exp.get_feedback()
                if feedback:
                    if hasattr(feedback, "correction") and feedback.correction:
                        feedback_text = f"Ideal response: {feedback.correction}"
                    elif hasattr(feedback, "critique") and feedback.critique:
                        feedback_text = f"Critique: {feedback.critique}"
                    else:
                        feedback_text = f"Score: {feedback.to_score():.2f}"
                else:
                    feedback_text = "No feedback provided"

                failure_descriptions.append(
                    f"- User: \"{user_text[:100]}...\"\n"
                    f"  Agent: \"{agent_text[:100]}...\"\n"
                    f"  Feedback: {feedback_text}"
                )

        failures_block = "\n".join(failure_descriptions)

        # Build judge prompt
        prompt_text = self._extract_prompt_text(prompt)
        target_hint = (
            f"\n\nOptimization target: \"{target}\". Keep this direction in mind."
            if target
            else ""
        )

        judge_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt engineer. Analyze conversation failures "
                    "and explain WHY the system prompt failed to guide the agent correctly. "
                    "Be specific and actionable."
                    + target_hint
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current System Prompt:\n\"\"\"\n{prompt_text}\n\"\"\"\n\n"
                    f"Failures:\n{failures_block}\n\n"
                    "Explain why the prompt led to these failures. "
                    "Focus on specific issues and how to fix them."
                ),
            },
        ]

        # Call judge model
        gradient_text = self._call_api(
            messages=judge_messages,
            system=None,
            temperature=0.3,  # Lower temperature for analysis
            **kwargs,
        )

        logger.info("Computed gradient: %s", gradient_text[:100] + "...")

        return SimpleGradient(
            text=gradient_text,
            metadata={
                "num_failures": len(failures),
                "target": target,
                "model": self.model_name,
            },
        )

    def apply_gradient(
        self,
        prompt: OptimizablePrompt,
        gradient: TextualGradient,
        conservative: bool = False,
        **kwargs,
    ) -> OptimizablePrompt:
        """Apply a gradient to update the prompt.

        This uses the model to rewrite the prompt based on the gradient.

        Parameters
        ----------
        prompt : OptimizablePrompt
            Current prompt.
        gradient : TextualGradient
            Gradient describing how to improve.
        conservative : bool
            If True, make minimal changes (analogous to low learning rate).
        **kwargs
            Additional parameters.

        Returns
        -------
        OptimizablePrompt
            New prompt with gradient applied.
        """
        prompt_text = self._extract_prompt_text(prompt)
        gradient_text = str(gradient)

        target_hint = ""
        if hasattr(prompt, "target") and prompt.target:
            target_hint = f"\nThe user's optimization target is: \"{prompt.target}\"."

        conservative_hint = ""
        if conservative:
            conservative_hint = (
                "\n\nIMPORTANT: Make MINIMAL changes. "
                "Only adjust the parts clearly identified as problematic."
            )

        rewrite_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt engineer. Based on the failure analysis, "
                    "rewrite the System Prompt to fix the identified issues WITHOUT "
                    "breaking existing functionality."
                    + target_hint
                    + conservative_hint
                    + "\n\nReturn ONLY the new prompt text. No commentary, no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current System Prompt:\n\"\"\"\n{prompt_text}\n\"\"\"\n\n"
                    f"Failure Analysis:\n{gradient_text}\n\n"
                    "Rewrite the system prompt now."
                ),
            },
        ]

        new_prompt_text = self._call_api(
            messages=rewrite_messages,
            system=None,
            temperature=0.5,
            **kwargs,
        )

        # Clean up response - remove markdown code fences if present
        new_prompt_text = new_prompt_text.strip()
        if new_prompt_text.startswith("```"):
            # Remove first and last line if they're code fences
            lines = new_prompt_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_prompt_text = "\n".join(lines).strip()

        # Create new prompt with updated content
        new_prompt = prompt.bump_version()

        logger.info("After bump_version:")
        logger.info(f"  new_prompt.version = {new_prompt.version}")
        logger.info(f"  new_prompt.content (before update) = {new_prompt.content[:100]}...")

        # Update the content
        if hasattr(new_prompt, "content"):
            new_prompt.content = new_prompt_text
            logger.info(f"  ✅ Updated new_prompt.content to: {new_prompt.content[:100]}...")
        elif hasattr(new_prompt, "text"):
            new_prompt.text = new_prompt_text
            logger.info(f"  ✅ Updated new_prompt.text to: {new_prompt.text[:100]}...")
        elif hasattr(new_prompt, "instruction"):
            new_prompt.instruction = new_prompt_text
            logger.info(f"  ✅ Updated new_prompt.instruction to: {new_prompt.instruction[:100]}...")

        logger.info(f"Final new_prompt.version = {new_prompt.version}")
        logger.info(f"Final new_prompt.content = {new_prompt.content[:100]}...")
        logger.info("Applied gradient → new prompt version: %s", new_prompt.version)

        return new_prompt

    def _extract_prompt_text(self, prompt: OptimizablePrompt) -> str:
        """Extract text content from a prompt."""
        if hasattr(prompt, "content"):
            return prompt.content
        elif hasattr(prompt, "text"):
            return prompt.text
        elif hasattr(prompt, "instruction"):
            return prompt.instruction
        else:
            model_input = prompt.to_model_input()
            if isinstance(model_input, str):
                return model_input
            elif isinstance(model_input, dict):
                return model_input.get("text", model_input.get("instruction", str(model_input)))
            else:
                return str(model_input)
