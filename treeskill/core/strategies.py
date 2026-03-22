"""Optimization Strategies.

Different strategies for applying gradients to prompts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from treeskill.core.abc import ModelAdapter, OptimizablePrompt, TextualGradient


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies.

    Strategies control how aggressively to apply gradients.
    """

    @abstractmethod
    def apply_gradient(
        self,
        adapter: "ModelAdapter",
        prompt: "OptimizablePrompt",
        gradient: "TextualGradient",
    ) -> "OptimizablePrompt":
        """Apply gradient to prompt according to strategy.

        Parameters
        ----------
        adapter : ModelAdapter
            The model adapter.
        prompt : OptimizablePrompt
            Current prompt.
        gradient : TextualGradient
            Gradient to apply.

        Returns
        -------
        OptimizablePrompt
            Updated prompt.
        """
        pass


class ConservativeStrategy(OptimizationStrategy):
    """Conservative optimization strategy.

    Makes minimal changes to the prompt, preserving existing structure.
    Analogous to using a low learning rate in gradient descent.

    Use when:
    - Prompt is already good and needs fine-tuning
    - Want to preserve existing functionality
    - Risk of breaking things is high
    """

    def apply_gradient(
        self,
        adapter: "ModelAdapter",
        prompt: "OptimizablePrompt",
        gradient: "TextualGradient",
    ) -> "OptimizablePrompt":
        """Apply gradient conservatively."""
        return adapter.apply_gradient(
            prompt=prompt,
            gradient=gradient,
            conservative=True,
        )


class AggressiveStrategy(OptimizationStrategy):
    """Aggressive optimization strategy.

    Makes substantial changes to the prompt when clear improvements are identified.
    Analogous to using a high learning rate in gradient descent.

    Use when:
    - Prompt has major issues
    - Need significant improvements
    - Willing to risk breaking things
    """

    def apply_gradient(
        self,
        adapter: "ModelAdapter",
        prompt: "OptimizablePrompt",
        gradient: "TextualGradient",
    ) -> "OptimizablePrompt":
        """Apply gradient aggressively."""
        return adapter.apply_gradient(
            prompt=prompt,
            gradient=gradient,
            conservative=False,
        )


class AdaptiveStrategy(OptimizationStrategy):
    """Adaptive optimization strategy.

    Starts aggressive, becomes more conservative as optimization progresses.
    Analogous to learning rate scheduling.

    Use when:
    - Want best of both worlds
    - Initial prompt needs work but don't want to overshoot later
    """

    def __init__(self, initial_patience: int = 2):
        """Initialize adaptive strategy.

        Parameters
        ----------
        initial_patience : int
            Number of steps before becoming conservative.
        """
        self.initial_patience = initial_patience
        self.step_count = 0

    def apply_gradient(
        self,
        adapter: "ModelAdapter",
        prompt: "OptimizablePrompt",
        gradient: "TextualGradient",
    ) -> "OptimizablePrompt":
        """Apply gradient adaptively."""
        self.step_count += 1
        conservative = self.step_count > self.initial_patience

        strategy = "conservative" if conservative else "aggressive"
        # Could add logging here

        return adapter.apply_gradient(
            prompt=prompt,
            gradient=gradient,
            conservative=conservative,
        )

    def reset(self):
        """Reset step counter."""
        self.step_count = 0


# Factory function
def get_strategy(name: str) -> OptimizationStrategy:
    """Get a strategy by name.

    Parameters
    ----------
    name : str
        Strategy name: "conservative", "aggressive", or "adaptive".

    Returns
    -------
    OptimizationStrategy
        The strategy instance.

    Raises
    ------
    ValueError
        If strategy name is unknown.
    """
    strategies = {
        "conservative": ConservativeStrategy(),
        "aggressive": AggressiveStrategy(),
        "adaptive": AdaptiveStrategy(),
    }

    if name not in strategies:
        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[name]
