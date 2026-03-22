"""Train-free Optimizer Configuration.

Configuration classes for the TGD optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from treeskill.core.abc import OptimizablePrompt, Experience


@dataclass
class OptimizerConfig:
    """Configuration for TrainFreeOptimizer.

    Attributes
    ----------
    max_steps : int
        Maximum number of optimization steps (default: 3).
    gradient_accumulation_steps : int
        Number of experiences to use for computing gradient (default: 10).
    conservative : bool
        If True, use conservative update strategy (default: False).
    early_stopping_patience : int
        Stop if no improvement after N steps (default: 2).
    early_stopping_threshold : float
        Minimum improvement threshold for early stopping (default: 0.01).
    validate_every_step : bool
        Run validation after each optimization step (default: True).
    target : Optional[str]
        Optimization direction/goal (e.g., "reduce verbosity").
    """

    max_steps: int = 3
    gradient_accumulation_steps: int = 10
    conservative: bool = False
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.01
    validate_every_step: bool = True
    target: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result of an optimization run.

    Attributes
    ----------
    initial_prompt : OptimizablePrompt
        The original prompt.
    optimized_prompt : OptimizablePrompt
        The optimized prompt.
    steps_taken : int
        Number of optimization steps performed.
    history : List[OptimizationStep]
        History of all optimization steps.
    final_score : Optional[float]
        Final validation score (if validation was run).
    improvement : Optional[float]
        Improvement over initial score (if validation was run).
    converged : bool
        Whether optimization converged (early stopping triggered).
    """

    initial_prompt: OptimizablePrompt
    optimized_prompt: OptimizablePrompt
    steps_taken: int
    history: List[OptimizationStep] = field(default_factory=list)
    final_score: Optional[float] = None
    improvement: Optional[float] = None
    converged: bool = False


@dataclass
class OptimizationStep:
    """A single optimization step.

    Attributes
    ----------
    step_num : int
        Step number.
    old_prompt : OptimizablePrompt
        Prompt before update.
    new_prompt : OptimizablePrompt
        Prompt after update.
    gradient : str
        Gradient text describing the update.
    num_failures : int
        Number of failure experiences used.
    validation_score : Optional[float]
        Validation score after this step (if run).
    improvement : Optional[float]
        Improvement from previous step.
    """

    step_num: int
    old_prompt: OptimizablePrompt
    new_prompt: OptimizablePrompt
    gradient: str
    num_failures: int
    validation_score: Optional[float] = None
    improvement: Optional[float] = None


# Type alias for validator functions
Validator = Callable[[OptimizablePrompt], float]
