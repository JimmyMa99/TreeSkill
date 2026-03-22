"""Prompt Validators.

Validators evaluate prompt quality on a test set or using other metrics.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from treeskill.core.abc import ModelAdapter, OptimizablePrompt, Experience
from treeskill.core.experience import CompositeFeedback


logger = logging.getLogger(__name__)


class AutoValidator:
    """Automatic validator that tests prompts on a validation set.

    This validator:
    1. Runs the prompt on test cases
    2. Collects feedback (manual or automatic)
    3. Aggregates scores to produce a validation score

    Parameters
    ----------
    adapter : ModelAdapter
        Model adapter to use for generation.
    test_cases : List[Experience]
        Test cases (experiences with input/output).
    feedback_fn : Optional[Callable]
        Function to generate feedback from (prompt, input, output).
        If None, assumes test_cases already have feedback.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        test_cases: List[Experience],
        feedback_fn: Optional[Callable] = None,
    ):
        self.adapter = adapter
        self.test_cases = test_cases
        self.feedback_fn = feedback_fn

    def validate(self, prompt: OptimizablePrompt) -> float:
        """Validate a prompt and return a score (higher is better).

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt to validate.

        Returns
        -------
        float
            Validation score (0.0 to 1.0).
        """
        logger.info(f"Validating prompt on {len(self.test_cases)} test cases")

        scores = []

        for i, test_case in enumerate(self.test_cases):
            # Generate response
            try:
                response = self.adapter.generate(
                    prompt=prompt,
                    context=[test_case],  # Provide test case as context
                )
            except Exception as e:
                logger.warning(f"Test case {i+1} failed: {e}")
                scores.append(0.0)
                continue

            # Get feedback
            if self.feedback_fn:
                feedback = self.feedback_fn(
                    prompt,
                    test_case.get_input(),
                    response,
                )
            else:
                feedback = test_case.get_feedback()

            # Extract score
            if feedback:
                if hasattr(feedback, 'to_score'):
                    score = feedback.to_score()
                elif isinstance(feedback, CompositeFeedback):
                    score = feedback.to_score()
                elif hasattr(feedback, 'score'):
                    score = feedback.score
                else:
                    score = 0.5  # Neutral score
                scores.append(score)
            else:
                logger.warning(f"Test case {i+1} has no feedback, using neutral score")
                scores.append(0.5)

        # Aggregate scores
        if not scores:
            logger.warning("No valid test cases, returning neutral score")
            return 0.5

        avg_score = sum(scores) / len(scores)
        logger.info(
            f"Validation score: {avg_score:.3f} "
            f"(from {len(scores)} cases, min={min(scores):.3f}, max={max(scores):.3f})"
        )

        return avg_score


class MetricValidator:
    """Validator that uses a specific metric function.

    Parameters
    ----------
    metric_fn : Callable
        Function that takes (prompt, adapter) and returns a score.
    """

    def __init__(self, metric_fn: Callable):
        self.metric_fn = metric_fn

    def validate(self, prompt: OptimizablePrompt) -> float:
        """Validate using the metric function."""
        return self.metric_fn(prompt)


class CompositeValidator:
    """Validator that combines multiple validators.

    Parameters
    ----------
    validators : List[Callable]
        List of validator functions.
    weights : Optional[List[float]]
        Weights for each validator (must sum to 1.0).
        If None, uses equal weights.
    """

    def __init__(
        self,
        validators: List[Callable],
        weights: Optional[List[float]] = None,
    ):
        if weights and len(weights) != len(validators):
            raise ValueError("Number of weights must match number of validators")

        if weights and abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

        self.validators = validators
        self.weights = weights or [1.0 / len(validators)] * len(validators)

    def validate(self, prompt: OptimizablePrompt) -> float:
        """Validate using all validators and combine scores."""
        scores = [
            validator(prompt) * weight
            for validator, weight in zip(self.validators, self.weights)
        ]
        return sum(scores)


# Convenience functions

def create_simple_validator(
    adapter: ModelAdapter,
    test_cases: List[Experience],
) -> Callable[[OptimizablePrompt], float]:
    """Create a simple validator function.

    Parameters
    ----------
    adapter : ModelAdapter
        Model adapter.
    test_cases : List[Experience]
        Test cases.

    Returns
    -------
    Callable
        Validator function.
    """
    validator = AutoValidator(adapter, test_cases)

    def validate_fn(prompt: OptimizablePrompt) -> float:
        return validator.validate(prompt)

    return validate_fn


def create_metric_validator(
    metric_fn: Callable,
) -> Callable[[OptimizablePrompt], float]:
    """Create a metric-based validator function.

    Parameters
    ----------
    metric_fn : Callable
        Metric function (prompt -> score).

    Returns
    -------
    Callable
        Validator function.
    """
    validator = MetricValidator(metric_fn)

    def validate_fn(prompt: OptimizablePrompt) -> float:
        return validator.validate(prompt)

    return validate_fn
