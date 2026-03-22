"""Train-free Prompt Optimizer using Textual Gradient Descent.

This is the core optimization engine that implements TGD-based prompt
optimization without any model training. It uses only API calls to
iteratively improve prompts based on failure experiences.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from treeskill.core.abc import ModelAdapter, OptimizablePrompt, Experience, TextualGradient
from treeskill.core.optimizer_config import (
    OptimizerConfig,
    OptimizationResult,
    OptimizationStep,
    Validator,
)
from treeskill.core.experience import CompositeFeedback


logger = logging.getLogger(__name__)


class TrainFreeOptimizer:
    """Train-free Prompt Optimizer using Textual Gradient Descent.

    This optimizer iteratively improves prompts by:
    1. Collecting failure experiences (negative feedback)
    2. Computing a textual "gradient" (why did it fail?)
    3. Applying the gradient to rewrite the prompt
    4. Validating the new prompt (optional)
    5. Repeating until convergence or max_steps

    Parameters
    ----------
    adapter : ModelAdapter
        The model adapter to use for generation and gradient computation.
    config : OptimizerConfig
        Optimizer configuration.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        config: Optional[OptimizerConfig] = None,
    ):
        self.adapter = adapter
        self.config = config or OptimizerConfig()

    def optimize(
        self,
        prompt: OptimizablePrompt,
        experiences: List[Experience],
        validator: Optional[Validator] = None,
    ) -> OptimizationResult:
        """Optimize a prompt using TGD.

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt to optimize.
        experiences : List[Experience]
            List of experiences (interactions with feedback).
        validator : Optional[Validator]
            Function that validates a prompt and returns a score.
            Higher is better.

        Returns
        -------
        OptimizationResult
            The optimization result with history and metrics.
        """
        logger.info(
            f"Starting optimization with {len(experiences)} experiences, "
            f"max_steps={self.config.max_steps}"
        )

        # Step 1: Initial validation (if validator provided)
        initial_score = None
        if validator and self.config.validate_every_step:
            initial_score = validator(prompt)
            logger.info(f"Initial validation score: {initial_score:.3f}")

        # Step 2: Extract failures (negative feedback)
        failures = self._extract_failures(experiences)
        logger.info(f"Found {len(failures)} failure experiences")

        if not failures:
            logger.warning("No failure experiences found, returning original prompt")
            return OptimizationResult(
                initial_prompt=prompt,
                optimized_prompt=prompt,
                steps_taken=0,
                final_score=initial_score,
                improvement=0.0 if initial_score else None,
                converged=False,
            )

        # Step 3: Optimization loop
        current_prompt = prompt
        best_prompt = prompt
        best_score = initial_score
        history: List[OptimizationStep] = []
        no_improvement_count = 0

        for step_num in range(1, self.config.max_steps + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimization Step {step_num}/{self.config.max_steps}")
            logger.info(f"{'='*60}")

            # 3a: Compute gradient
            logger.info("Computing gradient...")
            gradient = self.adapter.compute_gradient(
                prompt=current_prompt,
                failures=failures,
                target=self.config.target,
            )
            logger.info(f"Gradient computed: {str(gradient)[:100]}...")

            # 3b: Apply gradient
            logger.info("Applying gradient...")
            new_prompt = self.adapter.apply_gradient(
                prompt=current_prompt,
                gradient=gradient,
                conservative=self.config.conservative,
            )
            logger.info(f"Prompt updated: v{new_prompt.version}")

            # 3c: Validate new prompt (optional)
            new_score = None
            improvement = None
            if validator and self.config.validate_every_step:
                new_score = validator(new_prompt)
                if best_score is not None:
                    improvement = new_score - best_score
                    logger.info(f"Validation score: {new_score:.3f} (improvement: {improvement:+.3f})")

                # Track best prompt
                if best_score is None or new_score > best_score:
                    best_prompt = new_prompt
                    best_score = new_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement for {no_improvement_count} step(s)")

            # 3d: Record step
            step = OptimizationStep(
                step_num=step_num,
                old_prompt=current_prompt,
                new_prompt=new_prompt,
                gradient=str(gradient),
                num_failures=len(failures),
                validation_score=new_score,
                improvement=improvement,
            )
            history.append(step)

            # 3e: Early stopping check
            if self._should_stop_early(no_improvement_count, improvement):
                logger.info(f"Early stopping triggered after {step_num} steps")
                break

            # Move to next iteration
            current_prompt = new_prompt

        # Step 4: Return result
        # If no validator was used, return the last updated prompt (current_prompt)
        # Otherwise return the best prompt found during validation
        final_prompt = best_prompt if validator else current_prompt
        final_score = best_score
        improvement = None
        if initial_score is not None and final_score is not None:
            improvement = final_score - initial_score

        logger.info(f"\n{'='*60}")
        logger.info(f"Optimization Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Steps taken: {len(history)}")
        if initial_score is not None:
            logger.info(f"Initial score: {initial_score:.3f}")
            logger.info(f"Final score: {final_score:.3f}")
            logger.info(f"Total improvement: {improvement:+.3f}")
        logger.info(f"Prompt version: {prompt.version} → {final_prompt.version}")

        return OptimizationResult(
            initial_prompt=prompt,
            optimized_prompt=final_prompt,
            steps_taken=len(history),
            history=history,
            final_score=final_score,
            improvement=improvement,
            converged=len(history) < self.config.max_steps,
        )

    def _extract_failures(self, experiences: List[Experience]) -> List[Experience]:
        """Extract experiences with negative feedback.

        Parameters
        ----------
        experiences : List[Experience]
            All experiences.

        Returns
        -------
        List[Experience]
            Experiences with negative feedback.
        """
        failures = []

        for exp in experiences:
            feedback = exp.get_feedback()
            if feedback is None:
                continue

            # Check if feedback is negative
            is_negative = False

            # Check by score
            if hasattr(feedback, 'to_score'):
                score = feedback.to_score()
                # Assume score < 0.5 is negative
                is_negative = score < 0.5

            # Check by feedback type
            elif isinstance(feedback, CompositeFeedback):
                is_negative = feedback.feedback_type.value in ['negative', 'correction']

            # Check by correction presence
            elif hasattr(feedback, 'correction') and feedback.correction:
                is_negative = True

            # Check by critique presence
            elif hasattr(feedback, 'critique') and feedback.critique:
                is_negative = True

            if is_negative:
                failures.append(exp)

        # Limit to gradient_accumulation_steps most recent failures
        if len(failures) > self.config.gradient_accumulation_steps:
            failures = failures[-self.config.gradient_accumulation_steps:]

        return failures

    def _should_stop_early(
        self,
        no_improvement_count: int,
        improvement: Optional[float],
    ) -> bool:
        """Check if we should stop early.

        Parameters
        ----------
        no_improvement_count : int
            Number of steps without improvement.
        improvement : Optional[float]
            Last improvement value.

        Returns
        -------
        bool
            True if should stop early.
        """
        # Check patience
        if no_improvement_count >= self.config.early_stopping_patience:
            return True

        # Check improvement threshold
        if (
            improvement is not None
            and improvement < self.config.early_stopping_threshold
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def compute_gradient(
        self,
        prompt: OptimizablePrompt,
        experiences: List[Experience],
    ) -> TextualGradient:
        """Compute gradient for a prompt (convenience method).

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt.
        experiences : List[Experience]
            Experiences to learn from.

        Returns
        -------
        TextualGradient
            The computed gradient.
        """
        failures = self._extract_failures(experiences)
        return self.adapter.compute_gradient(
            prompt=prompt,
            failures=failures,
            target=self.config.target,
        )

    def apply_gradient(
        self,
        prompt: OptimizablePrompt,
        gradient: TextualGradient,
    ) -> OptimizablePrompt:
        """Apply a gradient to a prompt (convenience method).

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt to update.
        gradient : TextualGradient
            The gradient to apply.

        Returns
        -------
        OptimizablePrompt
            Updated prompt.
        """
        return self.adapter.apply_gradient(
            prompt=prompt,
            gradient=gradient,
            conservative=self.config.conservative,
        )
