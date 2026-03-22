"""Core abstraction layer for train-free prompt optimization.

This module provides the foundational abstractions that make the framework
model-agnostic and multimodal-ready.
"""

from treeskill.core.abc import (
    OptimizablePrompt,
    TextualGradient,
    Experience,
    Feedback,
    ModelAdapter,
    PromptSerializer,
)
from treeskill.core.prompts import (
    TextPrompt,
    MultimodalPrompt,
    StructuredPrompt,
)
from treeskill.core.gradient import (
    SimpleGradient,
    MultimodalGradient,
    GradientHistory,
)
from treeskill.core.experience import (
    ConversationExperience,
    MultimodalExperience,
    CompositeFeedback,
    FeedbackType,
)
from treeskill.core.base_adapter import BaseModelAdapter
from treeskill.core.optimizer import TrainFreeOptimizer
from treeskill.core.optimizer_config import (
    OptimizerConfig,
    OptimizationResult,
    OptimizationStep,
    Validator,
)
from treeskill.core.strategies import (
    OptimizationStrategy,
    ConservativeStrategy,
    AggressiveStrategy,
    AdaptiveStrategy,
    get_strategy,
)
from treeskill.core.validators import (
    AutoValidator,
    MetricValidator,
    CompositeValidator,
    create_simple_validator,
    create_metric_validator,
)
from treeskill.core.tree_optimizer import (
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    TreeOptimizationResult,
)

__all__ = [
    # Abstract base classes
    "OptimizablePrompt",
    "TextualGradient",
    "Experience",
    "Feedback",
    "ModelAdapter",
    "PromptSerializer",
    # Base adapter
    "BaseModelAdapter",
    # Concrete prompt types
    "TextPrompt",
    "MultimodalPrompt",
    "StructuredPrompt",
    # Gradient types
    "SimpleGradient",
    "MultimodalGradient",
    "GradientHistory",
    # Experience types
    "ConversationExperience",
    "MultimodalExperience",
    "CompositeFeedback",
    "FeedbackType",
    # Optimizer
    "TrainFreeOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "OptimizationStep",
    "Validator",
    # Strategies
    "OptimizationStrategy",
    "ConservativeStrategy",
    "AggressiveStrategy",
    "AdaptiveStrategy",
    "get_strategy",
    # Validators
    "AutoValidator",
    "MetricValidator",
    "CompositeValidator",
    "create_simple_validator",
    "create_metric_validator",
    # Tree Optimizer
    "TreeAwareOptimizer",
    "TreeOptimizerConfig",
    "TreeOptimizationResult",
]
