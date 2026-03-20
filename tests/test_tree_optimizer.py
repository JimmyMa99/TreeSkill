"""Test TreeAwareOptimizer functionality.

This test suite validates:
1. analyze_split_need() - Split analysis
2. generate_child_prompts() - Child prompt generation
3. analyze_prune_need() - Pruning decision
4. optimize_prompt_section() - Section-wise optimization
5. optimize_tree() - Full tree optimization

All tests use Mock adapters, no real API calls.
"""

import json
import sys
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill.core.tree_optimizer import (
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    TreeOptimizationResult,
)
from evoskill.core.prompts import TextPrompt
from evoskill.core.experience import (
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
)
from evoskill.core.abc import ModelAdapter, OptimizablePrompt


# ---------------------------------------------------------------------------
# Mock Adapter
# ---------------------------------------------------------------------------

class MockAdapter(ModelAdapter):
    """Mock adapter for testing without real API calls."""

    def __init__(self):
        self.call_count = 0
        self.responses = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def supports_vision(self) -> bool:
        return False

    @property
    def max_context_tokens(self) -> int:
        return 4096

    def generate(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Any:
        """Mock generate."""
        self.call_count += 1
        return f"Mock response #{self.call_count}"

    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Mock API call."""
        self.call_count += 1

        # Return canned response if available
        if self.responses:
            return self.responses.pop(0)

        # Default response based on content
        content = messages[-1].get("content", "")

        if "Should this prompt be split?" in content:
            # Split analysis response
            return json.dumps([
                {"name": "formal_style", "description": "Formal writing tasks", "focus": "Professional tone"},
                {"name": "casual_style", "description": "Casual writing tasks", "focus": "Friendly tone"},
            ])
        elif "Generate specialized prompts" in content or (system and "specialised child prompts" in system):
            # Child prompt generation response
            return json.dumps([
                {
                    "name": "formal_style",
                    "description": "Formal writing tasks",
                    "system_prompt": "You are a professional assistant. Use formal language.",
                },
                {
                    "name": "casual_style",
                    "description": "Casual writing tasks",
                    "system_prompt": "You are a friendly assistant. Use casual language.",
                },
            ])
        elif system and "Rewrite ONLY the instruction" in system:
            # Section rewrite response
            return "This is the rewritten instruction section."
        else:
            # Default gradient application response
            return "Improved system prompt based on feedback analysis."

    def _count_tokens_impl(self, text: str) -> int:
        """Mock token counting."""
        return len(text.split())

    def count_tokens(self, prompt) -> int:
        """Mock token counting."""
        return len(str(prompt).split())

    def compute_gradient(self, prompt=None, experiences=None, failures=None, target=None, **kwargs):
        """Mock gradient computation."""
        from evoskill.core.gradient import SimpleGradient
        return SimpleGradient(text="Mock gradient: improve clarity")

    def apply_gradient(self, prompt=None, gradient=None, **kwargs):
        """Mock gradient application."""
        return prompt

    def validate_prompt(self, prompt):
        """Mock validation — always passes."""
        return []


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_adapter():
    """Create a mock adapter."""
    return MockAdapter()


@pytest.fixture
def tree_config():
    """Create a tree optimizer config."""
    return TreeOptimizerConfig(
        auto_split=True,
        auto_prune=True,
        prune_threshold=0.3,
        min_samples_for_split=5,
        max_tree_depth=3,
        optimization_order="bottom_up",
        section="all",
    )


@pytest.fixture
def tree_optimizer(mock_adapter, tree_config):
    """Create a tree optimizer with mock adapter."""
    return TreeAwareOptimizer(
        adapter=mock_adapter,
        config=tree_config,
    )


@pytest.fixture
def sample_experiences():
    """Create sample experiences with feedback."""
    experiences = []

    # Create 7 experiences with mixed feedback
    for i in range(7):
        # Create conversation
        exp = ConversationExperience(
            messages=[
                {"role": "user", "content": f"Write a {['formal', 'casual'][i % 2]} email"},
            ],
            response=f"Here's your email draft {i}...",
        )

        # Add feedback
        if i % 3 == 0:
            # Negative feedback
            feedback = CompositeFeedback(
                feedback_type=FeedbackType.CRITIQUE,
                critique=f"Wrong tone for task {i}",
                score=0.3,
            )
        elif i % 3 == 1:
            # Positive feedback
            feedback = CompositeFeedback(
                feedback_type=FeedbackType.SCORE,
                critique="Good response",
                score=0.8,
            )
        else:
            # Neutral feedback
            feedback = CompositeFeedback(
                feedback_type=FeedbackType.SCORE,
                critique="Okay response",
                score=0.5,
            )

        exp = exp.attach_feedback(feedback)
        experiences.append(exp)

    return experiences


@pytest.fixture
def sample_prompt():
    """Create a sample prompt."""
    return TextPrompt(
        content="You are a helpful assistant. Write emails in an appropriate tone.",
        metadata={"version": "v1.0"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_analyze_split_need_recommends_split(tree_optimizer, sample_prompt, sample_experiences):
    """Test that split analysis recommends splitting when needed."""
    # Analyze
    specs = tree_optimizer.analyze_split_need(
        prompt=sample_prompt,
        experiences=sample_experiences,
    )

    # Should recommend split (because we have mixed tone requirements)
    assert specs is not None, "Should recommend split for contradictory feedback"
    assert isinstance(specs, list), "Should return a list"
    assert len(specs) >= 2, "Should recommend at least 2 children"

    # Check spec structure
    for spec in specs:
        assert "name" in spec, "Spec should have 'name'"
        assert "description" in spec, "Spec should have 'description'"
        assert "focus" in spec, "Spec should have 'focus'"


def test_analyze_split_need_insufficient_samples(tree_optimizer, sample_prompt, sample_experiences):
    """Test that split analysis returns None with insufficient samples."""
    # Only provide 2 experiences (less than min_samples_for_split=5)
    few_experiences = sample_experiences[:2]

    specs = tree_optimizer.analyze_split_need(
        prompt=sample_prompt,
        experiences=few_experiences,
    )

    # Should not recommend split
    assert specs is None, "Should not split with insufficient samples"


def test_generate_child_prompts(tree_optimizer, sample_prompt):
    """Test child prompt generation."""
    children_specs = [
        {"name": "formal_style", "description": "Formal writing", "focus": "Professional"},
        {"name": "casual_style", "description": "Casual writing", "focus": "Friendly"},
    ]

    children = tree_optimizer.generate_child_prompts(
        parent_prompt=sample_prompt,
        children_specs=children_specs,
    )

    # Check results
    assert isinstance(children, list), "Should return a list"
    assert len(children) == 2, "Should generate 2 child prompts"

    for child in children:
        assert isinstance(child, TextPrompt), "Should return TextPrompt"
        assert hasattr(child, "content"), "Should have content"


def test_analyze_prune_need_low_performance(tree_optimizer, tree_config):
    """Test pruning decision for low performance nodes."""
    # Create mock node
    node = Mock()
    node.name = "test_node"
    node.age = 10  # past protection period

    # Low performance metrics
    metrics = {
        "performance_score": 0.2,  # Below threshold (0.3)
        "usage_count": 10,
        "success_rate": 0.2,
    }

    should_prune = tree_optimizer.analyze_prune_need(node, metrics)

    # Should prune
    assert should_prune is True, "Should prune low performance nodes"


def test_analyze_prune_need_low_usage(tree_optimizer):
    """Test pruning decision for low usage nodes."""
    # Create mock node
    node = Mock()
    node.name = "unused_node"
    node.age = 10

    # Low usage metrics
    metrics = {
        "performance_score": 0.6,
        "usage_count": 1,  # Very low usage
        "success_rate": 0.5,
    }

    should_prune = tree_optimizer.analyze_prune_need(node, metrics)

    # Should prune
    assert should_prune is True, "Should prune low usage nodes"


def test_analyze_prune_need_low_success_rate(tree_optimizer):
    """Test pruning decision for low success rate nodes."""
    # Create mock node
    node = Mock()
    node.name = "failing_node"
    node.age = 10

    # Low success rate
    metrics = {
        "performance_score": 0.5,
        "usage_count": 10,
        "success_rate": 0.2,  # Below 0.3
    }

    should_prune = tree_optimizer.analyze_prune_need(node, metrics)

    # Should prune
    assert should_prune is True, "Should prune low success rate nodes"


def test_analyze_prune_need_keeps_good_nodes(tree_optimizer):
    """Test that good nodes are not pruned."""
    # Create mock node
    node = Mock()
    node.name = "good_node"
    node.age = 10

    # Good metrics
    metrics = {
        "performance_score": 0.7,
        "usage_count": 15,
        "success_rate": 0.8,
    }

    should_prune = tree_optimizer.analyze_prune_need(node, metrics)

    # Should NOT prune
    assert should_prune is False, "Should NOT prune good nodes"


def test_optimize_prompt_section_instruction_only(tree_optimizer, sample_prompt, sample_experiences):
    """Test section-wise optimization for instruction only."""
    optimized = tree_optimizer.optimize_prompt_section(
        prompt=sample_prompt,
        experiences=sample_experiences,
        section="instruction",
    )

    # Check result
    assert isinstance(optimized, TextPrompt), "Should return TextPrompt"
    assert hasattr(optimized, "content"), "Should have content"
    # The instruction section should be different
    # (In real implementation, we'd verify only instruction changed)


def test_optimize_prompt_section_all(tree_optimizer, sample_prompt, sample_experiences):
    """Test full prompt optimization."""
    optimized = tree_optimizer.optimize_prompt_section(
        prompt=sample_prompt,
        experiences=sample_experiences,
        section="all",
    )

    # Check result
    assert isinstance(optimized, TextPrompt), "Should return TextPrompt"
    assert hasattr(optimized, "content"), "Should have content"


def test_optimize_tree_integration(tree_optimizer, sample_experiences):
    """Test complete tree optimization (integration test)."""
    # Create mock tree
    tree = Mock()
    root = Mock()
    root.name = "root"
    root.age = 0
    root.children = {}

    # Use a real Skill-like object to avoid Mock subscript issues
    from evoskill.schema import Skill
    root.skill = Skill(
        name="root",
        system_prompt="You are a helpful assistant.",
        version="v1.0",
    )
    tree.root = root
    tree.add_child = Mock()
    tree.prune = Mock()

    # Optimize
    result = tree_optimizer.optimize_tree(
        tree=tree,
        experiences=sample_experiences,
    )

    # Check result
    assert isinstance(result, TreeOptimizationResult), "Should return TreeOptimizationResult"
    assert result.tree is tree, "Should return the same tree object"
    assert result.nodes_optimized >= 0, "Should track nodes optimized"
    assert result.splits_performed >= 0, "Should track splits"
    assert result.prunes_performed >= 0, "Should track prunes"


def test_tree_optimizer_config_defaults():
    """Test TreeOptimizerConfig default values."""
    config = TreeOptimizerConfig()

    assert config.auto_split is True
    assert config.auto_prune is True
    assert config.prune_threshold == 0.3
    assert config.min_samples_for_split == 5
    assert config.max_tree_depth == 3
    assert config.optimization_order == "bottom_up"
    assert config.section == "all"


def test_tree_optimizer_custom_config():
    """Test TreeOptimizerConfig with custom values."""
    config = TreeOptimizerConfig(
        auto_split=False,
        auto_prune=False,
        prune_threshold=0.5,
        min_samples_for_split=10,
        max_tree_depth=5,
        optimization_order="top_down",
        section="instruction",
    )

    assert config.auto_split is False
    assert config.auto_prune is False
    assert config.prune_threshold == 0.5
    assert config.min_samples_for_split == 10
    assert config.max_tree_depth == 5
    assert config.optimization_order == "top_down"
    assert config.section == "instruction"


# ---------------------------------------------------------------------------
# Run Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
