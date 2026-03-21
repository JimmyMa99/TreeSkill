"""
Complete optimizer example.

Demonstrates how to optimize prompts with TrainFreeOptimizer.
"""

# Import directly from core modules to avoid adapter dependencies.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill.core.prompts import TextPrompt
from evoskill.core.experience import ConversationExperience, CompositeFeedback, FeedbackType
from evoskill.core.gradient import SimpleGradient
from evoskill.core.base_adapter import BaseModelAdapter
from evoskill.core.optimizer import TrainFreeOptimizer
from evoskill.core.optimizer_config import OptimizerConfig
import logging

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def create_mock_failures():
    """Create mock failure experiences."""
    failures = []

    # Failure case 1: overly verbose answer.
    exp1 = ConversationExperience(
        messages=[
            {"role": "user", "content": "What is Python?"},
        ],
        response="Python is a high-level programming language created by Guido van Rossum in 1991... (another 500 words follow)",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CRITIQUE,
            critique="The answer is too verbose and does not answer the question directly",
            score=0.3,
        ),
    )
    failures.append(exp1)

    # Failure case 2: no direct answer.
    exp2 = ConversationExperience(
        messages=[
            {"role": "user", "content": "How do I sort a list?"},
        ],
        response="Sorting is a complex topic that touches the foundations of computer science...",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CORRECTION,
            critique="No direct answer was given",
            correction="You can use the sorted() function or the list.sort() method.",
        ),
    )
    failures.append(exp2)

    # Failure case 3: messy formatting.
    exp3 = ConversationExperience(
        messages=[
            {"role": "user", "content": "Explain machine learning"},
        ],
        response="Machine learning is a branch of AI it uses algorithms to learn from data and then make predictions...",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CRITIQUE,
            critique="The formatting is messy and hard to read",
            score=0.4,
        ),
    )
    failures.append(exp3)

    return failures


def create_mock_adapter():
    """Create a mock adapter for demonstration purposes."""
    class MockAdapter(BaseModelAdapter):
        def __init__(self):
            super().__init__(model_name="mock-model")

        def generate(self, prompt, context=None, **kwargs):
            return "This is a mock response."

        def _call_api(self, messages, system=None, temperature=0.7, **kwargs):
            return "Mock API response"

        def _count_tokens_impl(self, text):
            return len(text.split())

        def compute_gradient(self, prompt, failures, target=None, **kwargs):
            """Mock gradient computation."""
            # Analyze failures and generate improvement guidance.
            critiques = [f.feedback.critique for f in failures if f.feedback.critique]

            gradient_text = "Failure analysis found:\n"
            gradient_text += "1) Responses are too long and verbose\n"
            gradient_text += "2) Structured formatting is missing\n"
            gradient_text += "3) The user's question is not answered directly\n"
            gradient_text += "\nSuggestion: simplify the answer, use a clear structure, and provide the answer directly."

            return SimpleGradient(
                text=gradient_text,
                metadata={"num_failures": len(failures)}
            )

        def apply_gradient(self, prompt, gradient, conservative=False, **kwargs):
            """Mock gradient application."""
            # Create a new prompt version.
            new_prompt = prompt.bump_version()

            if conservative:
                # Conservative update: add a small instruction.
                new_prompt.content = f"{prompt.content}\n\nNote: answer directly and concisely, with a clear structure."
            else:
                # Aggressive update: rewrite completely.
                new_prompt.content = (
                    "You are a concise and clear assistant.\n\n"
                    "Answering principles:\n"
                    "1. Answer the user's question directly without rambling\n"
                    "2. Use clear structure and formatting\n"
                    "3. Provide practical examples and code when useful\n"
                    "4. Avoid over-explaining\n\n"
                    "Keep answers concise and effective."
                )

            return new_prompt

    return MockAdapter()


def example_basic_optimization():
    """Example 1: basic optimization flow."""
    print("\n" + "="*80)
    print("Example 1: Basic Optimization Flow")
    print("="*80 + "\n")

    # 1. Create the initial prompt.
    initial_prompt = TextPrompt(
        content="You are a helpful AI assistant.",
        version="v1.0",
    )
    print(f"Initial prompt (v{initial_prompt.version}):")
    print(f"{initial_prompt.content}\n")

    # 2. Create failure experiences.
    failures = create_mock_failures()
    print(f"Collected {len(failures)} failure cases:")
    for i, f in enumerate(failures, 1):
        critique = f.feedback.critique if f.feedback else "No critique"
        print(f"  {i}. {critique}")
    print()

    # 3. Create the adapter and optimizer.
    adapter = create_mock_adapter()
    config = OptimizerConfig(
        max_steps=3,
        conservative=False,
        validate_every_step=False,  # Skip validation for now.
    )
    optimizer = TrainFreeOptimizer(adapter, config)

    # 4. Run optimization.
    print("Starting optimization...\n")
    result = optimizer.optimize(
        prompt=initial_prompt,
        experiences=failures,
        validator=None,  # No validator for this example.
    )

    # 5. Show results.
    print("\n" + "-"*80)
    print("Optimization results:")
    print("-"*80)
    print(f"Steps taken: {result.steps_taken}")
    print(f"Converged: {result.converged}")
    print(f"\nFinal prompt (v{result.optimized_prompt.version}):")
    print(f"{result.optimized_prompt.content}\n")

    # Show history.
    print("Optimization history:")
    for step in result.history:
        print(f"  Step {step.step_num}: v{step.old_prompt.version} → v{step.new_prompt.version}")
        print(f"    Failures: {step.num_failures}")
        print(f"    Gradient summary: {step.gradient[:60]}...")
        print()


def example_with_validation():
    """Example 2: optimization with validation."""
    print("\n" + "="*80)
    print("Example 2: Optimization with Validation")
    print("="*80 + "\n")

    # Create the initial prompt.
    initial_prompt = TextPrompt(
        content="Answer the user's question.",
        version="v1.0",
    )
    print(f"Initial prompt: {initial_prompt.content}\n")

    # Create failure experiences.
    failures = create_mock_failures()

    # Create a mock validator function.
    def mock_validator(prompt):
        """Mock validator that checks prompt quality."""
        # Assume better prompts:
        # 1. include more guidance principles
        # 2. have a clearer structure

        score = 0.0

        # Check length: more detailed prompts score higher.
        score += min(len(prompt.content) / 300, 0.4)

        # Check for key guidance words.
        keywords = ["direct", "concise", "structure", "clear", "principles"]
        for kw in keywords:
            if kw in prompt.content:
                score += 0.12

        score = min(score, 1.0)  # Clamp to 1.0.

        print(f"  ✓ Validation score: {score:.3f} (length={len(prompt.content)}, keywords={sum(1 for kw in keywords if kw in prompt.content)})")
        return score

    # Create the optimizer.
    adapter = create_mock_adapter()
    config = OptimizerConfig(
        max_steps=3,
        conservative=False,
        validate_every_step=True,
        early_stopping_patience=2,
    )
    optimizer = TrainFreeOptimizer(adapter, config)

    # Run optimization.
    print("Starting optimization (with validation)...\n")
    result = optimizer.optimize(
        prompt=initial_prompt,
        experiences=failures,
        validator=mock_validator,
    )

    # Show results.
    print("\n" + "-"*80)
    print("Optimization results:")
    print("-"*80)
    print(f"Initial score: {result.final_score - result.improvement:.3f}")
    print(f"Final score: {result.final_score:.3f}")
    print(f"Total improvement: {result.improvement:+.3f}")
    print(f"\nFinal prompt:\n{result.optimized_prompt.content}\n")


def example_strategies():
    """Example 3: compare optimization strategies."""
    print("\n" + "="*80)
    print("Example 3: Conservative vs Aggressive Strategy")
    print("="*80 + "\n")

    initial_prompt = TextPrompt(
        content="Help the user.",
        version="v1.0",
    )
    failures = create_mock_failures()

    # Conservative strategy.
    print("[Conservative Strategy]")
    print("-"*80)
    adapter_conservative = create_mock_adapter()
    config_conservative = OptimizerConfig(
        max_steps=2,
        conservative=True,  # Conservative mode.
    )
    optimizer_c = TrainFreeOptimizer(adapter_conservative, config_conservative)
    result_c = optimizer_c.optimize(initial_prompt, failures, validator=None)

    print(f"\nFinal prompt (conservative):")
    print(f"{result_c.optimized_prompt.content}\n")

    # Aggressive strategy.
    print("\n[Aggressive Strategy]")
    print("-"*80)
    adapter_aggressive = create_mock_adapter()
    config_aggressive = OptimizerConfig(
        max_steps=2,
        conservative=False,  # Aggressive mode.
    )
    optimizer_a = TrainFreeOptimizer(adapter_aggressive, config_aggressive)
    result_a = optimizer_a.optimize(initial_prompt, failures, validator=None)

    print(f"\nFinal prompt (aggressive):")
    print(f"{result_a.optimized_prompt.content}\n")

    # Comparison.
    print("\n[Comparison]")
    print("-"*80)
    print(f"Conservative strategy: prompt length {len(result_c.optimized_prompt.content)} characters")
    print(f"Aggressive strategy: prompt length {len(result_a.optimized_prompt.content)} characters")
    print("\nConclusion:")
    print("- Conservative strategy: small edits that preserve the existing structure")
    print("- Aggressive strategy: major rewrite that introduces a new structure")


def main():
    """Run all examples."""
    print("\n" + "🔍 " * 20)
    print("TrainFreeOptimizer Complete Example")
    print("🔍 " * 20)

    # Run examples.
    example_basic_optimization()
    print("\n" + "="*80 + "\n")

    example_with_validation()
    print("\n" + "="*80 + "\n")

    example_strategies()

    # Summary.
    print("\n" + "="*80)
    print("✅ Example run completed!")
    print("="*80)
    print("\nKey takeaways:")
    print("1. TrainFreeOptimizer uses failure cases to improve prompts")
    print("2. Validators can be used to evaluate optimization quality")
    print("3. Conservative/aggressive strategies control update magnitude")
    print("4. Early stopping helps prevent over-optimization")
    print("\nNext steps:")
    print("- Use a real API adapter (OpenAI/Anthropic)")
    print("- Collect real failure cases")
    print("- Design suitable validation metrics")
    print("- Deploy the optimizer in production")


if __name__ == "__main__":
    main()
