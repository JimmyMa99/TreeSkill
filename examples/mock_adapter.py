"""Mock adapter for testing and demonstration.

This adapter doesn't call any real API, but demonstrates how to
implement the BaseModelAdapter interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evoskill.core.base_adapter import BaseModelAdapter
from evoskill.core.abc import OptimizablePrompt, Experience


class MockAdapter(BaseModelAdapter):
    """A mock adapter for testing purposes.

    This adapter simulates responses without calling any API.
    Useful for unit tests and demonstrations.
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="mock-model", **kwargs)
        self.call_count = 0
        self.last_messages = None

    @property
    def supports_vision(self) -> bool:
        return True  # Mock supports everything

    @property
    def max_context_tokens(self) -> int:
        return 100000

    def generate(
        self,
        prompt: OptimizablePrompt,
        context: Optional[List[Experience]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a mock response."""
        prompt_text = self._extract_prompt_text(prompt)

        # Simulate different responses based on prompt content
        if "writing" in prompt_text.lower() or "write" in prompt_text.lower():
            return "This is a mock writing response."
        elif "code" in prompt_text.lower():
            return "def example():\n    return 'mock code'"
        else:
            return f"Mock response to: {prompt_text[:50]}..."

    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Simulate API call."""
        self.call_count += 1
        self.last_messages = messages

        messages_str = str(messages)

        # Simulate gradient computation (analysis)
        if "analysis" in messages_str.lower() and "failures" in messages_str.lower():
            return (
                "Analysis shows that the current prompt has the following issues:\n"
                "1. The instructions are not specific enough\n"
                "2. Examples are missing\n"
                "3. The tone is too formal\n\n"
                "Consider improving those areas."
            )

        # Simulate prompt rewrite
        if "rewrite" in messages_str.lower():
            return (
                "You are a professional and friendly assistant.\n\n"
                "Your tasks are:\n"
                "1. Understand the user's needs\n"
                "2. Provide clear and accurate answers\n"
                "3. Maintain a friendly and professional tone\n\n"
                "Remember: concise beats verbose."
            )

        # Default response
        return "Mock API response"

    def _count_tokens_impl(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4


# Example usage
if __name__ == "__main__":
    from evoskill.core import (
        TextPrompt,
        ConversationExperience,
        CompositeFeedback,
    )

    print("=" * 60)
    print("MockAdapter Demo")
    print("=" * 60)

    # 1. Create adapter
    adapter = MockAdapter()
    print(f"\n✓ Created adapter: {adapter.model_name}")
    print(f"  - Supports vision: {adapter.supports_vision}")
    print(f"  - Max tokens: {adapter.max_context_tokens}")

    # 2. Create prompt
    prompt = TextPrompt(
        content="You are a writing assistant.",
        name="writing-assistant",
        version="v1.0",
        target="sound more natural",
    )
    print(f"\n✓ Created prompt: {prompt.name} v{prompt.version}")
    print(f"  - Tokens: {adapter.count_tokens(prompt)}")

    # 3. Validate prompt
    issues = adapter.validate_prompt(prompt)
    print(f"\n✓ Validation: {len(issues)} issues")
    if issues:
        for issue in issues:
            print(f"  - {issue}")

    # 4. Generate response
    response = adapter.generate(prompt)
    print(f"\n✓ Generated response:")
    print(f"  {response}")

    # 5. Create experience with feedback
    experience = ConversationExperience(
        messages=[{"role": "user", "content": "Write a poem"}],
        response="A spring morning slips by unnoticed...",
        feedback=CompositeFeedback(score=0.3, critique="Too clichéd"),
    )
    print(f"\n✓ Created experience (failure={experience.is_failure})")

    # 6. Compute gradient
    failures = [experience]
    gradient = adapter.compute_gradient(prompt, failures, target="be more original")
    print(f"\n✓ Computed gradient:")
    print(f"  {str(gradient)[:100]}...")

    # 7. Apply gradient
    new_prompt = adapter.apply_gradient(prompt, gradient)
    print(f"\n✓ Applied gradient:")
    print(f"  Version: {prompt.version} → {new_prompt.version}")
    print(f"  New content:\n{new_prompt.content}")

    print("\n" + "=" * 60)
    print(f"Total API calls: {adapter.call_count}")
    print("=" * 60)
