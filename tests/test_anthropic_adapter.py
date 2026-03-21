"""Test Anthropic Adapter.

This script tests the AnthropicAdapter with real API calls.
You need to set ANTHROPIC_API_KEY environment variable for full tests.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evoskill.core import (
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    MultimodalPrompt,
)
from evoskill.adapters.anthropic import (
    AnthropicAdapter,
    create_claude_35_sonnet,
    create_claude_35_haiku,
)


def test_basic_generation():
    """Test basic text generation."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Text Generation")
    print("=" * 60)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set. Skipping real API test.")
        return None

    # Create adapter
    adapter = create_claude_35_haiku()  # Use cheaper/faster model for testing
    print(f"✓ Created adapter: {adapter.model_name}")
    print(f"  - Vision support: {adapter.supports_vision}")
    print(f"  - Max context: {adapter.max_context_tokens:,} tokens")

    # Create prompt
    prompt = TextPrompt(
        content="You are a friendly assistant. Answer questions concisely.",
        name="test-assistant",
    )
    print(f"\n✓ Created prompt")

    # Validate
    issues = adapter.validate_prompt(prompt)
    if issues:
        print(f"⚠️ Validation issues: {issues}")
    else:
        print(f"✓ Prompt validation passed")

    # Generate
    print(f"\n→ Generating response...")
    try:
        response = adapter.generate(prompt, temperature=0.7, max_tokens=100)
        print(f"✓ Response: {response[:100]}...")
        return adapter, prompt, response
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_context():
    """Test generation with conversation context."""
    print("\n" + "=" * 60)
    print("Test 2: Generation with Context")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Skipping (no API key)")
        return

    adapter = create_claude_35_haiku()
    prompt = TextPrompt(
        content="You are an assistant. Remember the earlier conversation.",
        name="context-test",
    )

    # Create conversation context
    experiences = [
        ConversationExperience(
            messages=[{"role": "user", "content": "My name is Sam."}],
            response="Hi, Sam! Nice to meet you.",
        ),
    ]

    print(f"✓ Created context with {len(experiences)} turns")

    # Generate with context
    print(f"\n→ Generating response with context...")
    try:
        response = adapter.generate(prompt, context=experiences, temperature=0.7, max_tokens=100)
        print(f"✓ Response: {response}")

        # The model should remember the name
        if "Sam" in response:
            print("✓ Model correctly used context!")
        else:
            print("⚠️ Model may not have used context")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_gradient_computation():
    """Test gradient computation (requires API)."""
    print("\n" + "=" * 60)
    print("Test 3: Gradient Computation (Advanced)")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Skipping (no API key)")
        return

    adapter = create_claude_35_sonnet()  # Use Sonnet for better gradient computation

    # Create prompt
    prompt = TextPrompt(
        content="You are an assistant.",
        name="simple-assistant",
        target="be friendlier",
    )

    # Create failure experience
    experience = ConversationExperience(
        messages=[{"role": "user", "content": "Hello"}],
        response="Hello.",
        feedback=CompositeFeedback(
            score=0.3,
            critique="Too cold and not friendly enough"
        ),
    )

    print(f"✓ Created prompt and failure experience")

    # Compute gradient
    print(f"\n→ Computing gradient...")
    try:
        gradient = adapter.compute_gradient(prompt, [experience], target="be friendlier")
        print(f"✓ Gradient computed:")
        print(f"  {str(gradient)[:200]}...")

        # Apply gradient
        print(f"\n→ Applying gradient...")
        new_prompt = adapter.apply_gradient(prompt, gradient, conservative=True)
        print(f"✓ Applied gradient:")
        print(f"  Version: {prompt.version} → {new_prompt.version}")
        print(f"  New content:\n{new_prompt.content[:200]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_token_counting():
    """Test token counting."""
    print("\n" + "=" * 60)
    print("Test 4: Token Counting")
    print("=" * 60)

    adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")

    # Test text token counting
    text = "Hello, this is a test string."
    tokens = adapter.count_tokens(TextPrompt(content=text))
    print(f"✓ Text: '{text}'")
    print(f"  Tokens (approx): {tokens}")

    # Test message token counting
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]
    msg_tokens = adapter.count_messages_tokens(messages)
    print(f"\n✓ Messages: {len(messages)} messages")
    print(f"  Total tokens (approx): {msg_tokens}")


def test_model_variants():
    """Test different model variants."""
    print("\n" + "=" * 60)
    print("Test 5: Model Variants")
    print("=" * 60)

    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    for model_name in models:
        try:
            adapter = AnthropicAdapter(model=model_name)
            print(f"\n✓ {model_name}")
            print(f"  - Vision: {adapter.supports_vision}")
            print(f"  - Max context: {adapter.max_context_tokens:,}")
        except Exception as e:
            print(f"❌ {model_name}: {e}")


def test_vision_capability():
    """Test vision capability check."""
    print("\n" + "=" * 60)
    print("Test 6: Vision Capability")
    print("=" * 60)

    adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")

    # Create multimodal prompt
    prompt = MultimodalPrompt(
        text="Analyze this image",
        images=["example.jpg"],  # Placeholder
    )

    # Validate
    issues = adapter.validate_prompt(prompt)
    if issues:
        print(f"⚠️ Issues: {issues}")
    else:
        print(f"✓ Vision capability supported")

    print(f"  - Model: {adapter.model_name}")
    print(f"  - Supports vision: {adapter.supports_vision}")


def test_factory_functions():
    """Test factory functions."""
    print("\n" + "=" * 60)
    print("Test 7: Factory Functions")
    print("=" * 60)

    try:
        sonnet = create_claude_35_sonnet()
        print(f"✓ Created Claude 3.5 Sonnet: {sonnet.model_name}")

        haiku = create_claude_35_haiku()
        print(f"✓ Created Claude 3.5 Haiku: {haiku.model_name}")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Anthropic Adapter Test Suite")
    print("=" * 60)

    # Check for API key
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not has_api_key:
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not set")
        print("   Tests will skip actual API calls")
        print("   Set the environment variable to run full tests:")
        print("   export ANTHROPIC_API_KEY='your-key-here'")

    try:
        # Test 1: Basic generation (requires API key)
        result = test_basic_generation()

        if result and has_api_key:
            # Test 2: Context (requires API key)
            test_with_context()

            # Test 3: Gradient (requires API key)
            test_gradient_computation()

        # Test 4: Token counting (no API needed)
        test_token_counting()

        # Test 5: Model variants (no API needed)
        test_model_variants()

        # Test 6: Vision capability (no API needed)
        test_vision_capability()

        # Test 7: Factory functions (no API needed)
        test_factory_functions()

        print("\n" + "=" * 60)
        print("All tests completed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
