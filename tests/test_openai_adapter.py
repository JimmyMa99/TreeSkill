"""Test OpenAI Adapter.

This script tests the OpenAIAdapter with real API calls.
You need to set OPENAI_API_KEY environment variable.
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
)
from evoskill.adapters.openai import OpenAIAdapter


def test_basic_generation():
    """Test basic text generation."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Text Generation")
    print("=" * 60)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping real API test.")
        return None

    # Create adapter
    adapter = OpenAIAdapter(model="gpt-4o-mini")  # Use cheaper model for testing
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
    response = adapter.generate(prompt, temperature=0.7)
    print(f"✓ Response: {response[:100]}...")

    return adapter, prompt, response


def test_with_context():
    """Test generation with conversation context."""
    print("\n" + "=" * 60)
    print("Test 2: Generation with Context")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Skipping (no API key)")
        return

    adapter = OpenAIAdapter(model="gpt-4o-mini")
    prompt = TextPrompt(
        content="You are an assistant. Remember the earlier conversation.",
        name="context-test",
    )

    # Create conversation context
    experiences = [
        ConversationExperience(
            messages=[{"role": "user", "content": "My name is Alex."}],
            response="Hi, Alex! Nice to meet you.",
        ),
    ]

    print(f"✓ Created context with {len(experiences)} turns")

    # Generate with context
    print(f"\n→ Generating response with context...")
    response = adapter.generate(prompt, context=experiences, temperature=0.7)
    print(f"✓ Response: {response}")

    # The model should remember the name
    if "Alex" in response:
        print("✓ Model correctly used context!")
    else:
        print("⚠️ Model may not have used context")


def test_token_counting():
    """Test token counting accuracy."""
    print("\n" + "=" * 60)
    print("Test 3: Token Counting")
    print("=" * 60)

    adapter = OpenAIAdapter(model="gpt-4o-mini")

    # Test text token counting
    text = "Hello, this is a test string."
    tokens = adapter.count_tokens(TextPrompt(content=text))
    print(f"✓ Text: '{text}'")
    print(f"  Tokens: {tokens}")

    # Test message token counting
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Hello"},
    ]
    msg_tokens = adapter.count_messages_tokens(messages)
    print(f"\n✓ Messages: {len(messages)} messages")
    print(f"  Total tokens (including formatting): {msg_tokens}")


def test_gradient_computation():
    """Test gradient computation (requires API)."""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Computation (Advanced)")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Skipping (no API key)")
        return

    adapter = OpenAIAdapter(model="gpt-4o-mini")

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
    gradient = adapter.compute_gradient(prompt, [experience], target="be friendlier")
    print(f"✓ Gradient computed:")
    print(f"  {str(gradient)[:200]}...")

    # Apply gradient
    print(f"\n→ Applying gradient...")
    new_prompt = adapter.apply_gradient(prompt, gradient, conservative=True)
    print(f"✓ Applied gradient:")
    print(f"  Version: {prompt.version} → {new_prompt.version}")
    print(f"  New content:\n{new_prompt.content[:200]}...")


def test_model_variants():
    """Test different model variants."""
    print("\n" + "=" * 60)
    print("Test 5: Model Variants")
    print("=" * 60)

    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]

    for model_name in models:
        try:
            adapter = OpenAIAdapter(model=model_name)
            print(f"\n✓ {model_name}")
            print(f"  - Vision: {adapter.supports_vision}")
            print(f"  - Max context: {adapter.max_context_tokens:,}")
        except Exception as e:
            print(f"❌ {model_name}: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenAI Adapter Test Suite")
    print("=" * 60)

    # Check for API key
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_api_key:
        print("\n⚠️  WARNING: OPENAI_API_KEY not set")
        print("   Tests will skip actual API calls")
        print("   Set the environment variable to run full tests:")
        print("   export OPENAI_API_KEY='your-key-here'")

    try:
        # Test 1: Basic generation (requires API key)
        result = test_basic_generation()

        if result and has_api_key:
            # Test 2: Context (requires API key)
            test_with_context()

        # Test 3: Token counting (no API needed)
        test_token_counting()

        if has_api_key:
            # Test 4: Gradient (requires API key)
            test_gradient_computation()

        # Test 5: Model variants (no API needed)
        test_model_variants()

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
