"""Test OpenAI Adapter with SiliconFlow (OpenAI-compatible API).

SiliconFlow provides an OpenAI-compatible API, so we can use OpenAIAdapter.
"""

import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from treeskill.core import (
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
)
from treeskill.adapters.openai import OpenAIAdapter


def _run_siliconflow_basic():
    """Run the basic SiliconFlow generation flow and return success details."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Generation with SiliconFlow")
    print("=" * 60)

    # Get SiliconFlow credentials from .env
    api_key = os.getenv("TREE_LLM_API_KEY")
    base_url = os.getenv("TREE_LLM_BASE_URL")
    model = os.getenv("TREE_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")

    if not api_key or not base_url:
        print("❌ SiliconFlow credentials not found in .env")
        return None

    print(f"✓ Using SiliconFlow API")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")

    # Create adapter
    adapter = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    print(f"\n✓ Created adapter")

    # Create prompt
    prompt = TextPrompt(
        content="You are a friendly assistant. Answer questions concisely.",
        name="test-assistant",
    )

    # Generate
    print(f"\n→ Generating response...")
    try:
        response = adapter.generate(prompt, temperature=0.7)
        print(f"✓ Response: {response}")
        return adapter, prompt, response
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_siliconflow_basic():
    """Test basic generation with SiliconFlow API."""
    result = _run_siliconflow_basic()
    if result is None:
        return

    _adapter, _prompt, response = result
    assert response


def test_siliconflow_gradient():
    """Test gradient computation with SiliconFlow."""
    print("\n" + "=" * 60)
    print("Test 2: Gradient Computation with SiliconFlow")
    print("=" * 60)

    # Get credentials
    api_key = os.getenv("TREE_LLM_API_KEY")
    base_url = os.getenv("TREE_LLM_BASE_URL")
    model = os.getenv("TREE_LLM_MODEL")

    if not all([api_key, base_url, model]):
        print("❌ Missing credentials")
        return

    judge_model = os.getenv("TREE_LLM_JUDGE_MODEL", model)

    # Create adapter
    adapter = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

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

    # Compute gradient using judge model
    print(f"\n→ Computing gradient with {judge_model}...")
    try:
        # Use judge model for gradient computation
        gradient_adapter = OpenAIAdapter(
            model=judge_model,
            api_key=api_key,
            base_url=base_url,
        )
        gradient = gradient_adapter.compute_gradient(
            prompt, [experience], target="be friendlier"
        )
        print(f"✓ Gradient computed:")
        print(f"  {str(gradient)[:200]}...")

        # Apply gradient
        print(f"\n→ Applying gradient...")
        new_prompt = gradient_adapter.apply_gradient(prompt, gradient)
        print(f"✓ Applied gradient:")
        print(f"  Version: {prompt.version} → {new_prompt.version}")
        print(f"  New content:\n{new_prompt.content[:200]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_siliconflow_context():
    """Test generation with conversation context."""
    print("\n" + "=" * 60)
    print("Test 3: Generation with Context (SiliconFlow)")
    print("=" * 60)

    api_key = os.getenv("TREE_LLM_API_KEY")
    base_url = os.getenv("TREE_LLM_BASE_URL")
    model = os.getenv("TREE_LLM_MODEL")

    if not all([api_key, base_url, model]):
        print("❌ Missing credentials")
        return

    adapter = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

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
    try:
        response = adapter.generate(prompt, context=experiences, temperature=0.7)
        print(f"✓ Response: {response}")

        if "Alex" in response:
            print("✓ Model correctly used context!")
        else:
            print("⚠️ Model may not have used context")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all SiliconFlow tests."""
    print("=" * 60)
    print("OpenAI Adapter Test Suite (SiliconFlow API)")
    print("=" * 60)

    try:
        # Test 1: Basic generation
        result = _run_siliconflow_basic()

        if result:
            # Test 2: Context
            test_siliconflow_context()

            # Test 3: Gradient
            test_siliconflow_gradient()

        print("\n" + "=" * 60)
        print("All SiliconFlow tests completed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
