"""Test script to verify core abstraction layer works."""

import sys
import types
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

for missing_module, attrs in {
    "evoskill.script": {
        "ScriptValidator": object,
        "ScriptValidationResult": object,
        "ScriptIssue": object,
        "validate_script": lambda *args, **kwargs: None,
        "validate_script_file": lambda *args, **kwargs: None,
        "load_script": lambda *args, **kwargs: None,
        "save_script": lambda *args, **kwargs: None,
        "load_script_as_tools": lambda *args, **kwargs: {},
    },
    "evoskill.memory": {
        "MEMORY_FILE": "memory.json",
        "MemoryType": object,
        "MemoryEntry": object,
        "MemoryStore": object,
        "MemoryCompiler": object,
    },
    "evoskill.agenda": {
        "AgendaManager": object,
        "compile_agenda_context": lambda *args, **kwargs: "",
        "parse_due": lambda *args, **kwargs: None,
    },
}.items():
    module = types.ModuleType(missing_module)
    for name, value in attrs.items():
        setattr(module, name, value)
    sys.modules.setdefault(missing_module, module)

from evoskill.core import (
    TextPrompt,
    MultimodalPrompt,
    SimpleGradient,
    CompositeFeedback,
    ConversationExperience,
    FeedbackType,
)
from evoskill.schema import AudioContent, AudioURL, Message, TextContent


def test_text_prompt():
    """Test TextPrompt creation and serialization."""
    print("Testing TextPrompt...")

    prompt = TextPrompt(
        content="You are a helpful assistant.",
        name="test-prompt",
        version="v1.0",
        target="more concise",
    )

    # Test to_model_input
    assert prompt.to_model_input() == "You are a helpful assistant."

    # Test serialization
    data = prompt.serialize()
    assert data["content"] == "You are a helpful assistant."
    assert data["version"] == "v1.0"

    # Test deserialization
    prompt2 = TextPrompt.deserialize(data)
    assert prompt2.content == prompt.content
    assert prompt2.version == prompt.version

    # Test bump_version
    prompt3 = prompt.bump_version()
    assert prompt3.version == "v1.1"
    assert prompt3.content == prompt.content

    print("✓ TextPrompt tests passed")


def test_multimodal_prompt():
    """Test MultimodalPrompt."""
    print("\nTesting MultimodalPrompt...")

    # Create with image path (won't actually load it in this test)
    prompt = MultimodalPrompt(
        text="Analyze this image",
        images=[],  # Empty for now
        name="image-analyzer",
    )

    # Test to_model_input
    model_input = prompt.to_model_input()
    assert model_input["text"] == "Analyze this image"
    assert "images" not in model_input  # Empty list shouldn't be included

    # Test serialization
    data = prompt.serialize()
    assert data["text"] == "Analyze this image"

    print("✓ MultimodalPrompt tests passed")


def test_message_audio_content_part():
    """Test audio content parts serialize to OpenAI-compatible format."""
    print("\nTesting audio content part...")

    msg = Message(
        role="user",
        content=[
            AudioContent(
                audio_url=AudioURL(url="data:audio/wav;base64,ZmFrZQ==")
            ),
            TextContent(text="Please summarize this audio clip"),
        ],
    )

    data = msg.to_api_dict()
    assert data["role"] == "user"
    assert data["content"][0]["type"] == "audio_url"
    assert data["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert data["content"][1]["type"] == "text"

    print("✓ Audio content part tests passed")


def test_gradient():
    """Test SimpleGradient."""
    print("\nTesting SimpleGradient...")

    grad = SimpleGradient(
        text="The prompt is too formal. Make it more casual.",
        metadata={"source": "user_feedback"},
    )

    # Test __str__
    assert str(grad) == "The prompt is too formal. Make it more casual."

    # Test serialization
    data = grad.to_dict()
    assert data["text"] == grad.text

    # Test deserialization
    grad2 = SimpleGradient.from_dict(data)
    assert grad2.text == grad.text

    print("✓ SimpleGradient tests passed")


def test_feedback():
    """Test CompositeFeedback."""
    print("\nTesting CompositeFeedback...")

    # Test with score
    fb1 = CompositeFeedback(score=0.3)
    assert fb1.to_score() == 0.3
    assert fb1.is_negative == True
    assert fb1.feedback_type == FeedbackType.SCORE

    # Test with critique
    fb2 = CompositeFeedback(critique="Too formal")
    assert fb2.to_score() == 0.3  # Default for critique
    assert fb2.is_negative == True
    assert fb2.feedback_type == FeedbackType.CRITIQUE

    # Test with correction
    fb3 = CompositeFeedback(correction="Here's a better version...")
    assert fb3.feedback_type == FeedbackType.CORRECTION
    assert fb3.to_score() == 0.3

    # Test combined
    fb4 = CompositeFeedback(
        score=0.2, critique="Too formal", correction="Better: ..."
    )
    assert fb4.to_score() == 0.2  # Score takes precedence

    # Test serialization
    data = fb4.to_dict()
    fb5 = CompositeFeedback.from_dict(data)
    assert fb5.score == fb4.score
    assert fb5.critique == fb4.critique

    print("✓ CompositeFeedback tests passed")


def test_experience():
    """Test ConversationExperience."""
    print("\nTesting ConversationExperience...")

    exp = ConversationExperience(
        messages=[{"role": "user", "content": "Hello"}],
        response="Hi there!",
    )

    # Test basic properties
    assert exp.get_input() == [{"role": "user", "content": "Hello"}]
    assert exp.get_output() == "Hi there!"
    assert exp.get_feedback() is None
    assert exp.is_failure == False

    # Attach feedback
    fb = CompositeFeedback(score=0.3, critique="Not friendly enough")
    exp_with_fb = exp.attach_feedback(fb)

    assert exp_with_fb.feedback is not None
    assert exp_with_fb.is_failure == True
    assert exp_with_fb.id == exp.id  # Same ID

    # Test serialization
    data = exp_with_fb.to_training_sample()
    assert data["feedback"]["score"] == 0.3

    print("✓ ConversationExperience tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Core Abstraction Layer Tests")
    print("=" * 60)

    try:
        test_text_prompt()
        test_multimodal_prompt()
        test_gradient()
        test_feedback()
        test_experience()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
