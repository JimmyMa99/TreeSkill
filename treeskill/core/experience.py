"""Experience and Feedback implementations.

An Experience represents a single interaction with the model,
including the input, output, and optional feedback.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class FeedbackType(Enum):
    """Type of feedback provided."""

    SCORE = "score"  # Numeric score (0-1)
    CRITIQUE = "critique"  # Textual criticism
    CORRECTION = "correction"  # Ideal response example
    BINARY = "binary"  # Good/bad (thumb up/down)


@dataclass
class CompositeFeedback:
    """Feedback that can include score, critique, and correction.

    This is the most flexible feedback type, supporting all forms of feedback.
    At least one field must be non-None.

    Example
    -------
    >>> fb = CompositeFeedback(
    ...     score=0.3,
    ...     critique="Too formal, lacks emotion",
    ...     correction="Here's a better version: ...",
    ... )
    """

    score: Optional[float] = None
    critique: Optional[str] = None
    correction: Optional[str] = None
    feedback_type: FeedbackType = FeedbackType.CRITIQUE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate: at least one field must be set
        if all(v is None for v in [self.score, self.critique, self.correction]):
            raise ValueError("At least one feedback field must be provided")

        # Determine type based on what's provided
        if self.correction is not None:
            self.feedback_type = FeedbackType.CORRECTION
        elif self.critique is not None:
            self.feedback_type = FeedbackType.CRITIQUE
        elif self.score is not None:
            self.feedback_type = FeedbackType.SCORE

    def to_score(self) -> float:
        """Convert to scalar score in [0, 1].

        If score is provided, use it directly.
        If only critique/correction, assume negative feedback (score=0.3).
        """
        if self.score is not None:
            return self.score

        # Heuristic: presence of critique/correction implies negative feedback
        # In practice, you might use an LLM to score the critique
        return 0.3 if self.critique or self.correction else 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "critique": self.critique,
            "correction": self.correction,
            "feedback_type": self.feedback_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompositeFeedback":
        return cls(
            score=data.get("score"),
            critique=data.get("critique"),
            correction=data.get("correction"),
            feedback_type=FeedbackType(data.get("feedback_type", "critique")),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_negative(self) -> bool:
        """Return True if this indicates failure (score < 0.5)."""
        return self.to_score() < 0.5


# Alias for convenience
Feedback = CompositeFeedback


@dataclass
class ConversationExperience:
    """Experience from a text-based conversation.

    This stores the conversation context (list of messages) and
    the assistant's response, plus optional feedback.

    Example
    -------
    >>> exp = ConversationExperience(
    ...     messages=[
    ...         {"role": "user", "content": "Write a poem"},
    ...     ],
    ...     response="Roses are red...",
    ... )
    """

    messages: List[Dict[str, str]]  # OpenAI-style messages
    response: str
    feedback: Optional[Feedback] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_input(self) -> List[Dict[str, str]]:
        return self.messages

    def get_output(self) -> str:
        return self.response

    def get_feedback(self) -> Optional[Feedback]:
        return self.feedback

    def attach_feedback(self, feedback: Feedback) -> "ConversationExperience":
        """Return a copy with feedback attached."""
        return ConversationExperience(
            messages=self.messages,
            response=self.response,
            feedback=feedback,
            id=self.id,  # Keep same ID
            timestamp=self.timestamp,
            metadata=self.metadata,
        )

    def to_training_sample(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "messages": self.messages,
            "response": self.response,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @property
    def is_failure(self) -> bool:
        return self.feedback is not None and self.feedback.is_negative

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationExperience":
        return cls(
            messages=data["messages"],
            response=data["response"],
            feedback=Feedback.from_dict(data["feedback"])
            if data.get("feedback")
            else None,
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MultimodalExperience:
    """Experience with multimodal input/output.

    Supports text, images, audio, video.

    Example
    -------
    >>> exp = MultimodalExperience(
    ...     input_text="What's in this image?",
    ...     input_images=["/path/to/image.jpg"],
    ...     output_text="A cat sitting on a couch",
    ... )
    """

    input_text: str
    output_text: str
    input_images: List[Union[str, Path, bytes]] = field(default_factory=list)
    input_audio: Optional[Union[str, Path, bytes]] = None
    output_images: List[Union[str, Path, bytes]] = field(default_factory=list)
    feedback: Optional[Feedback] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_input(self) -> Dict[str, Any]:
        return {
            "text": self.input_text,
            "images": self.input_images,
            "audio": self.input_audio,
        }

    def get_output(self) -> Dict[str, Any]:
        return {
            "text": self.output_text,
            "images": self.output_images,
        }

    def get_feedback(self) -> Optional[Feedback]:
        return self.feedback

    def attach_feedback(self, feedback: Feedback) -> "MultimodalExperience":
        return MultimodalExperience(
            input_text=self.input_text,
            output_text=self.output_text,
            input_images=self.input_images,
            input_audio=self.input_audio,
            output_images=self.output_images,
            feedback=feedback,
            id=self.id,
            timestamp=self.timestamp,
            metadata=self.metadata,
        )

    def to_training_sample(self) -> Dict[str, Any]:
        # For multimodal, we might store images as base64 or file references
        return {
            "id": self.id,
            "input": self.get_input(),
            "output": self.get_output(),
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @property
    def is_failure(self) -> bool:
        return self.feedback is not None and self.feedback.is_negative

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalExperience":
        input_data = data["input"]
        output_data = data["output"]
        return cls(
            input_text=input_data["text"],
            output_text=output_data["text"],
            input_images=input_data.get("images", []),
            input_audio=input_data.get("audio"),
            output_images=output_data.get("images", []),
            feedback=Feedback.from_dict(data["feedback"])
            if data.get("feedback")
            else None,
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
        )


# Type alias
Experience = Union[ConversationExperience, MultimodalExperience]
