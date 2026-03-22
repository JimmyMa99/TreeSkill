"""Textual Gradient implementations.

Textual gradients are the core innovation of TGD (Textual Gradient Descent).
Instead of numerical gradients, we use natural language descriptions
of how to improve a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json


@dataclass
class SimpleGradient:
    """A basic textual gradient with a single text description.

    This is the most common form of gradient - a natural language
    explanation of why a prompt failed and how to fix it.

    Example
    -------
    >>> grad = SimpleGradient(
    ...     text="The prompt is too formal. Use contractions and casual language.",
    ...     metadata={"source": "user_feedback", "sample_count": 5}
    ... )
    >>> str(grad)
    'The prompt is too formal. Use contractions and casual language.'
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleGradient":
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
        )


@dataclass
class MultimodalGradient:
    """A gradient that includes feedback for multiple modalities.

    For multimodal prompts (text + images), the gradient can contain
    specific feedback for each modality.

    Example
    -------
    >>> grad = MultimodalGradient(
    ...     text_gradient="Improve the text analysis clarity",
    ...     image_gradient="Pay more attention to fine details in images",
    ...     audio_gradient=None,  # No audio feedback
    ... )
    """

    text_gradient: str
    image_gradient: Optional[str] = None
    audio_gradient: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        parts = [self.text_gradient]
        if self.image_gradient:
            parts.append(f"\n[Image feedback]: {self.image_gradient}")
        if self.audio_gradient:
            parts.append(f"\n[Audio feedback]: {self.audio_gradient}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_gradient": self.text_gradient,
            "image_gradient": self.image_gradient,
            "audio_gradient": self.audio_gradient,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalGradient":
        return cls(
            text_gradient=data["text_gradient"],
            image_gradient=data.get("image_gradient"),
            audio_gradient=data.get("audio_gradient"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
        )


class GradientHistory:
    """Maintains a history of gradients with optional momentum.

    Analogous to momentum in SGD, this allows smoothing out noise
    in textual gradients by considering recent history.

    Example
    -------
    >>> history = GradientHistory(alpha=0.9)
    >>> history.add(gradient1)
    >>> history.add(gradient2)
    >>> smoothed = history.get_smoothed_gradient()
    """

    def __init__(self, alpha: float = 0.9, max_history: int = 10):
        """
        Parameters
        ----------
        alpha : float
            Momentum coefficient (0 = no history, 1 = full history).
            Higher alpha means more smoothing.
        max_history : int
            Maximum number of past gradients to keep.
        """
        self.alpha = alpha
        self.max_history = max_history
        self.gradients: List[SimpleGradient] = []

    def add(self, gradient: SimpleGradient) -> None:
        """Add a new gradient to history."""
        self.gradients.append(gradient)
        if len(self.gradients) > self.max_history:
            self.gradients.pop(0)

    def get_smoothed_gradient(self, last_n: int = 3) -> Optional[SimpleGradient]:
        """Get a momentum-smoothed gradient.

        This aggregates the last N gradients into a single coherent description.
        For simplicity, we just concatenate them with weights.

        Parameters
        ----------
        last_n : int
            Number of recent gradients to consider.

        Returns
        -------
        Optional[SimpleGradient]
            A smoothed gradient, or None if history is empty.
        """
        if not self.gradients:
            return None

        recent = self.gradients[-last_n:]

        # For now, just return the most recent gradient
        # A more sophisticated implementation would use an LLM to merge them
        return recent[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "max_history": self.max_history,
            "gradients": [g.to_dict() for g in self.gradients],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradientHistory":
        obj = cls(alpha=data["alpha"], max_history=data["max_history"])
        obj.gradients = [
            SimpleGradient.from_dict(g) for g in data.get("gradients", [])
        ]
        return obj
