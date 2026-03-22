"""Concrete Prompt implementations.

This module provides ready-to-use prompt types:
- TextPrompt: Simple text-based prompts
- MultimodalPrompt: Prompts with text + images/audio
- StructuredPrompt: Prompts with JSON schema constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import base64


def _increment_version(version: str) -> str:
    """Bump a version string like 'v1.0' -> 'v1.1'."""
    if not version.startswith("v"):
        return version + ".1"
    parts = version[1:].split(".")
    if len(parts) == 2 and parts[1].isdigit():
        return f"v{parts[0]}.{int(parts[1]) + 1}"
    return version + ".1"


@dataclass
class TextPrompt:
    """Simple text-based system prompt.

    This is the most common type of prompt - just a text instruction
    that tells the model how to behave.

    Example
    -------
    >>> prompt = TextPrompt(
    ...     content="You are a helpful writing assistant.",
    ...     name="writing-assistant",
    ...     version="v1.0",
    ... )
    """

    content: str
    name: str = "prompt"
    version: str = "v1.0"
    target: Optional[str] = None  # Optimization direction
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_model_input(self) -> str:
        """Return the raw text content."""
        return self.content

    def apply_gradient(self, gradient) -> "TextPrompt":
        """Apply gradient - typically delegated to ModelAdapter.

        This is a placeholder that returns self. In practice,
        ModelAdapter.apply_gradient() will create a new instance.
        """
        return self

    def serialize(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "name": self.name,
            "version": self.version,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "TextPrompt":
        """Reconstruct from dictionary."""
        return cls(
            content=data["content"],
            name=data.get("name", "prompt"),
            version=data.get("version", "v1.0"),
            target=data.get("target"),
            metadata=data.get("metadata", {}),
        )

    def bump_version(self) -> "TextPrompt":
        """Return a copy with incremented version."""
        return TextPrompt(
            content=self.content,
            name=self.name,
            version=_increment_version(self.version),
            target=self.target,
            metadata=self.metadata,
        )


@dataclass
class MultimodalPrompt:
    """Prompt with text and optional images/audio.

    For vision-language models (GPT-4o, Claude 4.5 Sonnet, Gemini).

    Example
    -------
    >>> prompt = MultimodalPrompt(
    ...     text="Analyze this image for defects.",
    ...     images=["/path/to/defect.jpg"],
    ...     name="defect-detector",
    ... )
    """

    text: str
    images: List[Union[str, Path, bytes]] = field(default_factory=list)
    audio: Optional[Union[str, Path, bytes]] = None
    name: str = "multimodal-prompt"
    version: str = "v1.0"
    target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_model_input(self) -> Dict[str, Any]:
        """Convert to OpenAI-style multimodal format.

        Returns
        -------
        Dict[str, Any]
            {
                "text": "...",
                "images": [{"type": "image_url", "image_url": {"url": "data:..."}}],
                "audio": {"type": "audio_url", "audio_url": {"url": "data:..."}}
            }
        """
        result = {"text": self.text}

        if self.images:
            result["images"] = [
                {"type": "image_url", "image_url": {"url": self._encode_media(img)}}
                for img in self.images
            ]

        if self.audio:
            result["audio"] = {
                "type": "audio_url",
                "audio_url": {"url": self._encode_media(self.audio)},
            }

        return result

    def _encode_media(self, media: Union[str, Path, bytes]) -> str:
        """Encode image/audio to base64 data URL."""
        if isinstance(media, bytes):
            # Already raw bytes
            data = media
            media_type = "image/jpeg"  # Default assumption
        else:
            # Load from file
            path = Path(media)
            data = path.read_bytes()
            suffix = path.suffix.lower()

            # Infer media type
            media_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
            }
            media_type = media_types.get(suffix, "application/octet-stream")

        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{media_type};base64,{b64}"

    def apply_gradient(self, gradient) -> "MultimodalPrompt":
        """Placeholder - delegated to ModelAdapter."""
        return self

    def serialize(self) -> Dict[str, Any]:
        """Serialize - images/audio stored as paths or base64."""
        return {
            "text": self.text,
            "images": [
                str(img) if isinstance(img, (str, Path)) else base64.b64encode(img).decode()
                for img in self.images
            ],
            "audio": str(self.audio)
            if isinstance(self.audio, (str, Path))
            else base64.b64encode(self.audio).decode()
            if self.audio
            else None,
            "name": self.name,
            "version": self.version,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "MultimodalPrompt":
        """Reconstruct from dictionary."""
        # Images might be base64 strings or paths
        images = []
        for img in data.get("images", []):
            # Simple heuristic: if it starts with / or ., treat as path
            if isinstance(img, str) and (img.startswith("/") or img.startswith(".")):
                images.append(img)
            else:
                # Assume base64
                images.append(base64.b64decode(img) if isinstance(img, str) else img)

        audio = data.get("audio")
        if isinstance(audio, str) and not (audio.startswith("/") or audio.startswith(".")):
            audio = base64.b64decode(audio)

        return cls(
            text=data["text"],
            images=images,
            audio=audio,
            name=data.get("name", "multimodal-prompt"),
            version=data.get("version", "v1.0"),
            target=data.get("target"),
            metadata=data.get("metadata", {}),
        )

    def bump_version(self) -> "MultimodalPrompt":
        """Return a copy with incremented version."""
        return MultimodalPrompt(
            text=self.text,
            images=self.images.copy(),
            audio=self.audio,
            name=self.name,
            version=_increment_version(self.version),
            target=self.target,
            metadata=self.metadata,
        )


@dataclass
class StructuredPrompt:
    """Prompt with structured output constraints (JSON schema).

    For models that support constrained generation (e.g., GPT-4o with
    response_format, or Claude with tool use).

    Example
    -------
    >>> schema = {
    ...     "type": "object",
    ...     "properties": {
    ...         "name": {"type": "string"},
    ...         "age": {"type": "integer"},
    ...     },
    ...     "required": ["name", "age"],
    ... }
    >>> prompt = StructuredPrompt(
    ...     instruction="Extract person info from the text.",
    ...     json_schema=schema,
    ... )
    """

    instruction: str
    json_schema: Dict[str, Any]
    name: str = "structured-prompt"
    version: str = "v1.0"
    target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_model_input(self) -> Dict[str, Any]:
        """Return instruction + schema."""
        return {
            "instruction": self.instruction,
            "json_schema": self.json_schema,
        }

    def apply_gradient(self, gradient) -> "StructuredPrompt":
        """Placeholder - delegated to ModelAdapter."""
        return self

    def serialize(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "instruction": self.instruction,
            "json_schema": self.json_schema,
            "name": self.name,
            "version": self.version,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "StructuredPrompt":
        """Reconstruct from dictionary."""
        return cls(
            instruction=data["instruction"],
            json_schema=data["json_schema"],
            name=data.get("name", "structured-prompt"),
            version=data.get("version", "v1.0"),
            target=data.get("target"),
            metadata=data.get("metadata", {}),
        )

    def bump_version(self) -> "StructuredPrompt":
        """Return a copy with incremented version."""
        return StructuredPrompt(
            instruction=self.instruction,
            json_schema=self.json_schema.copy(),
            name=self.name,
            version=_increment_version(self.version),
            target=self.target,
            metadata=self.metadata,
        )
