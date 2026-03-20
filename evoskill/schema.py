"""Unified Data Schema — the source of truth for all I/O types.

Supports OpenAI's "Content Parts" format (Text + Image) from day one,
ensuring multimodal compatibility across the entire framework.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content Parts — Discriminated Union for Multimodality
# ---------------------------------------------------------------------------

class TextContent(BaseModel):
    """A plain-text content part."""

    type: Literal["text"] = "text"
    text: str


class ImageURL(BaseModel):
    """Inner payload for an image content part."""

    url: str  # Regular URL or base64 data-URL


class ImageContent(BaseModel):
    """An image_url content part (for vision models)."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


# Discriminated union keyed on `type`
ContentPart = Union[TextContent, ImageContent]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single chat message, compatible with the OpenAI Chat API.

    `content` may be a simple string *or* a list of ContentPart objects
    to support multimodal conversations (text + images).
    """

    role: Literal["system", "user", "assistant", "function"]
    content: Union[str, List[ContentPart]]

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize to the dict format expected by the OpenAI Python SDK."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {
            "role": self.role,
            "content": [part.model_dump() for part in self.content],
        }


# ---------------------------------------------------------------------------
# Skill — a versioned system-prompt container
# ---------------------------------------------------------------------------

class Skill(BaseModel):
    """A Skill following the Agent Skills standard (https://agentskills.io).

    On disk, a Skill is a directory containing a ``SKILL.md`` with YAML
    frontmatter (name, description, metadata) and a Markdown body that
    serves as the system prompt.  An optional ``config.yaml`` stores
    few-shot examples and model-level configuration.

    Field mapping::

        SKILL.md frontmatter.name        → name
        SKILL.md frontmatter.description → description
        SKILL.md frontmatter.metadata    → metadata (version, target, …)
        SKILL.md body                    → system_prompt
        config.yaml few_shot_messages    → few_shot_messages
        config.yaml (rest)               → config
    """

    name: str = Field(
        ...,
        description="Skill 名称，kebab-case，≤64字符，需与目录名一致。",
    )
    description: str = Field(
        default="",
        description="Skill 描述，≤1024字符。说明 skill 的用途和触发条件。",
    )
    version: str = "v1.0"
    system_prompt: str = Field(
        default="",
        description="SKILL.md Markdown body — 即 LLM 的 system prompt，也是 APO 优化目标。",
    )
    target: Optional[str] = Field(
        default=None,
        description="用户一句话优化方向，如'更像人'、'更简洁'。APO 优化时会参考此目标。",
    )
    few_shot_messages: List[Message] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Skill Meta — metadata for skill-tree directories
# ---------------------------------------------------------------------------

class SkillMeta(BaseModel):
    """Metadata for a skill-tree node (directory-level ``_meta.yaml``).

    Each sub-directory in a skill tree may contain a ``_meta.yaml`` that
    stores the group name, a human-readable description, and the creation
    timestamp.
    """

    name: str
    description: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class Feedback(BaseModel):
    """Human (or auto-judge) feedback on a single interaction."""

    score: float = Field(..., ge=0.0, le=1.0)
    critique: Optional[str] = None
    correction: Optional[str] = None


# ---------------------------------------------------------------------------
# Trace — the atomic unit of storage
# ---------------------------------------------------------------------------

class Trace(BaseModel):
    """An immutable record of one agent interaction.

    Stores the full conversation context (`inputs`), the agent's
    `prediction`, and optional `feedback` used by the APO optimizer.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    inputs: List[Message]
    prediction: Message
    feedback: Optional[Feedback] = None
