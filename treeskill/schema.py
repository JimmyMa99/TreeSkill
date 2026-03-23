"""Unified Data Schema — the source of truth for all I/O types.

Supports OpenAI's "Content Parts" format (Text + Image) from day one,
ensuring multimodal compatibility across the entire framework.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from enum import Enum
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


class AudioURL(BaseModel):
    """Inner payload for an audio content part."""

    url: str  # Regular URL or base64 data-URL


class ImageContent(BaseModel):
    """An image_url content part (for vision models)."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class AudioContent(BaseModel):
    """An audio_url content part (for omni/audio-capable models)."""

    type: Literal["audio_url"] = "audio_url"
    audio_url: AudioURL


# Discriminated union keyed on `type`
ContentPart = Union[TextContent, ImageContent, AudioContent]


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
# Agenda — long-horizon event entries stored inside config.yaml
# ---------------------------------------------------------------------------

class AgendaType(str, Enum):
    """Agenda event type."""
    REMINDER = "reminder"
    RECURRING = "recurring"
    DEADLINE = "deadline"
    MILESTONE = "milestone"


class Recurrence(str, Enum):
    """Recurrence rule."""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class AgendaEntry(BaseModel):
    """A single agenda entry stored in the ``agenda`` list of config.yaml.

    It is loaded automatically with the Skill and injected as context
    when a conversation begins.

    Example config.yaml format::

        agenda:
          - type: recurring
            title: Wedding anniversary
            recurrence: yearly
            month: 3
            day: 14
            origin_year: 2021
          - type: reminder
            title: Meeting reminder
            due: "2026-03-21T15:30:00"
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: AgendaType = Field(..., description="Event type")
    title: str = Field(..., description="Event title")
    description: str = Field(default="", description="Detailed description")
    tags: List[str] = Field(default_factory=list)

    # -- Time --
    due: Optional[datetime] = Field(
        default=None,
        description="Due timestamp (used by reminder/deadline events)",
    )
    month: Optional[int] = Field(default=None, ge=1, le=12)
    day: Optional[int] = Field(default=None, ge=1, le=31)
    weekday: Optional[int] = Field(
        default=None, ge=0, le=6,
        description="Day of week (0=Mon, 6=Sun)",
    )
    time_of_day: Optional[str] = Field(
        default=None, description="Time in HH:MM format",
    )

    # -- Recurrence --
    recurrence: Recurrence = Field(default=Recurrence.ONCE)

    # -- State --
    done: bool = Field(default=False)
    active: bool = Field(default=True)

    # -- Metadata --
    origin_year: Optional[int] = Field(
        default=None,
        description="Origin year (used to compute the Nth anniversary)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def matches_date(self, target: date) -> bool:
        """Return whether this event matches the given date."""
        if not self.active:
            return False
        if self.done and self.recurrence == Recurrence.ONCE:
            return False

        if self.recurrence == Recurrence.ONCE:
            return self.due is not None and self.due.date() == target
        if self.recurrence == Recurrence.DAILY:
            return True
        if self.recurrence == Recurrence.WEEKLY:
            return self.weekday is not None and target.weekday() == self.weekday
        if self.recurrence == Recurrence.MONTHLY:
            return self.day is not None and target.day == self.day
        if self.recurrence == Recurrence.YEARLY:
            return (
                self.month is not None
                and self.day is not None
                and target.month == self.month
                and target.day == self.day
            )
        return False

    def display_info(self, reference_date: Optional[date] = None) -> str:
        """Generate a human-readable description of the event."""
        parts = [self.title]
        if self.recurrence == Recurrence.YEARLY and self.origin_year and reference_date:
            years = reference_date.year - self.origin_year
            if years > 0:
                parts.append(f"(year {years})")
        if self.due and self.recurrence == Recurrence.ONCE:
            parts.append(f"[{self.due.strftime('%Y-%m-%d %H:%M')}]")
        elif self.time_of_day:
            parts.append(f"[{self.time_of_day}]")
        if self.description:
            parts.append(f"— {self.description}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Tool Reference — declare external tools used by a Skill
# ---------------------------------------------------------------------------

class ToolRef(BaseModel):
    """Skill-level tool declaration stored in the ``tools`` list of config.yaml.

    A tool can come from one of three sources:
    - **script** — inferred automatically from script.py in the same directory
    - **http**   — remote HTTP API
    - **mcp**    — tool exposed by an MCP server

    Example config.yaml format::

        tools:
          - name: weather
            type: http
            endpoint: https://api.weather.com/current
            method: GET
            description: Fetch current weather information

          - name: database
            type: mcp
            mcp_server: localhost:5000
            tool_name: query
            description: Query the database
    """

    name: str = Field(..., description="Tool name")
    type: str = Field(
        default="http",
        description="Tool type: http or mcp",
    )
    description: str = Field(default="", description="Tool description for the LLM")

    # HTTP tool fields
    endpoint: Optional[str] = Field(default=None, description="HTTP API endpoint")
    method: str = Field(default="GET", description="HTTP method")
    headers: Dict[str, str] = Field(default_factory=dict)

    # MCP tool fields
    mcp_server: Optional[str] = Field(default=None, description="MCP server address")
    tool_name: Optional[str] = Field(default=None, description="MCP tool name")
    auth_token: Optional[str] = Field(default=None, description="MCP auth token")


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
        description="Skill name in kebab-case, at most 64 characters, and matching the directory name.",
    )
    description: str = Field(
        default="",
        description="Skill description, up to 1024 characters, explaining the use case and trigger conditions.",
    )
    version: str = "v1.0"
    system_prompt: str = Field(
        default="",
        description="Markdown body of SKILL.md — the LLM system prompt and the target optimized by APO.",
    )
    target: Optional[str] = Field(
        default=None,
        description="One-line optimization target from the user, such as 'more human' or 'more concise'. APO uses this as guidance.",
    )
    few_shot_messages: List[Message] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    script: Optional[str] = Field(
        default=None,
        description="Optional Python source code from script.py that defines custom tool functions.",
    )
    agenda: List[AgendaEntry] = Field(
        default_factory=list,
        description="Long-horizon agenda entries stored in the agenda field of config.yaml and loaded automatically.",
    )
    tools: List[ToolRef] = Field(
        default_factory=list,
        description="External tool declarations for the Skill, stored in the tools field of config.yaml.",
    )


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
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    inputs: List[Message]
    prediction: Message
    feedback: Optional[Feedback] = None
    node_path: Optional[str] = Field(
        default=None,
        description="Dot-separated path of the SkillNode that handled this trace "
        "(e.g. 'social.moments'). Used by APO to route traces to the correct node.",
    )
